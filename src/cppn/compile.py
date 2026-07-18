from collections import defaultdict

import torch
from neat.graphs import feed_forward_layers

from src.cppn.activations import TORCH_ACTIVATIONS
from src.cppn.coords import make_coord_grid, reshape_pattern


def compile_genome(genome, genome_config, coords: torch.Tensor) -> torch.Tensor:
    """Evaluates a NEAT genome over an entire coordinate grid at once as
    batched torch tensor ops, replicating `neat.nn.FeedForwardNetwork`'s
    per-node semantics exactly:
        value = activation(bias + response * sum(weight_i * input_i))
    (aggregation is restricted to 'sum' by the NEAT config — see
    configs/neat/cppn_neat.cfg — so no other neat-python aggregation function
    needs to be mirrored here.)

    neat-python's own `FeedForwardNetwork.activate()` evaluates one input
    vector at a time in pure Python, which is far too slow to evaluate a
    28x28+ coordinate grid across a population every generation; this
    function is the vectorized replacement, cross-checked against
    neat-python's own `activate()` in tests/test_compile_matches_neat.py.

    coords: [N, num_inputs] float tensor.
    Returns: [N, num_outputs] tensor, squashed to [0,1] by an outer sigmoid
    applied independent of whatever activation NEAT evolved at the output
    node (the genome's own output activation may be unbounded, e.g.
    'identity' or 'abs' — don't rely on evolution converging to a bounded
    one for range safety).
    """
    device = coords.device
    n_points = coords.shape[0]

    connections = [cg.key for cg in genome.connections.values() if cg.enabled]
    layers, _required = feed_forward_layers(
        genome_config.input_keys, genome_config.output_keys, connections
    )

    incoming: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for src, dst in connections:
        cg = genome.connections[(src, dst)]
        incoming[dst].append((src, cg.weight))

    values: dict[int, torch.Tensor] = {}
    for idx, input_key in enumerate(genome_config.input_keys):
        values[input_key] = coords[:, idx]

    for layer in layers:
        for node_key in layer:
            node_gene = genome.nodes[node_key]
            conns = [(src, w) for src, w in incoming.get(node_key, []) if src in values]
            if conns:
                agg = sum(values[src] * w for src, w in conns)
            else:
                agg = torch.zeros(n_points, device=device)  # matches sum([]) == 0.0
            z = node_gene.bias + node_gene.response * agg
            activation_fn = TORCH_ACTIVATIONS[node_gene.activation]
            values[node_key] = activation_fn(z)

    zeros = torch.zeros(n_points, device=device)
    outputs = [values.get(out_key, zeros) for out_key in genome_config.output_keys]
    raw = torch.stack(outputs, dim=-1)  # [N, num_outputs]
    return torch.sigmoid(raw)


def genome_to_pattern(
    genome, genome_config, image_size: int, channels: int, device: torch.device
) -> torch.Tensor:
    """Convenience wrapper: compiles a genome once over the full image-resolution
    coordinate grid and reshapes straight to [H, W, C], ready for apply_pattern."""
    coord_grid = make_coord_grid(image_size, image_size, channels, device=device)
    pattern_flat = compile_genome(genome, genome_config, coord_grid)
    return reshape_pattern(pattern_flat, image_size, image_size, channels)
