import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def save_genome(path: str | Path, genome, config=None) -> None:
    with open(path, "wb") as f:
        pickle.dump({"genome": genome, "config": config}, f)


def load_genome(path: str | Path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["genome"], data["config"]


def save_top_k_genomes(path: str | Path, genomes: list) -> None:
    with open(path, "wb") as f:
        pickle.dump(genomes, f)


def load_top_k_genomes(path: str | Path) -> list:
    with open(path, "rb") as f:
        return pickle.load(f)


def genome_summary(genome) -> dict:
    num_enabled = sum(1 for cg in genome.connections.values() if cg.enabled)
    return {
        "genome_id": genome.key,
        "fitness": genome.fitness,
        "num_nodes": len(genome.nodes),
        "num_connections": num_enabled,
        "activations": sorted({ng.activation for ng in genome.nodes.values()}),
    }


def save_pattern(pattern: torch.Tensor, pt_path: str | Path, png_path: str | Path | None = None) -> None:
    torch.save(pattern.detach().cpu(), pt_path)
    if png_path is not None:
        arr = pattern.detach().cpu().numpy()
        fig, ax = plt.subplots(figsize=(3, 3))
        if arr.shape[-1] == 1:
            ax.imshow(arr[..., 0], cmap="gray", vmin=0, vmax=1)
        else:
            ax.imshow(arr, vmin=0, vmax=1)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
