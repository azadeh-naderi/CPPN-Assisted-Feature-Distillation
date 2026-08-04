"""Visualize what an evolved CPPN's view actually looks like when applied to
real images -- i.e. exactly what the teacher/student see during the
kd_evolved_cppn consistency loss, not just the raw pattern.png mask.

Usage:
    python scripts/visualize_evolved_view.py --config configs/datasets/cifar10_resnet18.yaml \
        --genome-path results/cppn_genomes/<run_id>/best_genome.pkl \
        --output results/cppn_genomes/<run_id>/view_examples.png

    # or for the ensemble (shows each top-k genome's view separately):
    python scripts/visualize_evolved_view.py --config configs/datasets/cifar10_resnet18.yaml \
        --ensemble-genomes results/cppn_genomes/<run_id>/top_k_genomes.pkl \
        --output results/cppn_genomes/<run_id>/view_examples_ensemble.png
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import torch

from src.cppn.apply import apply_pattern
from src.cppn.compile import genome_to_pattern
from src.cppn.serialize import genome_summary, load_genome, load_top_k_genomes
from src.data.datasets import get_probe_batch, load_dataset
from src.utils.config import load_config


def to_imshow(img_chw: torch.Tensor):
    """[C,H,W] raw-[0,1] tensor -> imshow-ready array, handling grayscale vs RGB."""
    arr = img_chw.permute(1, 2, 0).clamp(0, 1).numpy()
    if arr.shape[-1] == 1:
        return arr[..., 0], "gray"
    return arr, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--genome-path", default=None, help="single genome .pkl (best_genome.pkl)")
    parser.add_argument("--ensemble-genomes", default=None, help="top_k_genomes.pkl, shows each member")
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if not args.genome_path and not args.ensemble_genomes:
        raise ValueError("Pass --genome-path or --ensemble-genomes")

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the genome(s) + patterns
    if args.ensemble_genomes:
        top_k = load_top_k_genomes(args.ensemble_genomes)
        genomes = [g for _fitness, g in top_k]
        # neat_config isn't stored alongside top_k_genomes.pkl -- reuse the
        # one saved with best_genome.pkl if given, else the cppn config file.
        if args.genome_path:
            _best, neat_config = load_genome(args.genome_path)
        else:
            from src.cppn.evolve import load_neat_config

            neat_config = load_neat_config(cfg["cppn"]["neat_config"])
        labels = [f"evolved #{i} (fitness={g.fitness:.4f})" for i, g in enumerate(genomes)]
    else:
        genome, neat_config = load_genome(args.genome_path)
        genomes = [genome]
        labels = [f"evolved best (fitness={genome.fitness:.4f})"]

    patterns = [
        genome_to_pattern(g, neat_config.genome_config, cfg["image_size"], cfg["input_channels"], device)
        for g in genomes
    ]
    for g, label in zip(genomes, labels):
        print(label, genome_summary(g))

    # Load a handful of real raw-[0,1] images from the dataset
    _train_ds, eval_ds = load_dataset(cfg["dataset_name"], cfg.get("data_root", "./data"))
    images, labels_idx = get_probe_batch(eval_ds, args.num_samples, seed=args.seed, device=device)

    view_op = cfg.get("cppn", {}).get("view_op", "multiplicative")
    view_scale = cfg.get("cppn", {}).get("view_scale", 0.5)

    n_rows = args.num_samples
    n_cols = 1 + len(patterns)  # original + one column per genome's view
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.2 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    for row in range(n_rows):
        img, cmap = to_imshow(images[row].cpu())
        axes[row, 0].imshow(img, cmap=cmap)
        axes[row, 0].axis("off")
        if row == 0:
            axes[row, 0].set_title("original", fontsize=9)

        for col, (pattern, label) in enumerate(zip(patterns, labels), start=1):
            view = apply_pattern(images[row : row + 1], pattern, mode=view_op, scale=view_scale)
            view_img, view_cmap = to_imshow(view[0].cpu())
            axes[row, col].imshow(view_img, cmap=view_cmap)
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(label, fontsize=8)

    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
