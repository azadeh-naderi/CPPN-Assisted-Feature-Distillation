import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import Subset

from src.cppn.compile import genome_to_pattern
from src.cppn.evolve import run_evolution
from src.cppn.serialize import genome_summary, save_genome, save_pattern, save_top_k_genomes
from src.data.datasets import load_dataset
from src.models.registry import build_model
from src.utils.config import load_config
from src.utils.logging import RunLogger, get_logger
from src.utils.seed import set_seed

log = get_logger("evolve_cppn")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--teacher-ckpt", required=True)
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["seed"]
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _train_ds, eval_ds = load_dataset(cfg["dataset_name"], cfg.get("data_root", "./data"))
    if args.smoke and "smoke_subset_size" in cfg:
        n = min(cfg["smoke_subset_size"], len(eval_ds))
        eval_ds = Subset(eval_ds, range(n))

    teacher = build_model(cfg["model_name"], cfg["input_channels"], cfg["num_classes"], cfg.get("pretrained", False))
    teacher.load_state_dict(torch.load(args.teacher_ckpt, map_location=device))
    teacher = teacher.to(device)

    cppn_cfg = cfg["cppn"]
    num_generations = args.generations or cppn_cfg["num_generations"]

    run_id = f"{cfg['dataset_name']}_{cfg['model_name']}_cppn_{seed}_{int(time.time())}"
    run_dir = Path("results/cppn_genomes") / run_id
    run_logger = RunLogger(run_dir)
    run_logger.log_config(cfg)

    best_genome, neat_config, _evo_log_df, _gen_summary_df, top_k = run_evolution(
        teacher=teacher,
        probe_dataset=eval_ds,
        neat_config_path=cppn_cfg["neat_config"],
        image_size=cfg["image_size"],
        channels=cfg["input_channels"],
        view_op=cppn_cfg["view_op"],
        view_scale=cppn_cfg["view_scale"],
        tau_low=cppn_cfg["tau_low"],
        tau_high=cppn_cfg["tau_high"],
        gamma=cppn_cfg["gamma"],
        probe_batch_size=cppn_cfg["probe_batch_size"],
        num_generations=num_generations,
        seed=cppn_cfg.get("random_seed", seed),
        device=device,
        log_dir=run_dir,
        top_k=cppn_cfg["top_k"],
    )

    save_genome(run_dir / "best_genome.pkl", best_genome, neat_config)
    save_top_k_genomes(run_dir / "top_k_genomes.pkl", top_k)
    pattern = genome_to_pattern(
        best_genome, neat_config.genome_config, cfg["image_size"], cfg["input_channels"], device
    )
    save_pattern(pattern, run_dir / "pattern.pt", run_dir / "pattern.png")

    run_logger.log_summary(
        best_fitness=best_genome.fitness,
        genome_summary=genome_summary(best_genome),
        num_generations=num_generations,
        smoke=args.smoke,
    )
    log.info("best genome fitness: %.4f", best_genome.fitness)
    print(run_dir)


if __name__ == "__main__":
    main()
