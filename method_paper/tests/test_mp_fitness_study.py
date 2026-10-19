import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from method_paper.scripts.fitness_study import analyze, collect_runs, plot
from method_paper.scripts.launch import build_jobs
from method_paper.scripts.train import run_dir_for, run_training
from method_paper.src.data import get_probe_dataset
from method_paper.src.fitness_study import bank_dir, build_bank, load_bank_patterns, stratified_pick
from method_paper.src.models.registry import build_model
from src.cppn.fitness import DISQUALIFIED_FITNESS
from src.utils.config import apply_overrides, load_config

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "cifar100.yaml"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _smoke_cfg():
    cfg = load_config(CONFIG_PATH)
    cfg["views"]["neat_config"] = str(REPO_ROOT / "configs" / "neat" / "cppn_neat_smoke.cfg")
    cfg["views"]["probe_batch_size"] = 8
    cfg["views"]["random_search"] = {"max_mutations": 3}
    cfg["fitness_study"] = {**cfg["fitness_study"], "num_genomes": 6, "generations": 2, "rescore_probe_size": 16}
    return {**cfg, "num_workers": 0}


def test_stratified_pick_dedupes_skips_disqualified_and_spans_the_range():
    g = lambda key: SimpleNamespace(key=key)  # noqa: E731
    pool = [(0.1, g(1)), (0.9, g(1)),          # same genome re-evaluated: keep the latest (0.9)
            (DISQUALIFIED_FITNESS, g(2))]       # disqualified: never picked
    pool += [(float(i), g(10 + i)) for i in range(10)]
    picks = stratified_pick(pool, 3)
    assert [f for f, _ in picks] == [0.0, 4.0, 9.0]  # quantiles of the 11 valid genomes (0.9 is one of them)
    assert all(gen.key != 2 for _, gen in stratified_pick(pool, 20))
    assert [f for f, gen in stratified_pick(pool, 20) if gen.key == 1] == [0.9]


def test_build_bank_on_fake_data(tmp_path):
    cfg = _smoke_cfg()
    table = build_bank(build_model("resnet8", 100).eval(), "cifar100",
                       get_probe_dataset("cifar100", "./data", fake_data=True), cfg["views"], 6, 2, 16, 0,
                       torch.device("cpu"), tmp_path)
    assert set(table.source) <= {"evolved", "random"} and 2 <= len(table) <= 6
    assert {"fitness", "search_fitness", "rel_teacher_acc", "pattern_mean", "pattern_std"} <= set(table.columns)
    patterns = torch.load(tmp_path / "patterns.pt")
    assert patterns.shape == (len(table), 32, 32, 3)
    assert (tmp_path / "bank.png").exists()


def test_fitness_study_stage_matches_the_sbatch_array():
    cfg = load_config(CONFIG_PATH)
    fs = cfg["fitness_study"]
    jobs = build_jobs(cfg, "fitness_study")
    assert len(jobs) == len(fs["kd_seeds"]) + fs["num_genomes"] * len(fs["seeds"]) == 83  # --array=0-82
    assert [j["tag"] for j in jobs[:3]] == ["fs_kd"] * 3 and jobs[0]["mode"] == "kd"
    assert jobs[3]["mode"] == "kd_bank_view" and jobs[3]["tag"] == "fs00"
    assert "fitness_study.index=0" in jobs[3]["overrides"]
    assert apply_overrides(cfg, jobs[3]["overrides"])["training"]["lr_milestones"] == fs["lr_milestones"]


def test_short_runs_end_to_end_then_collect(tmp_path):
    cfg = _smoke_cfg()
    common = dict(results_root=tmp_path, fake_data=True, max_batches=1, epochs=1)
    run_training(cfg, "teacher", "resnet56", cfg["teacher_seed"], **common)
    teacher = build_model("resnet56", 100)
    teacher.load_state_dict(torch.load(run_dir_for(tmp_path, "cifar100", "teacher", "resnet56", "ce", 0)
                                       / "checkpoint.pt"))
    build_bank(teacher, "cifar100", get_probe_dataset("cifar100", "./data", fake_data=True), cfg["views"], 4, 1, 16,
               0, torch.device("cpu"), bank_dir(tmp_path, "cifar100"))
    assert load_bank_patterns(tmp_path, "cifar100").shape[1:] == (32, 32, 3)

    with pytest.raises(ValueError):  # bank view without an index
        run_training(cfg, "student", "resnet8", 0, mode="kd_bank_view", teacher="resnet56", tag="x", **common)

    for job in build_jobs(cfg, "fitness_study")[:4]:  # kd reference x 3 + first bank view
        run_training(apply_overrides(cfg, job["overrides"]), job["role"], job["arch"], job["seed"], job["mode"],
                     job["teacher"], tag=job["tag"], **common)
    runs = collect_runs(tmp_path / "cifar100" / "students")
    assert sorted(runs.tag) == ["fs00", "fs_kd", "fs_kd", "fs_kd"]


def test_partial_spearman_removes_a_shared_cause():
    import numpy as np

    from method_paper.scripts.fitness_study import partial_spearman

    rng = np.random.default_rng(0)
    z = rng.normal(size=200)
    x = z + 0.3 * rng.normal(size=200)       # x tracks z
    y = z + 0.3 * rng.normal(size=200)       # y depends on z only
    rho, p = partial_spearman(x, y, z)
    assert abs(rho) < 0.2 and p > 0.01        # no association once z is held fixed
    y2 = x + 0.1 * rng.normal(size=200)      # y depends on x beyond z
    assert partial_spearman(x, y2, z)[0] > 0.8


def test_pool_summary_compares_evolved_and_random():
    from method_paper.scripts.fitness_study import pool_summary

    per = pd.DataFrame({"source": ["evolved"] * 4 + ["random"] * 4,
                        "delta_vs_kd": [-3.0, -3.2, -2.8, -3.1, -1.0, -1.2, -0.9, -1.1]})
    summary = pool_summary(per)
    means = dict(zip(summary.source, summary["mean"]))
    assert means["evolved"] < means["random"] and summary.attrs["welch_p"] < 0.01


def test_analyze_recovers_a_known_negative_correlation(tmp_path):
    n = 8
    bank = pd.DataFrame({
        "index": range(n), "source": ["evolved"] * 4 + ["random"] * 4,
        "fitness": [float(i) for i in range(n)],
        **{c: [float((i * 3) % n) for i in range(n)] for c in
           ("diversity", "agreement", "top1_agreement", "rel_teacher_acc", "teacher_view_acc", "pattern_mean",
            "pattern_std", "channel_divergence", "num_connections")},
    })
    runs = pd.DataFrame(
        [{"tag": "fs_kd", "seed": s, "top1": 60.0 + 0.1 * s} for s in range(3)]
        + [{"tag": f"fs{i:02d}", "seed": s, "top1": 60.0 - i + 0.05 * s} for i in range(n) for s in range(2)]
    )
    per, corr, ref = analyze(bank, runs)
    assert ref["kd_n"] == 3 and len(per) == n
    fit_all = corr[(corr.subset == "all") & (corr.predictor == "fitness")].iloc[0]
    assert fit_all.spearman_rho == pytest.approx(-1.0) and fit_all.ci_high <= -0.9
    assert set(corr.subset) == {"all", "evolved", "random"}
    plot(per, ref, tmp_path / "fig.png")
    assert (tmp_path / "fig.png").exists()
