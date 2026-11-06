"""TEVC-version additions: Tiny-ImageNet (64x64) support, untagged seed
overrides, and the full-length check of the fitness study's short schedule."""

import json
from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image

from method_paper.scripts.fitness_study import FULL_TAG, SHORT_TAG, collect_runs, full_kd_reference, validate
from method_paper.scripts.launch import build_jobs
from method_paper.scripts.train import prepare_config, run_training
from method_paper.src.data import IMAGE_SIZES, NUM_CLASSES, get_loaders, get_probe_dataset, train_transform
from method_paper.src.models.registry import build_model
from method_paper.src.online_views import OnlineViewEvolver
from method_paper.src.views import build_views
from src.utils.config import load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _tiny_cfg():
    cfg = load_config(CONFIGS / "tinyimagenet.yaml")
    cfg["views"]["neat_config"] = str(REPO_ROOT / "configs" / "neat" / "cppn_neat_smoke.cfg")
    cfg["views"]["probe_batch_size"] = 8
    cfg["views"]["evolution"] = {**cfg["views"]["evolution"], "num_generations": 1, "top_k": 2}
    cfg["views"]["random_search"] = {"max_mutations": 2}
    cfg["online"] = {**cfg["online"], "evolve_every": 1, "initial_generations": 1, "generations_per_round": 1,
                     "top_k": 2}
    return {**cfg, "num_workers": 0}


# --- Tiny-ImageNet data ----------------------------------------------------

def test_tinyimagenet_sizes_and_crop():
    assert IMAGE_SIZES["tinyimagenet"] == 64 and NUM_CLASSES["tinyimagenet"] == 200
    crop = train_transform(dataset="tinyimagenet").transforms[0]
    assert crop.size == (64, 64) and crop.padding == 8
    assert train_transform().transforms[0].padding == 4  # CIFAR default unchanged


def test_tinyimagenet_fake_loader_yields_64px():
    train, _val, test = get_loaders("tinyimagenet", "./data", 4, 0, fake_data=True)
    images, labels = next(iter(train))
    assert images.shape == (4, 3, 64, 64) and labels.max() < 200


def _fake_tinyimagenet(root: Path, classes=("n01", "n02"), per_class=3):
    """Directory layout of the real download, with tiny solid-color JPEGs."""
    lines = []
    for c in classes:
        (root / "train" / c / "images").mkdir(parents=True)
        for i in range(per_class):
            Image.new("RGB", (64, 64), (i * 40, 0, 0)).save(root / "train" / c / "images" / f"{c}_{i}.JPEG")
    (root / "val" / "images").mkdir(parents=True)
    for k, c in enumerate(classes * 2):
        name = f"val_{k}.JPEG"
        Image.new("RGB", (64, 64), (0, k * 30, 0)).save(root / "val" / "images" / name)
        lines.append(f"{name}\t{c}\t0\t0\t63\t63")
    (root / "val" / "val_annotations.txt").write_text("\n".join(lines) + "\n")


def test_staging_sorts_val_images_into_class_folders_and_loads(tmp_path, capsys):
    from method_paper.scripts.stage_tinyimagenet import stage
    from method_paper.src.data import _build_datasets

    root = tmp_path / "tiny-imagenet-200"
    _fake_tinyimagenet(root)
    stage(root)
    assert "PROBLEMS" in capsys.readouterr().out  # 2 classes instead of 200 is reported, not fatal
    assert sorted(p.name for p in (root / "val_by_class").iterdir()) == ["n01", "n02"]
    train_aug, train_plain, test = _build_datasets("tinyimagenet", str(tmp_path), fake_data=False)
    assert len(train_aug) == 6 and len(test) == 4
    assert train_aug.classes == test.classes  # same label order for train and test
    image, _ = test[0]
    assert image.shape == (3, 64, 64)


def test_missing_staging_fails_clearly(tmp_path):
    from method_paper.src.data import _build_datasets

    with pytest.raises(FileNotFoundError, match="stage_tinyimagenet"):
        _build_datasets("tinyimagenet", str(tmp_path), fake_data=False)


# --- views at 64x64 ----------------------------------------------------------

@pytest.mark.parametrize("source", ["random", "random_search"])
def test_offline_views_are_compiled_at_64px(tmp_path, source):
    cfg = _tiny_cfg()
    patterns = build_views(source, build_model("resnet8", 200).eval(), "tinyimagenet",
                           get_probe_dataset("tinyimagenet", "./data", fake_data=True), cfg["views"], 0,
                           torch.device("cpu"), tmp_path)
    assert all(p.shape == (64, 64, 3) for p in patterns)


@pytest.mark.parametrize("family", ["cppn", "noise"])
def test_online_views_are_compiled_at_64px(tmp_path, family):
    cfg = _tiny_cfg()
    cfg["online"]["family"] = family
    evolver = OnlineViewEvolver(
        build_model("resnet8", 200).eval(), "tinyimagenet", get_probe_dataset("tinyimagenet", "./data", fake_data=True),
        cfg["views"]["neat_config"], "disagreement", cfg["online"], cfg["views"]["evolution"], "multiplicative",
        0.5, 8, 4.0, 0, torch.device("cpu"), tmp_path, max_mutations=2,
    )
    assert all(p.shape == (64, 64, 3) for p in evolver.evolve(build_model("resnet8", 200), 1, epoch=0))


@pytest.mark.parametrize("mode", ["kd_evolved_cppn", "kd_online_disagree"])
def test_tinyimagenet_end_to_end_on_fake_data(tmp_path, mode):
    cfg = _tiny_cfg()
    common = dict(results_root=tmp_path, fake_data=True, max_batches=1, epochs=1)
    run_training(cfg, "teacher", "resnet8", cfg["teacher_seed"], **common)
    run_dir = run_training(cfg, "student", "resnet8", 0, mode=mode, teacher="resnet8", **common)
    assert run_dir.parent.parent.name == "tinyimagenet"
    assert json.loads((run_dir / "summary.json").read_text())["mode"] == mode


def test_tinyimagenet_stage_sizes():
    cfg = load_config(CONFIGS / "tinyimagenet.yaml")
    assert len(build_jobs(cfg, "teachers")) == 1
    assert len(build_jobs(cfg, "students")) == 6      # student_only + kd, 3 seeds
    assert len(build_jobs(cfg, "phase1")) == 9        # random view / evolved / random search
    assert len(build_jobs(cfg, "phase2")) == 6        # online teacher fitness / disagreement


# --- extra seeds and the short-schedule check ---------------------------------

def test_seed_overrides_need_no_tag_and_only_add_new_seeds():
    cfg = prepare_config(str(CONFIGS / "cifar100.yaml"), ["student_seeds=[3, 4]"], None)
    jobs = build_jobs(cfg, "phase1")
    assert {j["seed"] for j in jobs} == {3, 4} and len(jobs) == 12
    with pytest.raises(ValueError):
        prepare_config(str(CONFIGS / "cifar100.yaml"), ["online.gate.tau_low=0.5"], None)


def test_fitness_validation_stage():
    cfg = load_config(CONFIGS / "cifar100.yaml")
    jobs = build_jobs(cfg, "fitness_validation")
    fs = cfg["fitness_study"]
    assert len(jobs) == len(fs["validate_indices"]) * len(fs["validate_seeds"]) == 20  # fitness_validation.sbatch
    assert all(j["mode"] == "kd_bank_view" and FULL_TAG.fullmatch(j["tag"]) for j in jobs)
    assert all("training.epochs" not in " ".join(j["overrides"]) for j in jobs)  # full schedule


def test_short_and_full_runs_are_collected_separately(tmp_path):
    for tag, seed, top1 in (("fs_kd", 0, 60.0), ("fs03", 0, 59.0), ("fsfull03", 0, 70.0)):
        d = tmp_path / f"run_{tag}_{seed}"
        d.mkdir()
        (d / "summary.json").write_text(json.dumps({"tag": tag, "seed": seed, "final_test_top1": top1}))
    assert sorted(collect_runs(tmp_path).tag) == ["fs03", "fs_kd"]
    assert list(collect_runs(tmp_path, FULL_TAG).tag) == ["fsfull03"]
    assert SHORT_TAG.fullmatch("fsfull03") is None


def test_validate_correlates_short_and_full_effects(tmp_path):
    per = pd.DataFrame({"index": range(6), "source": ["evolved"] * 3 + ["random"] * 3,
                        "fitness": [0.1, 0.2, 0.3, 0.0, 0.05, 0.02], "diversity": [0.1, 0.2, 0.3, 0.0, 0.05, 0.02],
                        "agreement": [0.8] * 6, "delta_vs_kd": [-1.0, -0.6, -0.2, 0.1, 0.3, -0.4]})
    full = pd.DataFrame([{"tag": f"fsfull{i:02d}", "seed": s, "top1": 70.0 + d + 0.05 * s}
                         for i, d in enumerate(per.delta_vs_kd) for s in (0, 1)])
    kd = pd.Series([70.0, 70.1])
    table, stats = validate(per, full, kd)
    assert len(table) == 6 and stats["spearman_rho"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        validate(per, full, pd.Series([], dtype=float))

    d = tmp_path / "resnet56__resnet20__kd__seed0"
    d.mkdir()
    (d / "summary.json").write_text(json.dumps({"final_test_top1": 70.5}))
    assert list(full_kd_reference(tmp_path, "resnet56", "resnet20")) == [70.5]
