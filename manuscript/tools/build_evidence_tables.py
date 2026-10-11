from __future__ import annotations


import argparse


import csv


import hashlib


import io


import json


import math


from collections import defaultdict


from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


REGIONS = ("wt", "tc", "et")


THREE = (
    ("Canonical U-Net", "BraTS23GLICanonicalUNetTrainer", "brats23_gli_canonical_unet_crop4.json"),
    ("Attention-derived fixed", "BraTS23GLIMKFixedScheduleAttention100Trainer", "brats23_gli_mk_fixed_schedule_attention_100.json"),
    ("Drop-derived fixed", "BraTS23GLIMKFixedScheduleDrop100Trainer", "brats23_gli_mk_fixed_schedule_drop_100.json"),
    ("Multi-stage uniform", "BraTS23GLIMKMultiStageUniform100Trainer", "brats23_gli_mk_multistage_uniform_100.json"),
    ("Multi-stage SK", "BraTS23GLIMKMultiStageSK100Trainer", "brats23_gli_mk_multistage_sk_100.json"),
)


FIVE = (
    ("Canonical U-Net", "BraTS23GLICanonicalUNetTrainer", "brats23_gli_canonical_unet_crop4.json"),
    ("Uniform multi-kernel", "BraTS23GLIMKUniformTrainer", "brats23_gli_mk_uniform_crop4.json"),
    ("SK multi-kernel", "BraTS23GLIMKSKTrainer", "brats23_gli_mk_sk_crop4.json"),
    ("Fixed $3^3$", "BraTS23GLIMKFixedK3Trainer", "brats23_gli_mk_fixed_k3_crop4.json"),
    ("Fixed $5^3$", "BraTS23GLIMKFixedK5Trainer", "brats23_gli_mk_fixed_k5_crop4.json"),
    ("Fixed dilated $3^3$", "BraTS23GLIMKFixedDilatedK3Trainer", "brats23_gli_mk_fixed_dilated_k3_crop4.json"),
    ("Fixed anisotropic", "BraTS23GLIMKFixedAnisotropicTrainer", "brats23_gli_mk_fixed_anisotropic_crop4.json"),
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finite(value: str, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite {name}: {value}")
    return number


def present(value: str) -> bool:
    if value not in ("True", "False"):
        raise ValueError(f"Unexpected GT-positive flag: {value}")
    return value == "True"


def region_macro(rows: list[dict[str, str]]) -> float:
    means = []
    for region in REGIONS:
        scores = [finite(row[f"dice_{region}"], f"dice_{region}") for row in rows
                  if present(row[f"gt_positive_{region}"])]
        if not scores or any(not 0 <= score <= 1 for score in scores):
            raise ValueError(f"Invalid positive-GT {region} scores")
        means.append(sum(scores) / len(scores))
    return sum(means) / len(means)


def read_csv(path: Path, inputs: dict) -> list[dict[str, str]]:
    inputs[str(path.relative_to(ROOT))] = sha(path)
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    return rows


def read_json(path: Path, inputs: dict):
    inputs[str(path.relative_to(ROOT))] = sha(path)
    return json.loads(path.read_text())


def config_differences(source: dict, actual: dict) -> dict:
    """Preserve non-path changes rather than silently treating a source config as run truth."""
    return {key: {"source": source.get(key), "run": actual.get(key)}
            for key in sorted(set(source) | set(actual))
            if key not in ("data_dir", "results_root", "output_dir")
            and source.get(key) != actual.get(key)}


def brats_family(name: str, trainer: str, config_name: str, folds: range, inputs: dict) -> dict:
    config_path = ROOT / "configs" / config_name
    config = read_json(config_path, inputs)
    if config["trainer"] != trainer:
        raise ValueError(f"Trainer/config mismatch: {config_path}")
    scores, flops, counts, capacities, fold_details = [], [], [], [], []
    for fold in folds:
        base = ROOT / "results/brats2023_gli" / trainer / f"fold_{fold}"
        rows = read_csv(base / "metrics/fold_validation_final.csv", inputs)
        summaries = [row for row in rows if row["case_id"].startswith("__")]
        if len(summaries) != 1 or summaries[0]["case_id"] != "__mean_positive_gt__":
            raise ValueError(f"Unexpected BraTS summary rows: {base}")
        rows = [row for row in rows if not row["case_id"].startswith("__")]
        actual_config = read_json(base / "config.json", inputs)
        run = read_json(base / "run.json", inputs)
        if actual_config["trainer"] != trainer or run["fold"] != fold:
            raise ValueError(f"Run identity mismatch: {base}")
        if any(row["checkpoint"] != "checkpoints/latest.pt" or int(row["fold"]) != fold for row in rows):
            raise ValueError(f"Nonterminal or wrong-fold metric: {base}")
        ids = [row["case_id"] for row in rows]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate study ID: {base}")
        if not (base / "checkpoints/latest.pt").exists():
            raise ValueError(f"Missing terminal checkpoint: {base}")
        fs = {finite(row["active_flops_per_volume"], "active_flops_per_volume") for row in rows}
        ps = {int(row["parameters"]) for row in rows}
        if len(fs) != 1 or len(ps) != 1:
            raise ValueError(f"Within-fold capacity/profile varies: {base}")
        score = region_macro(rows)
        scores.append(score)
        flops.extend(fs)
        capacities.extend(ps)
        counts.append(len(rows))
        fold_details.append({"fold": fold, "studies": len(rows), "dice": score,
                             "active_flops": next(iter(fs)), "parameters": next(iter(ps)),
                             "config_sha256": inputs[str((base / 'config.json').relative_to(ROOT))],
                             "run_config_differences": config_differences(config, actual_config)})
    if len(set(capacities)) != 1:
        raise ValueError(f"Parameter count differs by fold: {trainer}")
    return {"name": name, "trainer": trainer, "config": str(config_path.relative_to(ROOT)),
            "model": config["model"], "folds": fold_details, "dice": sum(scores) / len(scores),
            "active_tflops_range": [min(flops) / 1e12, max(flops) / 1e12],
            "parameters": capacities[0], "study_counts": counts,
            "settings": {key: config.get(key) for key in
                         ("model", "input_shape", "batch_size", "epochs", "learning_rate",
                          "weight_decay", "augmentation", "precision", "checkpoint_selection",
                          "seed", "model_kwargs", "selector_decoder_indices")}}

