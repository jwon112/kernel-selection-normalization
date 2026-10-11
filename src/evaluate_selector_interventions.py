#!/usr/bin/env python3
import argparse
import csv
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from seg3d.data import BraTSVolumeDataset
from seg3d.model_zoo import build_model
from seg3d.reporting import REGION_BURDEN_FIELDS, REGION_STATUS_FIELDS, full_volume_validation


def atomic_csv_write(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(".tmp")
    with open(temporary, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader(); writer.writerows(rows); file.flush(); os.fsync(file.fileno())
    os.replace(temporary, path)


def intervention_conditions(branches, mode, condition_set="full"):
    if condition_set not in ("core", "competence", "full"):
        raise ValueError(f"Unknown condition_set={condition_set!r}")
    yield {"condition": "learned", "kind": None, "intervention_branch": None, "renormalize": None}
    if mode == "soft":
        yield {"condition": "force_hard", "kind": "force_hard", "intervention_branch": None, "renormalize": True}
    yield {"condition": "force_uniform", "kind": "force_uniform", "intervention_branch": None, "renormalize": True}
    if condition_set == "core":
        return
    for branch in branches:
        yield {"condition": f"force_{branch}", "kind": "force_branch", "intervention_branch": branch, "renormalize": True}
        if condition_set == "competence":
            continue
        if mode == "moe":
            yield {"condition": f"drop_{branch}_without_replacement", "kind": "drop_branch", "intervention_branch": branch, "renormalize": False}
            yield {"condition": f"drop_{branch}_and_reroute", "kind": "drop_branch", "intervention_branch": branch, "renormalize": True}
        else:
            yield {"condition": f"drop_{branch}_without_renormalization", "kind": "drop_branch", "intervention_branch": branch, "renormalize": False}
            yield {"condition": f"drop_{branch}_renormalized", "kind": "drop_branch", "intervention_branch": branch, "renormalize": True}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("fold_dir", type=Path); parser.add_argument("--checkpoint", default="checkpoints/latest.pt", help="final epoch checkpoint by default; pass best.pt only for diagnostic comparison"); parser.add_argument("--output-dir", type=Path); parser.add_argument("--condition-set", choices=("core", "competence", "full"), default="full")
    args = parser.parse_args(); fold_dir = args.fold_dir.resolve(); config = json.loads((fold_dir / "config.json").read_text()); run = json.loads((fold_dir / "run.json").read_text()); fold = int(run["fold"])
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    device = torch.device("cuda"); torch.manual_seed(config["seed"] + fold); torch.cuda.manual_seed_all(config["seed"] + fold)
    model = build_model(config["model"], in_channels=4, out_channels=3, base_channels=config.get("base_channels", 16), **config.get("model_kwargs", {})).to(device)
    if not callable(getattr(model, "set_intervention", None)): raise TypeError(f"Model {config['model']!r} does not support selector interventions")
    state = torch.load(fold_dir / args.checkpoint, map_location=device, weights_only=False); model.load_state_dict(state["model"] if "model" in state else state)
    spec = model.experiment_spec(); branches, mode = spec["branch_names"], spec["mode"]
    validation = BraTSVolumeDataset(config["data_dir"], fold, "val"); loader = DataLoader(validation, batch_size=config["batch_size"], shuffle=False, num_workers=0, pin_memory=True)
    output = args.output_dir or fold_dir / "metrics" / "selector_interventions_final_epoch"; summary, case_rows, selector_rows = [], [], []
    for condition in intervention_conditions(branches, mode, args.condition_set):
        model.clear_intervention()
        if condition["kind"] is not None: model.set_intervention(condition["kind"], branch=condition["intervention_branch"], renormalize=condition["renormalize"])
        loss, dice, records, signals = full_volume_validation(model, loader, device)
        summary.append({**condition, "checkpoint": args.checkpoint, "val_loss": loss, "dice_wt": dice[0], "dice_tc": dice[1], "dice_et": dice[2], "mean_dice": sum(dice) / 3})
        case_rows.extend({**condition, "checkpoint": args.checkpoint, **record} for record in records); selector_rows.extend({**condition, "checkpoint": args.checkpoint, **signal} for signal in signals)
    model.clear_intervention(); conditions = ("condition", "kind", "intervention_branch", "renormalize", "checkpoint")
    atomic_csv_write(output / "summary.csv", conditions + ("val_loss", "dice_wt", "dice_tc", "dice_et", "mean_dice"), summary)
    atomic_csv_write(output / "cases.csv", conditions + ("case_id", *REGION_BURDEN_FIELDS, *REGION_STATUS_FIELDS, "valid_dice_regions", "dice_wt", "dice_tc", "dice_et", "mean_dice"), case_rows)
    atomic_csv_write(output / "selector_signals.csv", conditions + ("case_id", "stage", "mode", "branch", "weight", "dense_probability", "executed", "contribution_rms", "entropy", *REGION_BURDEN_FIELDS), selector_rows)
    print(f"checkpoint={args.checkpoint} conditions={len(summary)} cases={len(case_rows)} output={output}")


if __name__ == "__main__": main()
