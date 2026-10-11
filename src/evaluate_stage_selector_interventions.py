#!/usr/bin/env python3
"""Evaluate branch drops at one decoder selector stage at a time."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from seg3d.data import BraTSVolumeDataset
from seg3d.model_zoo import build_model
from seg3d.reporting import REGION_BURDEN_FIELDS, REGION_STATUS_FIELDS, full_volume_validation


def atomic_csv_write(path: Path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with open(temporary, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fold_dir", type=Path)
    parser.add_argument("--checkpoint", default="checkpoints/latest.pt")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument(
        "--intervention",
        choices=("drop", "keep"),
        default="drop",
        help="Drop one branch or keep only one branch at the selected stage.",
    )
    parser.add_argument(
        "--renormalize",
        action="store_true",
        help="Redistribute dropped branch mass over the remaining branches.",
    )
    args = parser.parse_args()
    if args.intervention == "keep" and args.renormalize:
        parser.error("--renormalize applies only to --intervention drop")

    cpu_threads = int(os.environ.get("SEG3D_CPU_THREADS", "8"))
    if not 1 <= cpu_threads <= 8:
        raise ValueError(
            f"SEG3D_CPU_THREADS must be between 1 and 8, got {cpu_threads}"
        )
    torch.set_num_threads(cpu_threads)
    torch.set_num_interop_threads(1)

    fold_dir = args.fold_dir.resolve()
    config = json.loads((fold_dir / "config.json").read_text())
    run = json.loads((fold_dir / "run.json").read_text())
    fold = int(run["fold"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    device = torch.device("cuda")
    model = build_model(
        config["model"],
        in_channels=4,
        out_channels=3,
        base_channels=config.get("base_channels", 16),
        **config.get("model_kwargs", {}),
    ).to(device)
    selectors = getattr(model, "multi_kernel_selectors", None)
    stage_indices = tuple(getattr(model, "selector_decoder_indices", ()))
    if selectors is None or len(selectors) < 2:
        raise TypeError("Stage-specific intervention requires a multi-stage selector model")
    if len(selectors) != len(stage_indices):
        raise RuntimeError("Selector/stage index count mismatch")

    state = torch.load(fold_dir / args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    branches = tuple(selectors[0].branch_names)
    if any(tuple(selector.branch_names) != branches for selector in selectors):
        raise RuntimeError("All selector stages must share branch names")

    validation = BraTSVolumeDataset(config["data_dir"], fold, "val")
    if args.max_cases is not None:
        validation = Subset(validation, range(min(args.max_cases, len(validation))))
    loader = DataLoader(
        validation,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    if args.intervention == "keep":
        default_output = "stage_selector_keep_one_final_epoch"
    elif args.renormalize:
        default_output = "stage_selector_interventions_reroute_final_epoch"
    else:
        default_output = "stage_selector_interventions_final_epoch"
    output = args.output_dir or fold_dir / "metrics" / default_output
    summary, cases = [], []

    conditions = [(None, None)]
    conditions.extend(
        (stage_position, branch)
        for stage_position in range(len(selectors))
        for branch in branches
    )
    for stage_position, branch in conditions:
        model.clear_intervention()
        if stage_position is None:
            condition = "learned"
            decoder_index = None
        else:
            decoder_index = int(stage_indices[stage_position])
            if args.intervention == "keep":
                selectors[stage_position].set_intervention(
                    "force_branch", branch=branch
                )
                condition = f"keep_decoder_{decoder_index}_{branch}"
            else:
                selectors[stage_position].set_intervention(
                    "drop_branch", branch=branch, renormalize=args.renormalize
                )
                suffix = "_and_reroute" if args.renormalize else ""
                condition = f"drop_decoder_{decoder_index}_{branch}{suffix}"
        loss, dice, records, _ = full_volume_validation(model, loader, device)
        row = {
            "condition": condition,
            "stage_position": stage_position,
            "decoder_index": decoder_index,
            "branch": branch,
            "checkpoint": args.checkpoint,
            "intervention_kind": args.intervention,
            "renormalize": args.renormalize,
            "val_loss": loss,
            "dice_wt": dice[0],
            "dice_tc": dice[1],
            "dice_et": dice[2],
            "mean_dice": sum(dice) / 3,
        }
        summary.append(row)
        case_context = {
            key: row[key] for key in (
                "condition", "stage_position", "decoder_index", "branch", "checkpoint",
                "intervention_kind", "renormalize",
            )
        }
        cases.extend({**case_context, **record} for record in records)

    model.clear_intervention()
    base_fields = (
        "condition", "stage_position", "decoder_index", "branch", "checkpoint",
        "intervention_kind", "renormalize",
    )
    atomic_csv_write(
        output / "summary.csv",
        base_fields + ("val_loss", "dice_wt", "dice_tc", "dice_et", "mean_dice"),
        summary,
    )
    atomic_csv_write(
        output / "cases.csv",
        base_fields
        + ("case_id", *REGION_BURDEN_FIELDS, *REGION_STATUS_FIELDS,
           "valid_dice_regions", "dice_wt", "dice_tc", "dice_et", "mean_dice"),
        cases,
    )
    print(
        f"fold={fold} conditions={len(summary)} cases={len(cases)} output={output}"
    )


if __name__ == "__main__":
    main()
