import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from seg3d.losses import dice_bce_loss


REGION_NAMES = ("wt", "tc", "et")
REGION_BURDEN_FIELDS = tuple(field for region in REGION_NAMES for field in (f"target_voxels_{region}", f"target_fraction_{region}"))
REGION_STATUS_FIELDS = tuple(field for region in REGION_NAMES for field in (f"gt_positive_{region}", f"predicted_positive_{region}", f"predicted_voxels_{region}", f"empty_gt_false_positive_{region}"))


def target_region_burden(target_case):
    if target_case.ndim != 4 or target_case.shape[0] != len(REGION_NAMES):
        raise ValueError(f"Expected one target with shape [3, D, H, W], got {tuple(target_case.shape)}")
    spatial_voxels = target_case[0].numel(); counts = (target_case > .5).reshape(3, -1).sum(1).cpu()
    return {field: value for region, count in zip(REGION_NAMES, counts) for field, value in ((f"target_voxels_{region}", int(count)), (f"target_fraction_{region}", int(count) / spatial_voxels))}


def case_region_metrics(logits_case, target_case):
    """Positive-reference Dice plus explicit empty-reference false-positive state."""
    prediction = logits_case.sigmoid() >= .5; truth = target_case > .5
    metrics, valid_dice = {}, []
    for index, region in enumerate(REGION_NAMES):
        truth_voxels = int(truth[index].sum().item()); predicted_voxels = int(prediction[index].sum().item())
        positive = truth_voxels > 0
        if positive:
            intersection = int((prediction[index] & truth[index]).sum().item())
            dice = 2 * intersection / max(predicted_voxels + truth_voxels, 1)
            valid_dice.append(dice)
        else:
            dice = None
        metrics.update({f"dice_{region}": dice, f"gt_positive_{region}": positive, f"predicted_positive_{region}": predicted_voxels > 0, f"predicted_voxels_{region}": predicted_voxels, f"empty_gt_false_positive_{region}": (not positive) and predicted_voxels > 0})
    metrics["mean_dice"] = sum(valid_dice) / len(valid_dice) if valid_dice else None
    metrics["valid_dice_regions"] = len(valid_dice)
    return metrics


@torch.no_grad()
def full_volume_validation(model, loader, device):
    """Case-wise positive-GT Dice; empty GT classes are reported separately."""
    model.eval(); records = []; selector_records = []; loss_sum = 0.; cases = 0
    positive_sum = {region: 0. for region in REGION_NAMES}; positive_count = {region: 0 for region in REGION_NAMES}
    for image, target, case_ids in loader:
        image = image.to(device, dtype=torch.bfloat16, non_blocking=True); target = target.to(device, dtype=torch.float32, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16): logits = model(image); loss = dice_bce_loss(logits, target)
        snapshot_fn = getattr(model, "selection_snapshot", None); snapshot = snapshot_fn() if callable(snapshot_fn) else None
        loss_sum += loss.item() * len(case_ids); cases += len(case_ids)
        for index, case_id in enumerate(case_ids):
            metrics = case_region_metrics(logits[index], target[index]); burden = target_region_burden(target[index])
            record = {"case_id": case_id, **metrics, **burden}; records.append(record)
            for region in REGION_NAMES:
                dice = metrics[f"dice_{region}"]
                if dice is not None: positive_sum[region] += dice; positive_count[region] += 1
            if snapshot is not None:
                snapshots = [("aggregate", snapshot)]
                snapshots.extend(
                    (f"decoder_{stage}", stage_snapshot)
                    for stage, stage_snapshot in zip(
                        snapshot.get("stage_indices", ()),
                        snapshot.get("stage_snapshots", ()),
                    )
                )
                for stage, stage_snapshot in snapshots:
                    weights = stage_snapshot["weights"]; contributions = stage_snapshot["contribution_rms"]; probabilities = stage_snapshot.get("dense_probabilities", weights); selected = stage_snapshot.get("selected_mask")
                    for branch_index, branch in enumerate(stage_snapshot["branch_names"]):
                        selector_records.append({"case_id": case_id, "stage": stage, "mode": stage_snapshot["mode"], "branch": branch, "weight": float(weights[index, branch_index].cpu()), "dense_probability": float(probabilities[index, branch_index].cpu()), "executed": bool(True if selected is None else selected[index, branch_index].cpu()), "contribution_rms": float(contributions[index, branch_index].cpu()), "entropy": float(stage_snapshot["entropy"][index].cpu()), **burden})
    means = [positive_sum[region] / max(positive_count[region], 1) for region in REGION_NAMES]
    return loss_sum / max(cases, 1), means, records, selector_records


def append_epoch_jsonl(path: Path, record):
    with open(path, "a") as file:
        file.write(json.dumps(record, sort_keys=True) + "\n"); file.flush(); os.fsync(file.fileno())


def read_epoch_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def update_training_curves(history_path: Path, output_dir: Path):
    rows = read_epoch_jsonl(history_path)
    if not rows: return
    output_dir.mkdir(parents=True, exist_ok=True); epochs = [row["epoch"] for row in rows]
    specs = (("loss_curve.png", (("train_loss", "Train"), ("val_loss", "Validation")), "Train / Validation Loss", "Dice + BCE", None), ("dice_curve.png", (("dice_wt", "WT"), ("dice_tc", "TC"), ("dice_et", "ET"), ("mean_dice", "Mean")), "Validation Dice (positive-GT cases)", "Dice", (0, 1)))
    for filename, fields, title, ylabel, ylim in specs:
        figure, axis = plt.subplots(figsize=(10, 6))
        for key, label in fields: axis.plot(epochs, [row[key] for row in rows], label=label, linewidth=2)
        axis.set(title=title, xlabel="Epoch", ylabel=ylabel, ylim=ylim); axis.legend(); axis.grid(alpha=.25); figure.tight_layout()
        temporary = output_dir / f"{filename}.tmp"; figure.savefig(temporary, format="png", dpi=170); plt.close(figure); os.replace(temporary, output_dir / filename)


def _positive_region_mean(records, region):
    values = [record[f"dice_{region}"] for record in records if record[f"dice_{region}"] is not None]
    return sum(values) / max(len(values), 1)


def _empty_fp_rate(records, region):
    empty = [record for record in records if not record[f"gt_positive_{region}"]]
    return sum(record[f"empty_gt_false_positive_{region}"] for record in empty) / max(len(empty), 1)


def write_final_fold_csv(path: Path, records, run_id, model_name, fold, checkpoint, complexity):
    fields = ("run_id", "model", "fold", "checkpoint", "case_id", "parameters", "trainable_parameters", "macs_per_volume", "flops_per_volume", "active_macs_per_volume", "active_flops_per_volume", "flops_input_shape", "flops_method", *REGION_BURDEN_FIELDS, *REGION_STATUS_FIELDS, "valid_dice_regions", "dice_wt", "dice_tc", "dice_et", "mean_dice")
    temporary = path.with_suffix(".tmp")
    with open(temporary, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader()
        for record in records: writer.writerow({"run_id": run_id, "model": model_name, "fold": fold, "checkpoint": checkpoint, **complexity, **record})
        dice = {region: _positive_region_mean(records, region) for region in REGION_NAMES}
        writer.writerow({"run_id": run_id, "model": model_name, "fold": fold, "checkpoint": checkpoint, "case_id": "__mean_positive_gt__", **complexity, **{field: sum(x[field] for x in records) / len(records) for field in REGION_BURDEN_FIELDS}, **{f"empty_gt_false_positive_{region}": _empty_fp_rate(records, region) for region in REGION_NAMES}, "dice_wt": dice["wt"], "dice_tc": dice["tc"], "dice_et": dice["et"], "mean_dice": sum(dice.values()) / len(dice)})
        file.flush(); os.fsync(file.fileno())
    os.replace(temporary, path)


def write_selector_csv(path: Path, records, run_id, model_name, fold, checkpoint):
    fields = ("run_id", "model", "fold", "checkpoint", "case_id", "stage", "mode", "branch", "weight", "dense_probability", "executed", "contribution_rms", "entropy", *REGION_BURDEN_FIELDS)
    temporary = path.with_suffix(".tmp")
    with open(temporary, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader()
        for record in records: writer.writerow({"run_id": run_id, "model": model_name, "fold": fold, "checkpoint": checkpoint, **record})
        file.flush(); os.fsync(file.fileno())
    os.replace(temporary, path)
