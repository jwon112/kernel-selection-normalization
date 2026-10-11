#!/usr/bin/env python3
import argparse
import json
import os
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from seg3d.data import BraTSVolumeDataset, gpu_augment
from seg3d.branch_p import ConditionedDataset, apply_condition, inner_case_ids
from seg3d.losses import dice_bce_loss, region_dice
from seg3d.complexity import profile_model_complexity
from seg3d.model_zoo import build_model
from seg3d.tracking import setup_logging
from seg3d.reporting import (
    append_epoch_jsonl, full_volume_validation, update_training_curves,
    write_final_fold_csv, write_selector_csv,
)


def atomic_torch_save(payload, path):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_json_save(payload, path):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def summarize_selector_records(records):
    if not records:
        return {}
    summary = {
        "selector_entropy": sum(row["entropy"] for row in records) / len(records),
    }
    branches = sorted({row["branch"] for row in records})
    for branch in branches:
        rows = [row for row in records if row["branch"] == branch]
        summary[f"selector_weight_{branch}"] = sum(row["weight"] for row in rows) / len(rows)
        summary[f"selector_probability_{branch}"] = (
            sum(row["dense_probability"] for row in rows) / len(rows)
        )
        summary[f"selector_execution_rate_{branch}"] = (
            sum(float(row["executed"]) for row in rows) / len(rows)
        )
        summary[f"selector_contribution_rms_{branch}"] = (
            sum(row["contribution_rms"] for row in rows) / len(rows)
        )
    return summary


def accumulate_training_selector_batch(stats, snapshot):
    if snapshot is None:
        return stats
    branch_names = tuple(snapshot["branch_names"])
    if stats is None:
        stats = {
            "branch_names": branch_names,
            "cases": 0,
            "steps": 0,
            "entropy_sum": 0.0,
            "selected_counts": {name: 0 for name in branch_names},
            "probability_sums": {name: 0.0 for name in branch_names},
            "gradient_steps": {name: 0 for name in branch_names},
        }
    elif stats["branch_names"] != branch_names:
        raise RuntimeError("Selector branch order changed within one training epoch")

    probabilities = snapshot.get("dense_probabilities", snapshot["weights"])
    selected = snapshot.get("selected_mask")
    if selected is None:
        selected = torch.ones_like(snapshot["weights"], dtype=torch.bool)
    batch_cases = int(probabilities.shape[0])
    stats["cases"] += batch_cases
    stats["entropy_sum"] += float(snapshot["entropy"].sum().cpu())
    selected_counts = selected.sum(dim=0).cpu().tolist()
    probability_sums = probabilities.float().sum(dim=0).cpu().tolist()
    for index, name in enumerate(branch_names):
        stats["selected_counts"][name] += int(selected_counts[index])
        stats["probability_sums"][name] += float(probability_sums[index])
    return stats


def accumulate_training_selector_gradients(stats, model):
    if stats is None:
        return stats
    snapshot_fn = getattr(model, "expert_gradient_snapshot", None)
    if not callable(snapshot_fn):
        raise RuntimeError("Selector model does not expose expert_gradient_snapshot()")
    gradient_presence = snapshot_fn()
    stats["steps"] += 1
    for name in stats["branch_names"]:
        stats["gradient_steps"][name] += int(bool(gradient_presence[name]))
    return stats


def summarize_training_selector_stats(stats):
    if stats is None:
        return {}
    cases = max(stats["cases"], 1)
    steps = max(stats["steps"], 1)
    summary = {
        "train_selector_cases": stats["cases"],
        "train_selector_steps": stats["steps"],
        "train_selector_entropy": stats["entropy_sum"] / cases,
    }
    starved = []
    for name in stats["branch_names"]:
        selected_count = stats["selected_counts"][name]
        gradient_steps = stats["gradient_steps"][name]
        summary[f"train_selector_selected_count_{name}"] = selected_count
        summary[f"train_selector_execution_rate_{name}"] = selected_count / cases
        summary[f"train_selector_probability_{name}"] = (
            stats["probability_sums"][name] / cases
        )
        summary[f"train_selector_gradient_steps_{name}"] = gradient_steps
        summary[f"train_selector_gradient_coverage_{name}"] = gradient_steps / steps
        if selected_count == 0 or gradient_steps == 0:
            starved.append(name)
    summary["train_selector_starved_expert_count"] = len(starved)
    summary["train_selector_starved_experts"] = starved
    return summary


def save_artifacts(paths, model, optimizer, scheduler, epoch, best, config, is_best):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_mean_dice": best,
        "config": config,
    }
    atomic_torch_save(checkpoint, paths["checkpoints"] / "latest.pt")
    atomic_torch_save(model.state_dict(), paths["weights"] / "latest.pt")
    if is_best:
        atomic_torch_save(checkpoint, paths["checkpoints"] / "best.pt")
        atomic_torch_save(model.state_dict(), paths["weights"] / "best.pt")


def create_run(config, config_path, attempt_id, fold, resume):
    trainer = config.get("trainer", config["experiment"])
    trainer_dir = Path(config["results_root"]) / config["dataset"] / trainer
    fold_dir = trainer_dir / f"fold_{fold}"
    attempt_dir = fold_dir / "logs" / attempt_id
    paths = {"run": fold_dir, "trainer": trainer_dir, "logs": attempt_dir}
    for name in ("visualizations", "checkpoints", "weights", "metrics"):
        paths[name] = fold_dir / name
        paths[name].mkdir(parents=True, exist_ok=True)
    attempt_dir.mkdir(parents=True, exist_ok=True)

    if not resume:
        for path in (
            paths["metrics"] / "epoch_history.jsonl",
            paths["metrics"] / "fold_validation_final.csv",
            paths["metrics"] / "fold_validation_final_epoch.csv",
            paths["metrics"] / "fold_validation_best.csv",
            paths["metrics"] / "selector_validation_final.csv",
            paths["metrics"] / "selector_validation_final_epoch.csv",
            paths["metrics"] / "selector_validation_best.csv",
            paths["visualizations"] / "training_curves.png",
            paths["visualizations"] / "loss_curve.png",
            paths["visualizations"] / "dice_curve.png",
        ):
            path.unlink(missing_ok=True)
    shutil.copy2(config_path, fold_dir / "config.json")
    shutil.copy2(config_path, attempt_dir / "config.json")
    metadata = {
        "attempt_id": attempt_id,
        "trainer": trainer,
        "dataset": config["dataset"],
        "experiment": config["experiment"],
        "model": config["model"],
        "fold": fold,
        "resume": resume,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    for metadata_path in (fold_dir / "run.json", attempt_dir / "run.json"):
        temporary = metadata_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(metadata, indent=2) + "\n")
        os.replace(temporary, metadata_path)
    with (fold_dir / "attempts.jsonl").open("a") as stream:
        stream.write(json.dumps(metadata, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return paths


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    dice_sum = torch.zeros(3, device=device)
    cases = 0
    for image, target, _ in loader:
        image = image.to(device, dtype=torch.bfloat16, non_blocking=True)
        target = target.to(device, dtype=torch.float32, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(image)
        dice_sum += region_dice(logits, target)
        cases += 1
    return (dice_sum / max(cases, 1)).cpu().tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--attempt-id", "--run-id", dest="attempt_id", required=True)
    parser.add_argument("--fold", type=int, default=0, choices=range(5))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    with open(args.config) as file:
        config = json.load(file)
    paths = create_run(config, args.config, args.attempt_id, args.fold, args.resume)
    logger = setup_logging(paths["logs"] / "training.log")
    history_path = paths["metrics"] / "epoch_history.jsonl"
    curves_dir = paths["visualizations"]

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    seed = config["seed"] + args.fold
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")

    train_set = BraTSVolumeDataset(config["data_dir"], args.fold, "train")
    val_set = BraTSVolumeDataset(config["data_dir"], args.fold, "val")
    branch_p = config.get("branch_p")
    if branch_p is not None:
        if branch_p.get("protocol") != "brats23_modality_reliability_p_v1":
            raise ValueError("Unknown Branch-P protocol")
        train_set.case_ids, gate_case_ids = inner_case_ids(
            train_set.case_ids, fold=args.fold,
            fraction=branch_p["inner_gate_fraction"], salt=branch_p["inner_split_salt"])
        if set(train_set.case_ids) & set(gate_case_ids):
            raise RuntimeError("Expert/gate inner split overlap")
        val_set = ConditionedDataset(val_set, branch_p["validation_condition"])
    train_loader_options = {
        "batch_size": config["batch_size"],
        "shuffle": True,
        "num_workers": config["num_workers"],
        "pin_memory": True,
        "drop_last": True,
        "persistent_workers": config["num_workers"] > 0,
    }
    if config["num_workers"] > 0:
        train_loader_options["prefetch_factor"] = config.get("prefetch_factor", 1)
    train_loader = DataLoader(train_set, **train_loader_options)
    val_loader = DataLoader(
        val_set, batch_size=config["batch_size"], shuffle=False, num_workers=0, pin_memory=True
    )
    model = build_model(
        config["model"], in_channels=4, out_channels=3,
        base_channels=config.get("base_channels", 16),
        **config.get("model_kwargs", {}),
    ).to(device)
    initialization_checkpoint = config.get("initialization_checkpoint")
    if initialization_checkpoint and not args.resume:
        initialization_state = torch.load(
            initialization_checkpoint, map_location=device, weights_only=False
        )
        source_state = initialization_state.get("model", initialization_state)
        excluded = tuple(config.get("initialization_exclude_patterns", ()))
        target_state = model.state_dict()
        compatible_state = {
            name: value
            for name, value in source_state.items()
            if name in target_state
            and target_state[name].shape == value.shape
            and not any(pattern in name for pattern in excluded)
        }
        load_result = model.load_state_dict(compatible_state, strict=False)
        logger.info(
            "Initialized %d/%d tensors from %s (missing=%d unexpected=%d excluded=%s)",
            len(compatible_state), len(target_state), initialization_checkpoint,
            len(load_result.missing_keys), len(load_result.unexpected_keys), excluded,
        )
    trainable_patterns = tuple(config.get("trainable_parameter_patterns", ()))
    if trainable_patterns:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = any(pattern in name for pattern in trainable_patterns)
        trainable_names = [
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        ]
        if not trainable_names:
            raise ValueError(
                f"No parameters matched trainable_parameter_patterns={trainable_patterns}"
            )
        logger.info(
            "Frozen-parameter experiment: %d trainable tensors matching %s",
            len(trainable_names), trainable_patterns,
        )
    auxiliary_loss_weight = float(config.get("auxiliary_loss_weight", 0.0))
    if auxiliary_loss_weight != 0.0 and not callable(getattr(model, "auxiliary_loss", None)):
        raise ValueError(
            f"auxiliary_loss_weight={auxiliary_loss_weight} but model "
            f"{config['model']!r} does not expose auxiliary_loss()"
        )
    experiment_spec = getattr(model, "experiment_spec", None)
    if callable(experiment_spec):
        spec = experiment_spec()
        atomic_json_save(spec, paths["run"] / "model_spec.json")
        atomic_json_save(spec, paths["logs"] / "model_spec.json")
    trainable_named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    router_learning_rate = config.get("router_learning_rate")
    if router_learning_rate is None:
        optimizer_groups = [{
            "params": [parameter for _, parameter in trainable_named_parameters],
            "lr": config["learning_rate"],
            "group_name": "all",
        }]
    else:
        router_parameters = [
            parameter for name, parameter in trainable_named_parameters
            if ".router." in f".{name}."
        ]
        expert_parameters = [
            parameter for name, parameter in trainable_named_parameters
            if ".router." not in f".{name}."
        ]
        optimizer_groups = []
        if expert_parameters:
            optimizer_groups.append({
                "params": expert_parameters,
                "lr": config["learning_rate"],
                "group_name": "expert",
            })
        if router_parameters:
            optimizer_groups.append({
                "params": router_parameters,
                "lr": float(router_learning_rate),
                "group_name": "router",
            })
        if not router_parameters:
            raise ValueError(
                "router_learning_rate was set but no trainable router parameters were found"
            )
        logger.info(
            "Optimizer groups: expert_tensors=%d expert_lr=%.6g "
            "router_tensors=%d router_lr=%.6g",
            len(expert_parameters), config["learning_rate"],
            len(router_parameters), float(router_learning_rate),
        )
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"], eta_min=1e-6
    )
    start_epoch, best = 0, -1.0
    epochs_without_improvement = 0
    latest = paths["checkpoints"] / "latest.pt"
    if args.resume:
        if not latest.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {latest}")
        state = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"] + 1
        best = state["best_mean_dice"]
        logger.info("Resumed from epoch %d", state["epoch"])

    logger.info("run_dir=%s", paths["run"])
    logger.info(
        "model=%s gpu=%s fold=%d train=%d val=%d batch=%d shape=%s params=%d workers=%d",
        config["model"], torch.cuda.get_device_name(0), args.fold, len(train_set), len(val_set),
        config["batch_size"], config["input_shape"],
        sum(parameter.numel() for parameter in model.parameters()),
        config["num_workers"],
    )
    for epoch in range(start_epoch, config["epochs"]):
        model.train()
        epoch_loss = 0.0
        epoch_segmentation_loss = 0.0
        epoch_auxiliary_loss = 0.0
        train_selector_stats = None
        epoch_started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        for step, (image, target, _) in enumerate(train_loader, 1):
            step_started = time.monotonic()
            image = image.to(device, dtype=torch.bfloat16, non_blocking=True)
            target = target.to(device, dtype=torch.float32, non_blocking=True)
            image, target = gpu_augment(image, target, config.get("augmentation", "light"))
            if branch_p is not None:
                image = apply_condition(image, branch_p["train_condition"])
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(image)
                segmentation_loss = dice_bce_loss(
                    logits, target, region_weights=config.get("region_loss_weights")
                )
                if callable(getattr(model, "auxiliary_loss", None)):
                    auxiliary_loss = model.auxiliary_loss()
                else:
                    auxiliary_loss = segmentation_loss.new_zeros(())
                loss = segmentation_loss + auxiliary_loss_weight * auxiliary_loss
            loss.backward()
            snapshot_fn = getattr(model, "selection_snapshot", None)
            snapshot = snapshot_fn() if callable(snapshot_fn) else None
            train_selector_stats = accumulate_training_selector_batch(
                train_selector_stats, snapshot
            )
            train_selector_stats = accumulate_training_selector_gradients(
                train_selector_stats, model
            )
            optimizer.step()
            epoch_loss += loss.item()
            epoch_segmentation_loss += segmentation_loss.item()
            epoch_auxiliary_loss += auxiliary_loss.item()
            if step == 1 or step % 10 == 0:
                logger.info(
                    "epoch=%d step=%d/%d loss=%.5f step_s=%.2f",
                    epoch, step, len(train_loader), loss.item(),
                    time.monotonic() - step_started,
                )
        scheduler.step()
        val_loss, dice, _, selector_records = full_volume_validation(
            model, val_loader, device
        )
        mean_dice = sum(dice) / len(dice)
        min_delta = config.get("early_stopping_min_delta", 0.0)
        is_best = mean_dice > best + min_delta
        best = max(best, mean_dice)
        epochs_without_improvement = 0 if is_best else epochs_without_improvement + 1
        epoch_seconds = time.monotonic() - epoch_started
        peak_vram = torch.cuda.max_memory_allocated() / 2**30
        row = {
            "epoch": epoch,
            "train_loss": epoch_loss / len(train_loader),
            "train_segmentation_loss": epoch_segmentation_loss / len(train_loader),
            "train_auxiliary_loss": epoch_auxiliary_loss / len(train_loader),
            "auxiliary_loss_weight": auxiliary_loss_weight,
            "val_loss": val_loss,
            "dice_wt": dice[0],
            "dice_tc": dice[1],
            "dice_et": dice[2],
            "mean_dice": mean_dice,
            "learning_rate": next(
                (group["lr"] for group in optimizer.param_groups
                 if group.get("group_name") in ("expert", "all")),
                optimizer.param_groups[0]["lr"],
            ),
            "router_learning_rate": next(
                (group["lr"] for group in optimizer.param_groups
                 if group.get("group_name") == "router"),
                None,
            ),
            "epoch_seconds": epoch_seconds,
            "peak_vram_gib": peak_vram,
        }
        train_selector_summary = summarize_training_selector_stats(
            train_selector_stats
        )
        row.update(train_selector_summary)
        row.update(summarize_selector_records(selector_records))
        append_epoch_jsonl(history_path, row)
        update_training_curves(history_path, curves_dir)
        save_artifacts(
            paths, model, optimizer, scheduler, epoch, best, config, is_best
        )
        logger.info(
            "epoch=%d train_loss=%.5f val_loss=%.5f dice_WT=%.4f dice_TC=%.4f "
            "dice_ET=%.4f mean_dice=%.4f epoch_s=%.1f peak_vram_gib=%.1f%s",
            epoch, row["train_loss"], val_loss, *dice, mean_dice, epoch_seconds,
            peak_vram, " NEW_BEST" if is_best else "",
        )
        if train_selector_summary.get("train_selector_starved_expert_count", 0):
            logger.warning(
                "epoch=%d starved_experts=%s (zero selections or zero gradient steps)",
                epoch,
                train_selector_summary["train_selector_starved_experts"],
            )
        patience = config.get("early_stopping_patience")
        minimum_epochs = config.get("early_stopping_min_epochs", 0)
        if patience is not None and epoch + 1 >= minimum_epochs and epochs_without_improvement >= patience:
            logger.info("Early stopping at epoch=%d after %d epochs without mean Dice improvement >= %.6f", epoch, epochs_without_improvement, min_delta)
            break

    checkpoint_selection = config.get("checkpoint_selection", "final_epoch")
    if checkpoint_selection == "final_epoch":
        selected_checkpoint = paths["checkpoints"] / "latest.pt"
        selected_checkpoint_label = "checkpoints/latest.pt"
    elif checkpoint_selection == "best":
        selected_checkpoint = paths["checkpoints"] / "best.pt"
        selected_checkpoint_label = "checkpoints/best.pt"
    else:
        raise ValueError(
            f"Unsupported checkpoint_selection={checkpoint_selection!r}; "
            "expected 'final_epoch' or 'best'"
        )
    logger.info(
        "Running primary full-volume fold evaluation with %s",
        selected_checkpoint_label,
    )
    final_state = torch.load(
        selected_checkpoint, map_location=device, weights_only=False
    )
    model.load_state_dict(final_state["model"])
    final_val_loss, final_dice, final_records, final_selector_records = (
        full_volume_validation(model, val_loader, device)
    )
    complexity = profile_model_complexity(
        model, tuple(config["input_shape"]), device,
        selector_records=final_selector_records,
    )
    final_csv = paths["metrics"] / "fold_validation_final.csv"
    write_final_fold_csv(
        final_csv, final_records, args.attempt_id, config["model"], args.fold,
        selected_checkpoint_label, complexity,
    )
    if final_selector_records:
        selector_csv = paths["metrics"] / "selector_validation_final.csv"
        write_selector_csv(
            selector_csv,
            final_selector_records,
            args.attempt_id,
            config["model"],
            args.fold,
            selected_checkpoint_label,
        )
        logger.info("Final selector signals: csv=%s", selector_csv)
    logger.info(
        "Final fold validation: loss=%.5f WT=%.4f TC=%.4f ET=%.4f mean=%.4f csv=%s",
        final_val_loss, *final_dice, sum(final_dice) / 3, final_csv,
    )


if __name__ == "__main__":
    main()
