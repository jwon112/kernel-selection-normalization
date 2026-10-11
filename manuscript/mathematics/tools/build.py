#!/usr/bin/env python3
"""Rebuild the kernel paper from immutable clinical artifacts and CPU probes."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parents[1]
sys.path.insert(0, str(ROOT / "manuscript/tools"))
sys.path.insert(0, str(ROOT / "src"))
from build_evidence_tables import THREE, FIVE, brats_family, region_macro, sha, config_differences
from build_statistics import patient_id, cluster_matrix, fold_scores
from seg3d.multikernel import MultiKernelSelector3D
from completed_diagnostics import normalization_data
from check_stage_robustness import analyze_cells as stage_sensitivity_data

SEED = 20261001
DRAWS = 4000
BRANCHES = ("k3", "k5", "dilated_k3", "anisotropic")
STAGE_SEEDS = (2026, 20261001, 20261011)
HASHES: dict[str, str] = {}
EXTENSION_BASES: dict[tuple[str, int], Path] = {}
TERMINALS: dict[Path, dict] = {}


def completion_data():
    """Verify the exact thirteen missing-fold fits; do not infer completion."""
    queue = ROOT / "results/mathematics_fold_completion_v1/queue"
    plan_path = queue / "plan.json"
    plan = json.loads(plan_path.read_text())
    HASHES[str(plan_path.relative_to(ROOT))] = sha(plan_path)
    if len(plan["jobs"]) != 13 or plan["base_seed"] != 2026:
        raise ValueError("Unexpected fold-completion scope")
    completed = []
    for job in plan["jobs"]:
        path = queue / (job["job"] + ".json")
        status = json.loads(path.read_text())
        HASHES[str(path.relative_to(ROOT))] = sha(path)
        if status["phase"] != "complete" or status["effective_seed"] != 2026 + job["fold"]:
            raise ValueError("Incomplete or incompatible missing-fold run")
        config_path = ROOT / job["config"]
        config = json.loads(config_path.read_text())
        HASHES[str(config_path.relative_to(ROOT))] = sha(config_path)
        base = ROOT / "results/mathematics_fold_completion_v1" / config["dataset"] / config["trainer"] / f"fold_{job['fold']}"
        if (config["trainer"], job["fold"]) in EXTENSION_BASES:
            raise ValueError("Duplicate missing-fold fit")
        EXTENSION_BASES[(config["trainer"], job["fold"])] = base
        for name, expected in status["artifacts"].items():
            actual = sha(ROOT / name)
            if actual != expected:
                raise ValueError("Changed completed missing-fold artifact: " + name)
            HASHES[name] = actual
        completed.append(dict(job=job["job"], family=job["family"], fold=job["fold"],
                              base=str(base.relative_to(ROOT)), artifacts=status["artifacts"]))
    return dict(jobs=completed, models=13, base_seed=2026,
                boundary=plan["interpretation"], source_plan=str(plan_path.relative_to(ROOT)))


def comparison_base(trainer, fold):
    return EXTENSION_BASES.get((trainer, fold), ROOT / "results/brats2023_gli" / trainer / f"fold_{fold}")


def comparison_metric(base):
    filename = "fold_validation_final_epoch.csv" if "mathematics_fold_completion_v1" in base.parts else "fold_validation_final.csv"
    return base / "metrics" / filename


def terminal_record(base):
    if base in TERMINALS:
        return TERMINALS[base]
    path = base / "metrics/final_epoch_evaluation.json"
    if path.exists():
        final = json.loads(path.read_text())
    else:
        # Two historical canonical folds predate the standalone JSON writer.
        # Verify the checkpoint itself and the corresponding terminal history.
        checkpoint = base / "checkpoints/latest.pt"
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        actual = json.loads((base / "config.json").read_text())
        if state["epoch"] != 99 or state["config"] != actual:
            raise ValueError("Historical canonical checkpoint is not terminal")
        HASHES[str(checkpoint.relative_to(ROOT))] = sha(checkpoint)
        del state
        path = base / "metrics/epoch_history.jsonl"
        history = [json.loads(line) for line in path.read_text().splitlines()]
        if history[-1]["epoch"] != 99:
            raise ValueError("Historical canonical history is not terminal")
        final = dict(history[-1], checkpoint="checkpoints/latest.pt")
    HASHES[str(path.relative_to(ROOT))] = sha(path)
    TERMINALS[base] = dict(final, evidence_path=str(path.relative_to(ROOT)))
    return TERMINALS[base]


def comparison_family(name, trainer, config_name):
    """Combine historical folds with terminal missing-fold fits, keeping provenance."""
    config_path = ROOT / "configs" / config_name
    config = json.loads(config_path.read_text())
    HASHES[str(config_path.relative_to(ROOT))] = sha(config_path)
    details = []
    for fold in range(5):
        base = comparison_base(trainer, fold)
        record = metric_rows(comparison_metric(base))
        for filename in ("config.json", "run.json"):
            HASHES[str((base / filename).relative_to(ROOT))] = sha(base / filename)
        actual = json.loads((base / "config.json").read_text())
        run = json.loads((base / "run.json").read_text())
        final = terminal_record(base)
        if actual["trainer"] != trainer or run["fold"] != fold or final["epoch"] != 99 or final["checkpoint"] != "checkpoints/latest.pt":
            raise ValueError("Wrong schedule run/terminal evaluation")
        difference = config_differences(config, actual)
        if difference:
            raise ValueError("Scientific recipe changed during fold extension")
        score = region_macro(list(record.values()))
        if not np.isclose(score, final["mean_dice"], atol=1e-12, rtol=0):
            raise ValueError("Final JSON/CSV mismatch")
        flops = {float(r["active_flops_per_volume"]) for r in record.values()}
        parameters = {int(r["parameters"]) for r in record.values()}
        if len(flops) != 1 or len(parameters) != 1 or not (base / "checkpoints/latest.pt").exists():
            raise ValueError("Invalid schedule profile/checkpoint")
        details.append(dict(fold=fold, studies=len(record), dice=score, active_flops=flops.pop(),
                            parameters=parameters.pop(), base=str(base.relative_to(ROOT)),
                            config_sha256=HASHES[str((base / 'config.json').relative_to(ROOT))],
                            run_config_differences=difference))
    if len({d["parameters"] for d in details}) != 1:
        raise ValueError("Schedule capacity differs between folds")
    return dict(name=name, trainer=trainer, config=str(config_path.relative_to(ROOT)), model=config["model"],
                folds=details, dice=float(np.mean([d["dice"] for d in details])), parameters=details[0]["parameters"],
                active_tflops_range=[min(d["active_flops"] for d in details)/1e12, max(d["active_flops"] for d in details)/1e12],
                study_counts=[d["studies"] for d in details])


def rows(path: Path) -> list[dict]:
    HASHES[str(path.relative_to(ROOT))] = sha(path)
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def metric_rows(path: Path) -> dict[str, dict]:
    raw = [r for r in rows(path) if not r["case_id"].startswith("__")]
    if any(r["checkpoint"] != "checkpoints/latest.pt" for r in raw):
        raise ValueError(f"Nonterminal metric: {path}")
    result = {r["case_id"]: r for r in raw}
    if len(result) != len(raw):
        raise ValueError(f"Duplicate study: {path}")
    return result


def allcase(records: list[dict]) -> list[dict]:
    """All studies/all regions; empty/empty scores 1, empty/nonempty scores 0."""
    result=[]
    for row in records:
        copy=dict(row)
        for region in ("wt", "tc", "et"):
            if row[f"gt_positive_{region}"] == "False":
                copy[f"dice_{region}"] = "1" if int(row[f"predicted_voxels_{region}"]) == 0 else "0"
            copy[f"gt_positive_{region}"] = "True"
        result.append(copy)
    return result


def stage_data(seed=None) -> tuple[list[dict], list[dict]]:
    cells, folds = [], []
    trainer = "BraTS23GLIMKMultiStageSK100Trainer"
    for fold in range(3 if seed is None else 5):
        base = ROOT / "results/brats2023_gli" / trainer / f"fold_{fold}"
        if seed is not None:
            result_root = "kernel_stage_thirdseed_v1" if seed == 20261011 else "kernel_stage_fivefold_twoseed_v1"
            base = ROOT / "results" / result_root / "brats2023_gli" / f"KernelStageSK100Seed{seed}Trainer" / f"fold_{fold}"
            final = terminal_record(base)
            final_path = ROOT / final["evidence_path"]
            if final["epoch"] != 99 or final["checkpoint"] != "checkpoints/latest.pt":
                raise ValueError("Incomplete replication checkpoint")
        if not (base / "checkpoints/latest.pt").exists():
            raise ValueError(f"Missing checkpoint: {base}")
        attention = defaultdict(dict)
        for row in rows(base / ("metrics/selector_validation_final.csv" if seed is None else "metrics/selector_validation_final_epoch.csv")):
            if row["stage"] == "aggregate":
                continue
            if row["checkpoint"] != "checkpoints/latest.pt":
                raise ValueError("Nonterminal selector")
            entry=attention[(row["stage"], row["branch"])]
            if row["case_id"] in entry: raise ValueError("Duplicate stage selector study")
            entry[row["case_id"]]=float(row["weight"])
        conditions = {}
        for key, folder in (
            ("drop", "stage_selector_interventions_final_epoch"),
            ("renorm", "stage_selector_interventions_reroute_final_epoch"),
            ("keep", "stage_selector_keep_one_final_epoch"),
        ):
            grouped = defaultdict(list)
            for row in rows(base / "metrics" / folder / "cases.csv"):
                if row["checkpoint"] != "checkpoints/latest.pt":
                    raise ValueError("Nonterminal intervention")
                grouped[row["condition"]].append(row)
            ref_ids = {r["case_id"] for r in grouped["learned"]}
            for name, records in grouped.items():
                if len(records) != len(ref_ids) or {r["case_id"] for r in records} != ref_ids:
                    raise ValueError(f"Unpaired or duplicate condition {name}")
            conditions[key] = grouped
        baseline_ids = {r["case_id"] for r in conditions["drop"]["learned"]}
        baseline = region_macro(conditions["drop"]["learned"])
        if seed is not None:
            splits = json.loads((ROOT / "data/processed/brats2023_gli_crop4/splits.json").read_text())
            if baseline_ids != set(splits[fold]["val"]) or not np.isclose(baseline, final["mean_dice"], atol=1e-12, rtol=0):
                raise ValueError("Replication split or final Dice mismatch")
        baseline_all = region_macro(allcase(conditions["drop"]["learned"]))
        for key in ("keep", "renorm"):
            if {r["case_id"] for r in conditions[key]["learned"]} != baseline_ids:
                raise ValueError("Unpaired intervention families")
            if not np.isclose(region_macro(conditions[key]["learned"]), baseline, atol=1e-10):
                raise ValueError("Intervention families use different learned baselines")
        this = []
        for stage in (1, 2, 3):
            for branch in BRANCHES:
                keep = region_macro(conditions["keep"][f"keep_decoder_{stage}_{branch}"])
                drop = region_macro(conditions["drop"][f"drop_decoder_{stage}_{branch}"])
                renorm = region_macro(conditions["renorm"][f"drop_decoder_{stage}_{branch}_and_reroute"])
                weight_map = attention[(f"decoder_{stage}", branch)]
                if set(weight_map) != baseline_ids:
                    raise ValueError("Selector and intervention study counts differ")
                weights=list(weight_map.values())
                cell = dict(fold=fold, stage=stage, branch=branch,
                            weight=float(np.mean(weights)), learned=baseline,
                            drop_effect=baseline-drop, renorm_effect=baseline-renorm,
                            keep_dice=keep,
                            allcase_drop_effect=baseline_all-region_macro(allcase(conditions["drop"][f"drop_decoder_{stage}_{branch}"])),
                            allcase_keep_dice=region_macro(allcase(conditions["keep"][f"keep_decoder_{stage}_{branch}"])))
                cells.append(cell)
                this.append(cell)
        corr = {key: float(spearmanr([r["weight"] for r in this], [r[key] for r in this]).statistic)
                for key in ("drop_effect", "renorm_effect", "keep_dice")}
        within = []
        for stage in (1, 2, 3):
            subset = [r for r in this if r["stage"] == stage]
            within.append({"stage": stage, **{
                key: float(spearmanr([r["weight"] for r in subset], [r[key] for r in subset]).statistic)
                for key in corr}})
        folds.append(dict(fold=fold, studies=len(baseline_ids),
                          patients=len({patient_id(c) for c in baseline_ids}),
                          correlations=corr, within_stage_correlations=within,
                          allcase_correlations={key:float(spearmanr([r["weight"] for r in this],[r["allcase_"+key] for r in this]).statistic) for key in ("drop_effect", "keep_dice")}))
    if seed is not None:
        for row in cells + folds:
            row["fold_index"] = row["fold"]
            row["base_seed"] = seed
            row["fold"] = f"S{STAGE_SEEDS.index(seed)+1}/{row['fold']}"
    return cells, folds


def replication_data():
    """Recompute all three seed groups against the completed 15-bank gate."""
    gate_path = ROOT / "results/kernel_normalization_audit_v1/queue/matrix_gate.json"
    gate = json.loads(gate_path.read_text())
    if gate["models"] != 15:
        raise ValueError("Expected fifteen completed models")
    for seed in STAGE_SEEDS:
        result_root="kernel_stage_thirdseed_v1" if seed==20261011 else "kernel_stage_fivefold_twoseed_v1"
        for fold in range(5):
            path=ROOT/"results"/result_root/"brats2023_gli"/f"KernelStageSK100Seed{seed}Trainer"/f"fold_{fold}/checkpoints/latest.pt"
            actual=sha(path)
            if actual!=gate["checkpoints"][f"seed{seed}_fold{fold}"]:
                raise ValueError("Stage checkpoint content changed")
            HASHES[str(path.relative_to(ROOT))]=actual
    for name, expected in gate["source_hashes"].items():
        actual = sha(ROOT / name)
        if actual != expected:
            raise ValueError(f"Completion-gate artifact changed: {name}")
        HASHES[name] = actual
    for path in (gate_path, ROOT / "results/kernel_stage_fivefold_twoseed_v1/queue/plan.json",
                 ROOT / "results/kernel_stage_thirdseed_v1/queue/plan.json",
                 ROOT / "data/processed/brats2023_gli_crop4/splits.json"):
        HASHES[str(path.relative_to(ROOT))] = sha(path)
    cells, folds, pooled = [], [], {}
    for seed in STAGE_SEEDS:
        cc, ff = stage_data(seed)
        cells.extend(cc); folds.extend(ff)
        pooled[str(seed)] = {key: float(spearmanr([r["weight"] for r in cc], [r[key] for r in cc]).statistic)
                            for key in ("drop_effect", "renorm_effect", "keep_dice")}
    summary_path = ROOT / "results/kernel_stage_thirdseed_v1/analysis/summary.json"
    summary = json.loads(summary_path.read_text())
    HASHES[str(summary_path.relative_to(ROOT))] = sha(summary_path)
    for run in summary["runs"]:
        match = next(r for r in folds if r["base_seed"] == run["base_seed"] and r["fold_index"] == run["fold"])
        for key in ("drop_effect", "keep_dice"):
            if not np.isclose(match["correlations"][key], run["correlations"][key], atol=1e-12, rtol=0):
                raise ValueError("Independent replication aggregation mismatch")
    expected = {(seed, fold) for seed in STAGE_SEEDS for fold in range(5)}
    if (len(summary["runs"]) != 15 or len(cells) != 180 or
            {(r["base_seed"], r["fold"]) for r in summary["runs"]} != expected):
        raise ValueError("Incomplete three-seed evidence")
    for entry in summary["seed_stage_winner_agreement"]:
        for seed in STAGE_SEEDS:
            subset = [c for c in cells if c["base_seed"] == seed and
                      c["fold_index"] == entry["fold"] and c["stage"] == entry["stage"]]
            for key in ("weight", "drop_effect", "keep_dice"):
                winner = max(subset, key=lambda c: c[key])["branch"]
                if winner != entry["winners"][str(seed)][key]:
                    raise ValueError("Cross-seed winner reconstruction mismatch")
    return cells, folds, pooled, summary


def paired_schedule_counts(by_model: dict, fold: int) -> dict:
    """Check common study eligibility and report actual GT-positive denominators."""
    reference = next(iter(by_model.values()))
    ids = set(reference)
    if not ids or any(set(record) != ids for record in by_model.values()):
        raise ValueError("Unpaired schedule cases")
    regions = ("wt", "tc", "et")
    for case in ids:
        for region in regions:
            key = f"gt_positive_{region}"
            flag = reference[case][key]
            if flag not in ("True", "False"):
                raise ValueError("Invalid reference-presence flag")
            if any(record[case][key] != flag for record in by_model.values()):
                raise ValueError("Model-specific reference eligibility")
    return dict(fold=fold, studies=len(ids), patients=len({patient_id(c) for c in ids}),
                positive_reference={region: dict(
                    studies=sum(row[f"gt_positive_{region}"] == "True" for row in reference.values()),
                    patients=len({patient_id(case) for case, row in reference.items()
                                  if row[f"gt_positive_{region}"] == "True"})) for region in regions})


def schedule_run_artifacts(families: list[dict]) -> list[dict]:
    """Bind each comparison row to actual recipe, terminal evaluation and profile."""
    records = []
    for family in families:
        for detail in family["folds"]:
            base = comparison_base(family["trainer"], detail['fold'])
            actual = json.loads((base / "config.json").read_text())
            run = json.loads((base / "run.json").read_text())
            final = terminal_record(base)
            final_path = ROOT / final["evidence_path"]
            if final["checkpoint"] != "checkpoints/latest.pt" or final["epoch"] != actual["epochs"] - 1:
                raise ValueError("Nonterminal schedule evaluation record")
            profile_path = comparison_metric(base)
            profile_rows = metric_rows(profile_path)
            profile = {}
            for key in ("flops_method", "flops_input_shape", "active_flops_per_volume", "parameters"):
                values = {r[key] for r in profile_rows.values()}
                if len(values) != 1:
                    raise ValueError(f"Inconsistent schedule profile: {key}")
                profile[key] = next(iter(values))
            final_score = float(final["mean_dice"])
            if not np.isclose(final_score, detail["dice"], rtol=0, atol=1e-12):
                raise ValueError("Terminal JSON and schedule CSV disagree")
            relative = str(base.relative_to(ROOT))
            records.append(dict(
                family=family["name"], trainer=family["trainer"], fold=detail["fold"],
                run_id=run["attempt_id"], source_config=family["config"],
                source_config_sha256=HASHES[family["config"]],
                actual_config=relative + "/config.json", actual_config_sha256=detail["config_sha256"],
                run_record=relative + "/run.json", run_record_sha256=HASHES[relative + "/run.json"],
                recipe={k: actual.get(k) for k in ("seed", "batch_size", "epochs", "learning_rate",
                    "weight_decay", "augmentation", "precision", "checkpoint_selection", "model_kwargs")},
                config_differences=detail["run_config_differences"],
                config_comparison_boundary="Compares recorded non-path fields; excludes data_dir, results_root, output_dir",
                checkpoint=relative + "/checkpoints/latest.pt",
                checkpoint_content_hashed=(relative + "/checkpoints/latest.pt") in HASHES,
                final_evaluation=str(final_path.relative_to(ROOT)),
                final_evaluation_sha256=HASHES[str(final_path.relative_to(ROOT))],
                terminal_epoch_zero_based=final["epoch"],
                metric_and_profile=str(profile_path.relative_to(ROOT)),
                metric_and_profile_sha256=HASHES[str(profile_path.relative_to(ROOT))],
                recorded_profile=profile))
    return records


def schedule_data() -> tuple[list[dict], list[dict], list[dict], dict]:
    families = [comparison_family(*family) for family in THREE]
    rng = np.random.default_rng(SEED)
    points, draws = defaultdict(list), defaultdict(list)
    allpoints, alldraws = defaultdict(list), defaultdict(list)
    counts = []
    for fold in range(5):
        by_model = {}
        reference = None
        for name, trainer, _ in THREE:
            record = metric_rows(comparison_metric(comparison_base(trainer, fold)))
            ids = set(record)
            if reference is None:
                reference = ids
            elif ids != reference:
                raise ValueError("Unpaired schedule cases")
            by_model[name] = record
        counts.append(paired_schedule_counts(by_model, fold))
        patients = sorted({patient_id(c) for c in reference})
        indices = rng.integers(0, len(patients), size=(DRAWS, len(patients)), dtype=np.int32)
        for name, record in by_model.items():
            point, sample = fold_scores(cluster_matrix(record, patients), indices)
            points[name].append(point)
            draws[name].append(sample)
            allrecord={r["case_id"]:r for r in allcase(list(record.values()))}
            ap, ab=fold_scores(cluster_matrix(allrecord,patients),indices)
            allpoints[name].append(ap);alldraws[name].append(ab)
    contrasts = []
    allcontrasts=[]
    for left, right in (("Attention-derived fixed", "Multi-stage SK"),
                        ("Drop-derived fixed", "Multi-stage SK"),
                        ("Attention-derived fixed", "Canonical U-Net"),
                        ("Drop-derived fixed", "Canonical U-Net"),
                        ("Multi-stage SK", "Multi-stage uniform")):
        samples = np.mean(draws[left], axis=0)-np.mean(draws[right], axis=0)
        contrasts.append(dict(left=left, right=right,
                              point=float(np.mean(points[left])-np.mean(points[right])),
                              ci95=np.quantile(samples, [.025, .975]).tolist()))
        asamples=np.mean(alldraws[left],axis=0)-np.mean(alldraws[right],axis=0)
        allcontrasts.append(dict(left=left,right=right,point=float(np.mean(allpoints[left])-np.mean(allpoints[right])),ci95=np.quantile(asamples,[.025,.975]).tolist()))
    reporting = dict(fold_counts=counts, runs=schedule_run_artifacts(families),
                     recorded_profile_boundary="Identical reported shape and THOP rule, but recorded counts have a shared 0.443925135360-TFLOP shift across historical/extension profiling regimes. Ranges are not harmonized re-profiling.",
                     model_specific_missing_predictions=False,
                     common_patient_indices_across_models=True,
                     interval_method="paired-difference percentile; quantiles 0.025 and 0.975",
                     zero_positive_reference_draw="raises ValueError, never silently dropped",
                     weighting="equal-fold, equal-region; study means within eligible region",
                     boundary="Conditional on fitted models; no patient-weighted target or retraining uncertainty")
    sk = next(r for r in families if r["name"] == "Multi-stage SK")
    reporting["active_flops_reduction_ranges"] = {}
    # This exact arithmetic match does not establish a historical package version.
    transpose_terms=[]
    for level, ci, co in ((3,512,256),(2,256,128),(1,128,64),(0,64,32)):
        output_voxels=int(np.prod([n//2**level for n in (192,224,192)]))
        input_voxels=output_voxels//8
        transpose_terms.append(dict(input_channels=ci,output_channels=co,
            output_voxels=output_voxels,input_voxels=input_voxels,
            flops_difference=2*(output_voxels-input_voxels)*ci*co*8))
    reporting["transpose_accounting_offset"] = dict(terms=transpose_terms,
        total_flops=sum(r["flops_difference"] for r in transpose_terms),
        boundary="Exact arithmetic match to the common recorded offset; historical imported profiler version is not established.")
    for family in families:
        if family["name"] in ("Attention-derived fixed", "Drop-derived fixed"):
            reductions = [1-d["active_flops"]/s["active_flops"] for d, s in zip(family["folds"], sk["folds"])]
            reporting["active_flops_reduction_ranges"][family["name"]] = [min(reductions), max(reductions)]
    return families, contrasts, allcontrasts, reporting


def stage_ranges(cells: list[dict]) -> list[dict]:
    result = []
    for fold in sorted({r["fold"] for r in cells}):
        for stage in (1, 2, 3):
            subset = [r for r in cells if r["fold"] == fold and r["stage"] == stage]
            if len(subset) != 4 or {r["branch"] for r in subset} != set(BRANCHES):
                raise ValueError("Stage range requires exactly the four named branches")
            result.append(dict(fold=fold, stage=stage, **{
                key: [min(r[key] for r in subset), max(r[key] for r in subset)]
                for key in ("weight", "drop_effect", "renorm_effect", "keep_dice")}))
    return result


def pooling_data() -> list[dict]:
    output=[]
    for pooling,trainer in (("GAP","BraTS23GLIMKSoftControlTrainer"),("Average + maximum","BraTS23GLIMKSoftAvgMax100Trainer")):
        for fold in range(5):
            base=comparison_base(trainer,fold)
            if not (base/"checkpoints/latest.pt").exists(): raise ValueError("Missing pooling checkpoint")
            cases=defaultdict(dict)
            for r in rows(base/"metrics/selector_interventions_final_epoch/cases.csv"):
                if r["checkpoint"]!="checkpoints/latest.pt": raise ValueError("Nonterminal pooling")
                condition=r["condition"]
                # Legacy drop_<k> and the new explicit suffix describe the same
                # unnormalized operation; bind by intervention fields, not name.
                if r["kind"]=="drop_branch" and r["renormalize"]=="False":
                    condition="drop_"+r["intervention_branch"]
                if r["case_id"] in cases[condition]: raise ValueError("Duplicate pooling case")
                cases[condition][r["case_id"]]=r
            weights=defaultdict(dict)
            for r in rows(base/"metrics/selector_interventions_final_epoch/selector_signals.csv"):
                if r["condition"]=="learned" and r.get("stage", "aggregate")=="aggregate":
                    weights[r["case_id"]][r["branch"]]=float(r["weight"])
            ids=sorted(cases["learned"])
            splits=json.loads((ROOT/"data/processed/brats2023_gli_crop4/splits.json").read_text())
            if set(ids)!=set(splits[fold]["val"]): raise ValueError("Pooling fold membership mismatch")
            if set(weights)!=set(ids): raise ValueError("Pooling signal mismatch")
            if any(set(record)!=set(ids) for record in cases.values()): raise ValueError("Unpaired pooling conditions")
            ww=np.array([[weights[i][k] for k in BRANCHES] for i in ids])
            forced=np.array([[float(cases[f"force_{k}"][i]["mean_dice"]) for k in BRANCHES] for i in ids])
            learned=np.array([float(cases["learned"][i]["mean_dice"]) for i in ids])
            drop=np.array([[float(cases[f"drop_{k}"][i]["mean_dice"]) for k in BRANCHES] for i in ids])
            correlations=[float(spearmanr(ww[:,j],learned-drop[:,j]).statistic) for j in range(4)]
            output.append(dict(pooling=pooling,fold=fold,studies=len(ids),
                               region_macro_dice=region_macro(list(cases["learned"].values())),
                               study_macro_available_region_dice=float(learned.mean()),
                               mean_branch_weight_std=float(ww.std(axis=0).mean()),
                               mean_branch_weight_drop_rho=float(np.mean(correlations)),
                               top_weight_forced_winner_agreement=float(np.mean(ww.argmax(axis=1)==forced.argmax(axis=1)))))
            for filename in ("config.json","run.json"):
                HASHES[str((base/filename).relative_to(ROOT))]=sha(base/filename)
            final=terminal_record(base)
            delta=output[-1]["region_macro_dice"]-float(final["mean_dice"])
            # Preserve the paired intervention baseline. Older GAP re-evaluation
            # differs slightly from its stored terminal validation; report both.
            tolerance=1e-12 if "mathematics_fold_completion_v1" in base.parts else 1e-4
            if final["epoch"]!=99 or abs(delta)>tolerance:
                raise ValueError("Pooling terminal evaluation mismatch")
            output[-1].update(terminal_validation_dice=float(final["mean_dice"]),
                              intervention_minus_terminal_dice=delta,
                              removal_definition="kind=drop_branch, renormalize=False; legacy and explicit condition names mapped by fields")
            source_name="brats23_gli_mk_soft_control_crop4.json" if pooling=="GAP" else "brats23_gli_mk_soft_avgmax_100.json"
            source=ROOT/"configs"/source_name
            HASHES[str(source.relative_to(ROOT))]=sha(source)
            if config_differences(json.loads(source.read_text()),json.loads((base/"config.json").read_text())):
                raise ValueError("Pooling scientific config mismatch")
    return output


def normalize(x: torch.Tensor, eps: float) -> torch.Tensor:
    centered = x-x.mean(dim=(-3, -2, -1), keepdim=True)
    return centered/(centered.square().mean(dim=(-3, -2, -1), keepdim=True)+eps).sqrt()


def numerical_probes() -> tuple[dict, list[dict]]:
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(SEED)
    # Central-difference verification of the local normalization derivative.
    jacobian_errors=[]
    for trial in range(12):
        z=torch.randn(2,4,5,6,7,generator=generator,dtype=torch.float64)
        h=torch.randn(z.shape,generator=generator,dtype=torch.float64)
        centered=z-z.mean(dim=(-3,-2,-1),keepdim=True)
        ch=h-h.mean(dim=(-3,-2,-1),keepdim=True)
        q=(centered.square().mean(dim=(-3,-2,-1),keepdim=True)+1e-5).sqrt()
        analytic=ch/q-centered*(centered*ch).mean(dim=(-3,-2,-1),keepdim=True)/q**3
        delta=1e-6
        numeric=(normalize(z+delta*h,1e-5)-normalize(z-delta*h,1e-5))/(2*delta)
        jacobian_errors.append(float((analytic-numeric).abs().max()))
    if max(jacobian_errors)>1e-7: raise AssertionError("Normalization derivative failed")
    # Reset the stream so the variance/architecture probes retain their v1 inputs.
    generator.manual_seed(SEED)
    errors = []
    curves = []
    for ratio in np.logspace(-6, 8, 71):
        z = torch.randn(2, 4, 5, 6, 7, generator=generator, dtype=torch.float64)
        z = normalize(z, 0.0)*np.sqrt(ratio*1e-5)
        for scale in (1.1, 2., 8.):
            actual = normalize(scale*z, 1e-5)
            predicted = normalize(z, 1e-5)*np.sqrt((ratio+1)/(ratio+1/scale**2))
            error = float((actual-predicted).abs().max())
            errors.append(error)
            relative = float(torch.linalg.vector_norm(actual-normalize(z, 1e-5))/torch.linalg.vector_norm(normalize(z, 1e-5)))
            curves.append(dict(variance_over_epsilon=float(ratio), scale=scale,
                               relative_change=relative, identity_residual=error))
    # Architectural boundary probes: same frozen branch tensors and weight edit.
    architecture = defaultdict(list)
    module_errors = []
    for trial in range(12):
        b = torch.randn(2, 4, 4, 5, 6, 7, generator=generator, dtype=torch.float64)
        a = torch.softmax(torch.randn(2, 4, 4, 1, 1, 1, generator=generator, dtype=torch.float64), dim=1)
        dropped = a.clone(); dropped[:, 0] = 0
        denominator = dropped.sum(dim=1, keepdim=True)
        z = (dropped*b).sum(dim=1)
        zr = (dropped/denominator*b).sum(dim=1)
        for label, left, right in (
            ("Post-fusion IN, epsilon=0", normalize(z, 0), normalize(zr, 0)),
            ("Post-fusion IN, epsilon=1e-5", normalize(z, 1e-5), normalize(zr, 1e-5)),
            ("No post-fusion normalization", z, zr),
            ("Branch-local IN only", (dropped*torch.stack([normalize(b[:, k], 1e-5) for k in range(4)], dim=1)).sum(dim=1),
             (dropped/denominator*torch.stack([normalize(b[:, k], 1e-5) for k in range(4)], dim=1)).sum(dim=1)),
            ("Post-fusion group norm", torch.nn.functional.group_norm(z, 1, eps=0), torch.nn.functional.group_norm(zr, 1, eps=0)),
        ):
            architecture[label].append(float(torch.linalg.vector_norm(left-right)/torch.linalg.vector_norm(left)))
        spatial = torch.softmax(torch.randn(2, 4, 4, 5, 6, 7, generator=generator, dtype=torch.float64), dim=1)
        spatial[:, 0] = 0
        zs = (spatial*b).sum(dim=1)
        zsr = (spatial/spatial.sum(dim=1, keepdim=True)*b).sum(dim=1)
        architecture["Spatial weights + post-fusion IN"].append(float(torch.linalg.vector_norm(normalize(zs, 0)-normalize(zsr, 0))/torch.linalg.vector_norm(normalize(zs, 0))))
        # Actual unchanged repository selector, CPU only. This is untrained instrumentation.
        torch.manual_seed(SEED+trial)
        block = MultiKernelSelector3D(4, "sk").double().eval()
        block.post_activation = torch.nn.Identity()
        with torch.no_grad():
            for head in block.sk_heads.values():
                head.weight.normal_(0, .15); head.bias.normal_(0, .1)
            x = torch.randn(2, 4, 5, 6, 7, generator=generator, dtype=torch.float64)
            block.set_intervention("drop_branch", "k3", renormalize=False)
            out = block(x)
            branch_outputs = [block.branches[n](x) for n in block.branch_names]
            hidden = block.sk_reduce(block._pool(sum(branch_outputs)/4))
            alpha = torch.softmax(torch.stack([block.sk_heads[n](hidden) for n in block.branch_names], dim=1), dim=1)
            d = 1-alpha[:, 0, :, None, None, None]
            zz = sum(alpha[:, k, :, None, None, None]*branch_outputs[k] for k in range(1, 4))
            variance = (zz-zz.mean(dim=(-3,-2,-1), keepdim=True)).square().mean(dim=(-3,-2,-1), keepdim=True)
            eps = block.post_norm.eps
            predicted = out*torch.sqrt((variance+eps)/(variance+eps*d.square()))
            block.set_intervention("drop_branch", "k3", renormalize=True)
            observed = block(x)
            module_errors.append(float((observed-predicted).abs().max()))
    if max(errors) > 1e-10 or max(module_errors) > 1e-10:
        raise AssertionError("Finite-epsilon identity failed")
    # Same simplex weights; a zero-loss mixture whose highest weight is not the best keep-one.
    alpha = np.array([.6, .3, .1]); features = np.array([5., 0., -30.]); target = 0.
    mixture = float(alpha@features)
    counterexample = dict(weights=alpha.tolist(), features=features.tolist(), target=target,
                         mixture_mse=(mixture-target)**2,
                         keep_mse=((features-target)**2).tolist(),
                         removal_mse=(((mixture-alpha*features)-target)**2).tolist())
    assert np.argmax(alpha) != np.argmin(counterexample["keep_mse"])
    return dict(seed=SEED, dtype="float64", trials=12, analytical_identity_max_residual=max(errors),
                local_derivative_max_central_difference_residual=max(jacobian_errors),
                actual_selector_identity_max_residual=max(module_errors),
                architecture=[dict(condition=k, mean_relative_change=float(np.mean(v)),
                                   minimum=float(min(v)), maximum=float(max(v))) for k,v in architecture.items()],
                counterexample=counterexample), curves


def table(header: str, body: list[str]) -> str:
    return "\\begin{tabular}{"+header+"}\n\\toprule\n"+"\n".join(body)+"\n\\bottomrule\n\\end{tabular}\n"


def multipage_table(columns: str, body: list[str], caption: str, label: str) -> str:
    """Repeat the column header when the 45 fold-stage rows cross a page."""
    header = "\\toprule\n" + "\n".join(body[:2]) + "\n"
    return ("\\begin{longtable}{" + columns + "}\n\\caption{" + caption +
            "}\\label{" + label + "}\\\\\n" + header + "\\endfirsthead\n" +
            header + "\\endhead\n\\midrule\n\\endfoot\n\\bottomrule\n\\endlastfoot\n" +
            "\n".join(body[2:]) + "\n\\end{longtable}\n")


def outputs(data: dict, curves: list[dict]) -> dict[str, str]:
    def number(v): return f"{v:.6f}"
    stage_lines = [r"Fold & Drop & Renormalized drop & Keep-one \\", r"\midrule"]
    for row in data["stage_folds"]:
        c=row["correlations"]
        stage_lines.append(f"{row['fold']} & {c['drop_effect']:.3f} & {c['renorm_effect']:.3f} & {c['keep_dice']:.3f}" + r" \\")
    stage_lines.append(r"\midrule")
    for index, c in enumerate(data["seed_pooled_correlations"].values(), 1):
        stage_lines.append(f"S{index} pooled & {c['drop_effect']:.3f} & {c['renorm_effect']:.3f} & {c['keep_dice']:.3f}" + r" \\")
    model_lines=[r"Family & Dice & Parameters (M) & Active TFLOPs \\", r"\midrule"]
    for row in data["schedule_families"]:
        lo,hi=row['active_tflops_range']
        cost=f"{lo:.2f}" if lo==hi else f"{lo:.2f}--{hi:.2f}"
        model_lines.append(f"{row['name']} & {number(row['dice'])} & {row['parameters']/1e6:.2f} & {cost}" + r" \\")
    contrast_lines=[r"Comparison & Difference & Conditional 95\% interval \\", r"\midrule"]
    for row in data["schedule_contrasts"]:
        lo,hi=row["ci95"]
        contrast_lines.append(f"{row['left']} $-$ {row['right']} & {row['point']:+.6f} & [{lo:+.6f}, {hi:+.6f}]" + r" \\")
    numeric=data["numerical"]
    fp_lines=[r"Fold & Positive-ref. drop & All-case drop & All-case keep \\",r"\midrule"]
    for row in data["stage_folds"]:
        c=row["allcase_correlations"]
        fp_lines.append(f"{row['fold']} & {row['correlations']['drop_effect']:.3f} & {c['drop_effect']:.3f} & {c['keep_dice']:.3f}" + r" \\")
    fp_contrasts=[r"Comparison & All-case difference & Conditional 95\% interval \\",r"\midrule"]
    for row in data["schedule_allcase_contrasts"]:
        lo,hi=row["ci95"]
        fp_contrasts.append(f"{row['left']} $-$ {row['right']} & {row['point']:+.6f} & [{lo:+.6f}, {hi:+.6f}]" + r" \\")
    pooling_lines=[r"Pooling & Fold & Dice & Weight SD & Weight--drop $\rho$ & Winner agreement \\",r"\midrule"]
    for row in data["pooling"]:
        pooling_lines.append(f"{row['pooling']} & {row['fold']} & {row['region_macro_dice']:.6f} & {row['mean_branch_weight_std']:.5f} & {row['mean_branch_weight_drop_rho']:.3f} & {100*row['top_weight_forced_winner_agreement']:.1f}\\%" + r" \\")
    macros=(f"\\newcommand{{\\AnalyticResidual}}{{{numeric['analytical_identity_max_residual']:.2e}}}\n"
            f"\\newcommand{{\\ModuleResidual}}{{{numeric['actual_selector_identity_max_residual']:.2e}}}\n")
    for row in data["schedule_families"]:
        key={"Canonical U-Net":"Canonical", "Attention-derived fixed":"Attention", "Drop-derived fixed":"Drop", "Multi-stage SK":"SK", "Multi-stage uniform":"Uniform"}[row["name"]]
        macros+=f"\\newcommand{{\\Schedule{key}Dice}}{{{row['dice']:.6f}}}\n"
    for key,label in (("Attention", "Attention-derived fixed"), ("Drop", "Drop-derived fixed")):
        lo,hi=data["schedule_reporting"]["active_flops_reduction_ranges"][label]
        macros+=f"\\newcommand{{\\Schedule{key}Savings}}{{{100*lo:.1f}--{100*hi:.1f}\\%}}\n"
    norms=data["normalization"]
    norm_lines=[r"Decoder & Primary median $|\Delta|$ & Primary max $|\Delta|$ & Subset BF16 max & Subset FP32 max \\", r"\midrule"]
    norm_frequency_lines=[r"Decoder & $|\Delta|>0.001$ & $|\Delta|>0.01$ & $|\Delta|>0.05$ \\", r"\midrule"]
    for stage in (1,2,3):
        selected={r["mode"]:r for r in norms["endpoints"] if r["stage"]==stage}
        primary=selected["primary_bf16"]
        norm_lines.append(f"{stage} & {primary['median_absolute_difference']:.2e} & {primary['maximum_absolute_difference']:.6f} & "
            f"{selected['sensitivity_bf16']['maximum_absolute_difference']:.6f} & {selected['sensitivity_fp32']['maximum_absolute_difference']:.6f}"+r" \\")
        counts = [f"{r['count']}/{r['total']} ({100*r['proportion']:.1f}\\%)"
                  for r in primary["exceedances"]]
        norm_frequency_lines.append(f"{stage} & " + " & ".join(counts) + r" \\")
    norm_channel_lines=[r"Mode/stage & Channels & Min $v/\epsilon$ & Max post RMS & Max prediction RMS & Floored \\", r"\midrule"]
    for row in norms["channel_distributions"]:
        label={"primary_bf16":"Full BF16", "sensitivity_bf16":"Subset BF16", "sensitivity_fp32":"Subset FP32"}[row["mode"]]
        m=row["metrics"]
        norm_channel_lines.append(f"{label}/{row['stage']} & {row['n']:,} & {m['variance_over_epsilon']['min']:.2e} & "
            f"{m['post_difference_rmse']['max']:.2e} & {m['finite_epsilon_prediction_rmse']['max']:.2e} & {row['below_floor']}"+r" \\")
    within_lines = [r"Fold & Decoder & Drop $\rho$ & Renormalized $\rho$ & Forced-path $\rho$ \\", r"\midrule"]
    for fold in data["stage_folds"]:
        for row in fold["within_stage_correlations"]:
            within_lines.append(f"{fold['fold']} & {row['stage']} & {row['drop_effect']:.3f} & "
                f"{row['renorm_effect']:.3f} & {row['keep_dice']:.3f}" + r" \\")
    range_lines = [r"Fold & Decoder & Weight & Drop effect & Renorm. effect & Forced-path Dice \\", r"\midrule"]
    sensitivity_lines = [r"Check & Positive & Stronger/tie/weaker & Removal $\rho$ & Forced-path $\rho$ \\", r"\midrule"]
    sensitivity_labels = (("all_stages", "All stages"),
                          ("stage_adjusted_ranks", "Stage-adjusted ranks"),
                          ("within_stage_ranks", "Within-stage ranks"),
                          ("leave_stage_1_out", "Omit stage 1"),
                          ("leave_stage_2_out", "Omit stage 2"),
                          ("leave_stage_3_out", "Omit stage 3"),
                          ("within_stage_1", "Stage 1 only"),
                          ("within_stage_2", "Stage 2 only"),
                          ("within_stage_3", "Stage 3 only"))
    for name, label in sensitivity_labels:
        summary = data["stage_sensitivity"]["primary"][name]["summary"]
        ranges = [f"{summary[key]['median']:.3f} [{summary[key]['min']:.3f}, {summary[key]['max']:.3f}]"
                  for key in ("removal_rho", "forced_path_rho")]
        sensitivity_lines.append(f"{label} & {summary['positive_removal']}/15 & "
            f"{summary['stronger']}/{summary['tie']}/{summary['weaker']} & " + " & ".join(ranges) + r" \\")
    for row in data["stage_ranges"]:
        ranges = []
        for key in ("weight", "drop_effect", "renorm_effect", "keep_dice"):
            precision = ".3g" if key in ("drop_effect", "renorm_effect") else ".3f"
            ranges.append("[" + ", ".join(format(value, precision) for value in row[key]) + "]")
        range_lines.append(f"{row['fold']} & {row['stage']} & " + " & ".join(ranges) + r" \\")
    count_lines = [r"Fold & All studies/patients & WT studies/patients & TC studies/patients & ET studies/patients \\", r"\midrule"]
    for row in data["schedule_reporting"]["fold_counts"]:
        counts = [f"{row['positive_reference'][region]['studies']}/{row['positive_reference'][region]['patients']}"
                  for region in ("wt", "tc", "et")]
        count_lines.append(f"{row['fold']} & {row['studies']}/{row['patients']} & " + " & ".join(counts) + r" \\")
    run_lines = [r"Family & Folds & Base seed & Terminal epoch & Config differences \\", r"\midrule"]
    for family in data["schedule_families"]:
        runs = [r for r in data["schedule_reporting"]["runs"] if r["family"] == family["name"]]
        seeds = ", ".join(str(s) for s in sorted({r["recipe"]["seed"] for r in runs}))
        epochs = ", ".join(str(e + 1) for e in sorted({r["terminal_epoch_zero_based"] for r in runs}))
        differences = "None" if all(not r["config_differences"] for r in runs) else "See manifest"
        run_lines.append(f"{family['name']} & 0--4 & {seeds} & {epochs} & {differences}" + r" \\")
    return {"stage.tex": table("lrrr",stage_lines), "schedules.tex": table("lrrr",model_lines),
            "contrasts.tex": table("lrr",contrast_lines), "macros.tex":macros,
            "fp_stage.tex":table("lrrr",fp_lines),"fp_contrasts.tex":table("lrr",fp_contrasts),
            "pooling.tex":table("lrrrrr",pooling_lines),
            "normalization_checkpoint.tex":table("lrrrr",norm_lines),
            "normalization_frequency.tex":table("lrrr",norm_frequency_lines),
            "normalization_channels.tex":table("lrrrrr",norm_channel_lines),
            "normalization_evidence.json":json.dumps(norms,indent=2,sort_keys=True)+"\n",
            "within_stage.tex":multipage_table("llrrr",within_lines,
                "Within-stage four-branch Spearman associations on fifteen terminal SK banks (S1--S3, five folds each). Decoder resolutions are given in Table~\\ref{tab:architecture}.", "tab:within"),
            "stage_sensitivity.tex":table("lrrll", sensitivity_lines),
            "stage_ranges.tex":multipage_table("llrrrr",range_lines,
                "Ranges across four branch summaries within each fold and stage. Effects are learned Dice minus intervened Dice; negative removal effects indicate an improvement. Forced-path values are Dice, not effects.", "tab:ranges"),
            "denominators.tex":table("lrrrr",count_lines),
            "run_summary.tex":table("lrrrr",run_lines),
            "run_reporting.json":json.dumps(data["schedule_reporting"],indent=2,sort_keys=True)+"\n",
            "evidence.json":json.dumps(data,indent=2,sort_keys=True)+"\n"}


def figures(data: dict, curves: list[dict]):
    plt.rcParams.update({"font.size":10, "axes.spines.top":False, "axes.spines.right":False})
    fig, ax=plt.subplots(1,2,figsize=(10,3.7),layout="constrained")
    for scale in (1.1,2.,8.):
        selected=[r for r in curves if r["scale"]==scale]
        ax[0].loglog([r["variance_over_epsilon"] for r in selected], [max(r["relative_change"],1e-16) for r in selected],label=f"scale={scale:g}")
    ax[0].set(xlabel=r"Spatial variance / $\epsilon$",ylabel="Relative change after IN",title="A. Finite-epsilon boundary")
    ax[0].legend(frameon=False)
    arch=data["numerical"]["architecture"]
    labels=["Post IN, ε=0","Post IN, ε=1e−5","No post norm.","Branch IN only","Group norm.","Spatial weights"]
    ax[1].barh(range(len(arch)),[max(r["mean_relative_change"],1e-16) for r in arch])
    ax[1].set_xscale("log"); ax[1].set_yticks(range(len(arch)),labels)
    ax[1].invert_yaxis()
    ax[1].set(xlabel="Mean relative tensor change",title="B. Architectural controls")
    fig.savefig(PAPER/"generated/normalization.pdf");plt.close(fig)
    fig,ax=plt.subplots(3,2,figsize=(9,8.5),layout="constrained",sharex=True,sharey="col")
    for index, seed in enumerate(STAGE_SEEDS):
        for stage in (1,2,3):
            cells=[r for r in data["stage_cells"] if r["stage"]==stage and r["base_seed"]==seed]
            for panel,key in enumerate(("drop_effect","keep_dice")):
                ax[index,panel].scatter([r["weight"] for r in cells],[r[key] for r in cells],label=f"Decoder {stage}",s=25,alpha=.75)
        ax[index,0].set(ylabel="Dice loss after removal",title=f"S{index+1}: removal dependence")
        ax[index,1].set(ylabel="Forced-path Dice",title=f"S{index+1}: keep-one performance")
    for panel in ax[-1]: panel.set_xlabel("Mean channel-averaged weight")
    ax[0,0].legend(frameon=False)
    fig.savefig(PAPER/"generated/interventions.pdf");plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--check",action="store_true");args=parser.parse_args()
    completion=completion_data()
    historical_cells,historical_folds=stage_data()
    cells,folds,seed_pooled,replication=replication_data()
    families,contrasts,allcontrasts,reporting=schedule_data();numeric,curves=numerical_probes();pooling=pooling_data()
    core=[brats_family(*family,range(5),HASHES) for family in FIVE]
    normalization=normalization_data(ROOT,HASHES,sha,cells)
    for path in (Path(__file__),PAPER/"tools/completed_diagnostics.py",PAPER/"tools/check_stage_robustness.py",ROOT/"src/seg3d/multikernel.py",
                 ROOT/"src/seg3d/models.py",ROOT/"src/seg3d/data.py",
                 ROOT/"src/seg3d/complexity.py",ROOT/"environment/pip-jaewon-freeze.txt",
                 PAPER/"main.tex",PAPER/"references.bib",PAPER/"Definitions/mdpi.cls",
                 ROOT/"literature/MATHEMATICS_REFERENCE_EXTENSION_2026-10-05.md",
                 ROOT/"literature/paper_origin/02_Byra_Selective_Kernel_UNet_BSPC_2020.pdf",
                 ROOT/"literature/paper_origin/75_Serrano_Smith_Is_Attention_Interpretable_2019.pdf",
                 ROOT/"literature/paper_origin/76_Ulyanov_Instance_Normalization_2016.pdf",
                 ROOT/"literature/paper_origin/77_Kobayashi_Attention_Vector_Norms_2020.pdf",
                 ROOT/"literature/paper_origin/78_Kobayashi_Residual_Normalization_2021.pdf",
                 ROOT/"literature/paper_origin/79_Chen_Dynamic_Convolution_CVPR_2020.pdf",
                 ROOT/"literature/paper_origin/80_Yang_CondConv_NeurIPS_2019.pdf",
                 ROOT/"literature/paper_origin/81_Chen_ESKNet_ESWA_2024.pdf",
                 ROOT/"literature/paper_origin/82_Shi_SGDC_MIDL_2026.pdf",
                 ROOT/"data/processed/brats2023_gli_crop4/metadata.json",
                 ROOT/"manuscript/tools/build_evidence_tables.py",ROOT/"manuscript/tools/build_statistics.py"):
        HASHES[str(path.relative_to(ROOT))]=sha(path)
    data=dict(schema_version=7,analysis_seed=SEED,bootstrap_draws=DRAWS,
              clinical_estimand="equal-fold mean of study-wise positive-GT region-macro Dice",
              bootstrap="paired patient-cluster resampling within fold; conditional on stored fits",
              stage_folds=folds,stage_cells=cells,stage_ranges=stage_ranges(cells),
              stage_sensitivity=stage_sensitivity_data(cells),
              historical_stage_cells=historical_cells,historical_stage_folds=historical_folds,
              seed_pooled_correlations=seed_pooled,replication=replication,
              schedule_families=families,schedule_contrasts=contrasts,schedule_allcase_contrasts=allcontrasts,
              schedule_reporting=reporting,fold_completion=completion,normalization=normalization,
              allcase_definition="all regions/all studies; Dice=1 for absent reference and empty prediction, else 0 for absent reference",
              pooling=pooling,core_fivefold=core,
              numerical=numeric,input_sha256=HASHES)
    generated=outputs(data,curves)
    if args.check:
        for name,content in generated.items():
            path=PAPER/"generated"/name
            if not path.exists() or path.read_text()!=content:
                raise SystemExit(f"STALE: {path}")
        print("PASS: source hashes, rebuilt clinical results and mathematical probes match")
    else:
        (PAPER/"generated").mkdir(exist_ok=True)
        for name,content in generated.items(): (PAPER/"generated"/name).write_text(content)
        figures(data,curves)
        print(json.dumps({"source_files":len(HASHES),"stage_folds":len(folds),"schedule_contrasts":contrasts,
                          "numerical_residuals":[numeric["analytical_identity_max_residual"],numeric["actual_selector_identity_max_residual"]]},indent=2))


if __name__=="__main__": main()
