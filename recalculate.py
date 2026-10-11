#!/usr/bin/env python3
"""CPU reanalysis from delivered study rows and lossless channel measurements.

No images, checkpoints, pickle, CUDA, or original result tree are read. Expected
summaries are used only for comparisons, never to construct recalculated values.
"""
from __future__ import annotations

from array import array
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

import numpy as np
from scipy.stats import spearmanr

REGIONS = ("wt", "tc", "et")
BRANCHES = ("k3", "k5", "dilated_k3", "anisotropic")
SEEDS = (2026, 20261001, 20261011)
MODES = ("primary_bf16", "sensitivity_bf16", "sensitivity_fp32")
METRICS = ("variance_over_epsilon", "surviving_mass_native", "pre_scale_relative",
           "post_difference_rmse", "finite_epsilon_prediction_rmse",
           "replay_difference_rms_eps_0", "replay_difference_rms_eps_1e-05",
           "replay_identity_max_error_eps_1e-05")
PRIMITIVES = ("variance", "epsilon", "scale", "finite_epsilon_ratio")
# Packed little-endian numeric records, with no executable/object payload.
CHANNEL_DTYPE = np.dtype([("case", "<u2"), ("branch", "u1"), ("channel", "u1"),
                          ("zero_mass", "u1"), ("below_floor", "u1")]
                         + [(key, "<f8") for key in METRICS + PRIMITIVES])
CHUNK_ROWS = 65536


def safe_path(root, name):
    relative = PurePosixPath(name)
    path = root / name
    if (relative.is_absolute() or ".." in relative.parts or "\\" in name
            or any(p.startswith(".") for p in relative.parts)
            or path.is_symlink() or not path.resolve().is_relative_to(root.resolve())):
        raise ValueError("Unsafe numerical data path: " + name)
    return path


def science_digest(evidence):
    scientific = {k: v for k, v in evidence.items() if k != "input_sha256"}
    return hashlib.sha256(json.dumps(scientific, sort_keys=True, allow_nan=False).encode()).hexdigest()


class Data:
    def __init__(self, root, evidence):
        self.root = root
        self.manifest = json.loads((root / "manifest.json").read_text())
        if self.manifest["scientific_evidence_sha256"] != science_digest(evidence):
            raise ValueError("Numerical export belongs to a different scientific snapshot")
        if self.manifest["channel_dtype"] != [list(row) for row in CHANNEL_DTYPE.descr]:
            raise ValueError("Unexpected packed-channel schema")
        if len(set(self.manifest["studies"])) != len(self.manifest["studies"]):
            raise ValueError("Duplicate study index")
        for name, record in self.manifest["files"].items():
            content = safe_path(root, name).read_bytes()
            if len(content) != record["bytes"] or hashlib.sha256(content).hexdigest() != record["sha256"]:
                raise ValueError("Changed numerical data: " + name)
        for source, expected in self.manifest["original_sources"].items():
            if evidence["input_sha256"].get(source) != expected:
                raise ValueError("Export source differs from manuscript ledger: " + source)

    def csv(self, name):
        if name not in self.manifest["files"]:
            raise ValueError("Unlisted CSV")
        with safe_path(self.root, name).open(newline="") as stream:
            result = list(csv.DictReader(stream))
        if len(result) != self.manifest["files"][name]["rows"]:
            raise ValueError("CSV row-count mismatch")
        return result

    def source(self, original):
        return self.csv(self.manifest["csv_sources"][original])

    def channels(self, name):
        if name not in self.manifest["files"]:
            raise ValueError("Unlisted channel chunk")
        record = self.manifest["files"][name]
        if record["bytes"] != record["rows"] * CHANNEL_DTYPE.itemsize or record["rows"] > CHUNK_ROWS:
            raise ValueError("Invalid channel chunk length")
        result = np.fromfile(safe_path(self.root, name), dtype=CHANNEL_DTYPE)
        if (np.any(result["case"] >= len(self.manifest["studies"]))
                or np.any(result["branch"] >= 4) or np.any(result["zero_mass"] > 1)
                or np.any(result["below_floor"] > 1)):
            raise ValueError("Invalid channel identity/flag")
        for field in METRICS + PRIMITIVES:
            if np.isinf(result[field]).any():
                raise ValueError("Infinite diagnostic value")
        return result


def patient(case):
    if not case.startswith("BraTS-GLI-") or case.count("-") < 3:
        raise ValueError("Unexpected de-identified challenge ID")
    return case.rsplit("-", 1)[0]


def flag(row, region):
    value = row["gt_positive_" + region]
    if value not in (True, False, "True", "False"):
        raise ValueError("Invalid reference-presence flag")
    return value is True or value == "True"


def macro(rows, prefix="", allcase=False):
    means = []
    for region in REGIONS:
        values = []
        for row in rows:
            if flag(row, region):
                value = float(row[prefix + "dice_" + region])
                if not np.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError("Invalid Dice")
                values.append(value)
            elif allcase:
                values.append(float(int(row[prefix + "predicted_voxels_" + region]) == 0))
        if not values:
            raise ValueError("Empty positive-reference region")
        means.append(sum(values) / len(values))
    return float(np.mean(means))


def groups(rows, expected_ids, canonical_pool=False):
    result = defaultdict(list)
    for row in rows:
        if row["case_id"].startswith("__"):
            continue
        if row["checkpoint"] != "checkpoints/latest.pt":
            raise ValueError("Nonterminal metric")
        condition = row.get("condition", "learned")
        if canonical_pool and row.get("kind") == "drop_branch" and row["renormalize"] == "False":
            condition = "drop_" + row["intervention_branch"]
        result[condition].append(row)
    for rr in result.values():
        if len(rr) != len(expected_ids) or {r["case_id"] for r in rr} != set(expected_ids):
            raise ValueError("Unpaired/duplicate study or incorrect fold membership")
    if "learned" not in result:
        raise ValueError("Missing learned baseline")
    return result


def cluster_scores(rows, patients, indices, allcase=False):
    matrix = np.zeros((len(patients), 3, 2), dtype=np.float64)
    lookup = {p: i for i, p in enumerate(patients)}
    for row in rows:
        i = lookup[patient(row["case_id"])]
        for j, region in enumerate(REGIONS):
            if flag(row, region) or allcase:
                value = (float(row["dice_" + region]) if flag(row, region)
                         else float(int(row["predicted_voxels_" + region]) == 0))
                if not np.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError("Invalid bootstrap Dice")
                matrix[i, j] += (value, 1)
    totals = matrix.sum(axis=0)
    sampled = matrix[indices].sum(axis=1)
    if np.any(totals[:, 1] == 0) or np.any(sampled[:, :, 1] == 0):
        raise ValueError("Zero positive-reference bootstrap draw")
    return float(np.mean(totals[:, 0] / totals[:, 1])), np.mean(sampled[:, :, 0] / sampled[:, :, 1], axis=1)


def comparison(data, evidence, splits, check):
    points, draws, apoints, adraws = (defaultdict(list) for _ in range(4))
    rng = np.random.default_rng(evidence["analysis_seed"])
    for fold, split in enumerate(splits):
        patients = sorted({patient(c) for c in split["val"]})
        indices = rng.integers(0, len(patients), size=(evidence["bootstrap_draws"], len(patients)), dtype=np.int32)
        for family in evidence["schedule_families"]:
            detail = family["folds"][fold]
            base = detail["base"]
            suffix = "final_epoch" if "mathematics_fold_completion_v1" in base else "final"
            rows = groups(data.source(base + "/metrics/fold_validation_" + suffix + ".csv"), split["val"])["learned"]
            check(macro(rows), detail["dice"], "schedule fold Dice")
            for field, output in (("parameters", "parameters"), ("active_flops_per_volume", "active_flops")):
                values = {float(r[field]) for r in rows}
                if len(values) != 1:
                    raise ValueError("Inconsistent compute profile")
                check(next(iter(values)), detail[output], "recorded compute profile")
            p, sample = cluster_scores(rows, patients, indices)
            ap, asample = cluster_scores(rows, patients, indices, True)
            points[family["name"]].append(p); draws[family["name"]].append(sample)
            apoints[family["name"]].append(ap); adraws[family["name"]].append(asample)
    for field, pp, dd in (("schedule_contrasts", points, draws), ("schedule_allcase_contrasts", apoints, adraws)):
        for target in evidence[field]:
            left, right = target["left"], target["right"]
            actual = dict(left=left, right=right, point=float(np.mean(pp[left]) - np.mean(pp[right])),
                          ci95=np.quantile(np.mean(dd[left], axis=0) - np.mean(dd[right], axis=0), [.025, .975]).tolist())
            check(actual, target, field)
    for family in evidence["core_fivefold"]:
        values = []
        for fold, detail in enumerate(family["folds"]):
            path = f"results/brats2023_gli/{family['trainer']}/fold_{fold}/metrics/fold_validation_final.csv"
            rows = groups(data.source(path), splits[fold]["val"])["learned"]
            values.append(macro(rows)); check(values[-1], detail["dice"], "core fold Dice")
        check(float(np.mean(values)), family["dice"], "core fivefold Dice")
    return dict(patient_bootstrap_intervals=10, draws=evidence["bootstrap_draws"], schedule_fits=25, core_fits=35)


def stages(data, evidence, splits, check):
    allcells, allfolds, historical_cells, historical_folds = [], [], [], []
    for seed in (None,) + SEEDS:
        cells, folds = [], []
        for fold in range(3 if seed is None else 5):
            base = f"results/brats2023_gli/BraTS23GLIMKMultiStageSK100Trainer/fold_{fold}"
            if seed is not None:
                study = "kernel_stage_thirdseed_v1" if seed == SEEDS[-1] else "kernel_stage_fivefold_twoseed_v1"
                base = f"results/{study}/brats2023_gli/KernelStageSK100Seed{seed}Trainer/fold_{fold}"
            weights = defaultdict(dict)
            suffix = "final" if seed is None else "final_epoch"
            for row in data.source(base + f"/metrics/selector_validation_{suffix}.csv"):
                if row["stage"] == "aggregate":
                    continue
                key = (row["stage"], row["branch"])
                if row["checkpoint"] != "checkpoints/latest.pt" or row["case_id"] in weights[key]:
                    raise ValueError("Invalid/duplicate selector row")
                weights[key][row["case_id"]] = float(row["weight"])
            conditions = {key: groups(data.source(base + "/metrics/" + folder + "/cases.csv"), splits[fold]["val"])
                for key, folder in (("drop", "stage_selector_interventions_final_epoch"),
                                    ("renorm", "stage_selector_interventions_reroute_final_epoch"),
                                    ("keep", "stage_selector_keep_one_final_epoch"))}
            learned = conditions["drop"]["learned"]
            baseline, baseline_all = macro(learned), macro(learned, allcase=True)
            for key in ("renorm", "keep"):
                check(macro(conditions[key]["learned"]), baseline, "paired learned baseline")
            this = []
            for stage in (1, 2, 3):
                for branch in BRANCHES:
                    ww = weights[(f"decoder_{stage}", branch)]
                    if set(ww) != set(splits[fold]["val"]):
                        raise ValueError("Selector fold membership mismatch")
                    drop = conditions["drop"][f"drop_decoder_{stage}_{branch}"]
                    renorm = conditions["renorm"][f"drop_decoder_{stage}_{branch}_and_reroute"]
                    keep = conditions["keep"][f"keep_decoder_{stage}_{branch}"]
                    this.append(dict(fold=fold, stage=stage, branch=branch, weight=float(np.mean(list(ww.values()))),
                        learned=baseline, drop_effect=baseline-macro(drop), renorm_effect=baseline-macro(renorm),
                        keep_dice=macro(keep), allcase_drop_effect=baseline_all-macro(drop, allcase=True),
                        allcase_keep_dice=macro(keep, allcase=True)))
            keys = ("drop_effect", "renorm_effect", "keep_dice")
            def rho(rr, key):
                return float(spearmanr([r["weight"] for r in rr], [r[key] for r in rr]).statistic)
            folds.append(dict(fold=fold, studies=len(learned), patients=len({patient(r["case_id"]) for r in learned}),
                correlations={key: rho(this, key) for key in keys},
                within_stage_correlations=[dict(stage=s, **{key: rho([r for r in this if r["stage"] == s], key) for key in keys}) for s in (1, 2, 3)],
                allcase_correlations={key: rho(this, "allcase_" + key) for key in ("drop_effect", "keep_dice")}))
            cells.extend(this)
        if seed is None:
            historical_cells, historical_folds = cells, folds
        else:
            for row in cells + folds:
                row.update(fold_index=row["fold"], base_seed=seed, fold=f"S{SEEDS.index(seed)+1}/{row['fold']}")
            check({key: float(spearmanr([r['weight'] for r in cells], [r[key] for r in cells]).statistic)
                   for key in keys}, evidence["seed_pooled_correlations"][str(seed)], "pooled seed ranks")
            allcells.extend(cells); allfolds.extend(folds)
    for actual, field in ((allcells, "stage_cells"), (allfolds, "stage_folds"),
                          (historical_cells, "historical_stage_cells"), (historical_folds, "historical_stage_folds")):
        check(actual, evidence[field], field)
    from check_stage_robustness import analyze_cells
    check(analyze_cells(allcells), evidence["stage_sensitivity"], "raw-row stage sensitivity")
    return dict(primary_stage_cells=len(allcells), historical_stage_cells=len(historical_cells))


def pooling(data, evidence, splits, check):
    sources = data.manifest["csv_sources"]
    for target in evidence["pooling"]:
        fold = target["fold"]
        trainer = "BraTS23GLIMKSoftControlTrainer" if target["pooling"] == "GAP" else "BraTS23GLIMKSoftAvgMax100Trainer"
        endings = f"/{trainer}/fold_{fold}/metrics/selector_interventions_final_epoch/cases.csv"
        candidates = [name for name in sources if name.endswith(endings)]
        if len(candidates) != 1:
            raise ValueError("Ambiguous pooling source")
        base = candidates[0].removesuffix("cases.csv")
        gg = groups(data.source(base + "cases.csv"), splits[fold]["val"], True)
        cases = {k: {r["case_id"]: r for r in v} for k, v in gg.items()}
        weights = defaultdict(dict)
        for row in data.source(base + "selector_signals.csv"):
            if row["condition"] == "learned" and row.get("stage", "aggregate") == "aggregate":
                if row["branch"] in weights[row["case_id"]]:
                    raise ValueError("Duplicate pooling selector")
                weights[row["case_id"]][row["branch"]] = float(row["weight"])
        ids = sorted(cases["learned"])
        if set(weights) != set(ids) or any(set(v) != set(BRANCHES) for v in weights.values()):
            raise ValueError("Incomplete pooling selector")
        ww = np.array([[weights[i][b] for b in BRANCHES] for i in ids])
        forced = np.array([[float(cases[f"force_{b}"][i]["mean_dice"]) for b in BRANCHES] for i in ids])
        dropped = np.array([[float(cases[f"drop_{b}"][i]["mean_dice"]) for b in BRANCHES] for i in ids])
        learned = np.array([float(cases["learned"][i]["mean_dice"]) for i in ids])
        actual = dict(studies=len(ids), region_macro_dice=macro(gg["learned"]),
            study_macro_available_region_dice=float(learned.mean()), mean_branch_weight_std=float(ww.std(axis=0).mean()),
            mean_branch_weight_drop_rho=float(np.mean([spearmanr(ww[:, j], learned-dropped[:, j]).statistic for j in range(4)])),
            top_weight_forced_winner_agreement=float(np.mean(ww.argmax(axis=1) == forced.argmax(axis=1))))
        check(actual, {key: target[key] for key in actual}, "pooling study rows")
    return dict(pooling_fits=len(evidence["pooling"]))


def normalization(data, evidence, splits, check):
    from completed_diagnostics import exceedance_summary
    collected = defaultdict(lambda: defaultdict(lambda: array("d")))
    counts = defaultdict(lambda: dict(n=0, zero_mass=0, below_floor=0))
    cells, links = [], []
    ratio_residual = ratio_relative_residual = 0.
    studies = data.manifest["studies"]
    jobkeys = {(j["base_seed"], j["fold"]) for j in data.manifest["normalization"]}
    if len(data.manifest["normalization"]) != 15 or jobkeys != {(s, f) for s in SEEDS for f in range(5)}:
        raise ValueError("Incomplete normalization bank grid")
    for job in data.manifest["normalization"]:
        seed, fold = job["base_seed"], job["fold"]
        for mode in MODES:
            expected_ids = set(splits[fold]["val"]) if mode == MODES[0] else set(job["sensitivity_studies"])
            rows = data.csv(job["modes"][mode]["cases"])
            identities = {(r["case_id"], int(r["stage"]), r["branch"]) for r in rows}
            grid = {(c, s, b) for c in expected_ids for s in (1, 2, 3) for b in BRANCHES}
            if len(identities) != len(rows) or identities != grid:
                raise ValueError("Incomplete normalization study grid")
            local = defaultdict(lambda: dict(n=0, post_max=0., prediction_max=0., pre_max=0., min_variance_over_epsilon=float("inf"), below_floor=0))
            for stage in (1, 2, 3):
                expected_channels = {1: 128, 2: 64, 3: 32}[stage]
                seen = set()
                for name in job["modes"][mode]["channels"][str(stage)]:
                    chunk = data.channels(name)
                    for row in chunk:
                        identity = (studies[int(row["case"])], stage, BRANCHES[int(row["branch"])])
                        key = (*identity, int(row["channel"]))
                        if key in seen or identity not in identities or not 0 <= key[-1] < expected_channels:
                            raise ValueError("Duplicate or invalid channel identity")
                        seen.add(key)
                        group = local[identity]
                        group["n"] += 1; group["below_floor"] += int(row["below_floor"])
                        for out, source in (("post_max", "post_difference_rmse"), ("prediction_max", "finite_epsilon_prediction_rmse"), ("pre_max", "pre_scale_relative")):
                            group[out] = max(group[out], float(row[source]) if np.isfinite(row[source]) else 0.)
                        group["min_variance_over_epsilon"] = min(group["min_variance_over_epsilon"], float(row["variance_over_epsilon"]))
                    count = counts[(mode, stage)]
                    count["n"] += len(chunk); count["zero_mass"] += int(chunk["zero_mass"].sum()); count["below_floor"] += int(chunk["below_floor"].sum())
                    for metric in METRICS:
                        collected[(mode, stage)][metric].frombytes(chunk[metric][np.isfinite(chunk[metric])].astype(np.float64).tobytes())
                    valid = np.isfinite(chunk["scale"]) & np.isfinite(chunk["finite_epsilon_ratio"])
                    if np.any(chunk["variance"][valid] < 0) or np.any(chunk["epsilon"][valid] <= 0) or np.any(chunk["scale"][valid] <= 0):
                        raise ValueError("Invalid finite-epsilon primitives")
                    ratio = np.sqrt((chunk["variance"][valid] + chunk["epsilon"][valid]) /
                                    (chunk["variance"][valid] + chunk["epsilon"][valid] / chunk["scale"][valid]**2))
                    if len(ratio):
                        errors = np.abs(ratio - chunk["finite_epsilon_ratio"][valid])
                        ratio_residual = max(ratio_residual, float(errors.max()))
                        ratio_relative_residual = max(ratio_relative_residual,
                            float(np.max(errors / np.maximum(1., np.abs(chunk["finite_epsilon_ratio"][valid])))))
                if len(seen) != len(expected_ids) * 4 * expected_channels:
                    raise ValueError("Incomplete channel grid")
            for stage in (1, 2, 3):
                for branch in BRANCHES:
                    rr = [r for r in rows if int(r["stage"]) == stage and r["branch"] == branch]
                    cell = dict(base_seed=seed, fold=fold, mode=mode, stage=stage, branch=branch, studies=len(rr))
                    for policy in ("drop", "renorm"):
                        for region in REGIONS:
                            vv = [float(r[policy + "_dice_" + region]) for r in rr if flag(r, region)]
                            if not vv or not np.isfinite(vv).all() or min(vv) < 0 or max(vv) > 1:
                                raise ValueError("Invalid normalization Dice")
                            cell[policy + "_dice_" + region] = sum(vv) / len(vv)
                        cell[policy + "_region_macro_dice"] = float(np.mean([cell[policy + "_dice_" + r] for r in REGIONS]))
                    cell["renorm_minus_drop"] = cell["renorm_region_macro_dice"] - cell["drop_region_macro_dice"]
                    cell["mask_disagreement_mean"] = float(np.mean([np.mean([float(r["mask_disagreement_" + region]) for region in REGIONS]) for r in rr]))
                    cells.append(cell)
            for row in rows:
                regions = [r for r in REGIONS if flag(row, r)]
                delta = float(np.mean([float(row["renorm_dice_" + r]) - float(row["drop_dice_" + r]) for r in regions])) if regions else None
                links.append(dict(base_seed=seed, fold=fold, mode=mode, case_id=row["case_id"], stage=int(row["stage"]), branch=row["branch"],
                    renorm_minus_drop_available_region_dice=delta, **local[(row["case_id"], int(row["stage"]), row["branch"]) ]))
        print(f"Recalculated normalization seed={seed} fold={fold}", flush=True)
    def cellkey(r):
        return (r["base_seed"], r["fold"], MODES.index(r["mode"]), r["stage"], BRANCHES.index(r["branch"]))
    check(sorted(cells, key=cellkey), sorted(evidence["normalization"]["cells"], key=cellkey), "raw normalization endpoints")
    for target in evidence["normalization"]["channel_distributions"]:
        key = (target["mode"], target["stage"])
        mm = {metric: dict(zip(("min", "median", "p95", "max"), np.quantile(np.frombuffer(vv, dtype=np.float64), [0, .5, .95, 1]).tolist()))
              for metric, vv in collected[key].items() if len(vv)}
        check(dict(mode=key[0], stage=key[1], **counts[key], metrics=mm), target, "raw channel quantiles")
    for target in evidence["normalization"]["endpoints"]:
        rr = [r for r in cells if (r["mode"], r["stage"]) == (target["mode"], target["stage"])]
        check(exceedance_summary(rr), target["exceedances"], "raw normalization frequencies")
        changes = [abs(r["renorm_minus_drop"]) for r in rr]
        check(float(np.median(changes)), target["median_absolute_difference"], "raw endpoint median")
        check(max(changes), target["maximum_absolute_difference"], "raw endpoint maximum")
    links.sort(key=lambda r: abs(r["renorm_minus_drop_available_region_dice"] or 0), reverse=True)
    check(links[:30], evidence["normalization"]["largest_case_connections"], "raw case/channel links")
    # Ratios can approach 1e8 in nearly zero-variance channels: compare using
    # a mixed tolerance, not an absolute threshold smaller than a float64 ULP.
    if ratio_relative_residual > 1e-12:
        raise ValueError("Finite-epsilon identity differs from recorded ratios")
    return dict(normalization_cells=len(cells), channel_records=sum(r["n"] for r in counts.values()),
                channel_distribution_groups=len(counts), finite_epsilon_ratio_max_residual=ratio_residual,
                finite_epsilon_ratio_max_scaled_residual=ratio_relative_residual,
                finite_epsilon_ratio_scaled_tolerance=1e-12)


def recalculate(root, data_root=None):
    paper = root / "manuscript/mathematics"
    sys.path[:0] = [str(paper / "tools"), str(paper / "reproducibility")]
    from verify_package import assert_close
    evidence = json.loads((paper / "generated/evidence.json").read_text())
    data = Data(data_root or root / "analysis_data", evidence)
    split_path = root / "data_contract/splits.json"
    if not split_path.exists():
        split_path = root / "data/processed/brats2023_gli_crop4/splits.json"
    splits = json.loads(split_path.read_text())
    result = comparison(data, evidence, splits, assert_close)
    result.update(stages(data, evidence, splits, assert_close))
    result.update(pooling(data, evidence, splits, assert_close))
    result.update(normalization(data, evidence, splits, assert_close))
    return dict(status="passed", **result,
                boundary="Reanalysis of recorded numerical outputs, not new image inference, training, or independent replication")
