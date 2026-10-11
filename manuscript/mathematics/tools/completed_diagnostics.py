"""Read-only reconstruction of the completed checkpoint-normalization audit.

The expensive GPU measurements are immutable inputs. Stream their channel
records, verify their completion hashes, and retain exact distribution summaries.
"""
from array import array
from collections import defaultdict
import hashlib
import json

import numpy as np


def exceedance_summary(cells, thresholds=(.001, .01, .05)):
    """Descriptive strict exceedances of absolute bank-level endpoint changes."""
    if not cells:
        raise ValueError("An exceedance summary needs endpoint cells")
    result = []
    for threshold in thresholds:
        selected = [c for c in cells if abs(c["renorm_minus_drop"]) > threshold]
        groups = {}
        for field in ("base_seed", "fold", "branch"):
            groups[field] = {str(value): sum(c[field] == value for c in selected)
                             for value in sorted({c[field] for c in cells})}
        result.append(dict(threshold=threshold, count=len(selected), total=len(cells),
                           proportion=len(selected) / len(cells), groups=groups,
                           cells=selected))
    return result


def normalization_data(root, hashes, sha, stage_cells):
    out = root / "results/kernel_normalization_audit_v1"

    def read(path):
        hashes[str(path.relative_to(root))] = sha(path)
        return json.loads(path.read_text())

    plan = read(out / "queue/plan.json")
    aggregate = read(out / "analysis/summary.json")
    if aggregate["status"] != "complete" or aggregate["models"] != 15:
        raise ValueError("Normalization audit is not complete")
    expected_jobs = {(s, f) for s in (2026, 20261001, 20261011) for f in range(5)}
    if {(j["base_seed"], j["fold"]) for j in plan["jobs"]} != expected_jobs:
        raise ValueError("Normalization matrix mismatch")
    channel_values = defaultdict(lambda: defaultdict(lambda: array("d")))
    channel_counts = defaultdict(lambda: dict(n=0, zero_mass=0, below_floor=0))
    metrics = ("variance_over_epsilon", "surviving_mass_native", "pre_scale_relative",
               "post_difference_rmse", "finite_epsilon_prediction_rmse",
               "replay_difference_rms_eps_0", "replay_difference_rms_eps_1e-05",
               "replay_identity_max_error_eps_1e-05")
    cells, jobs, case_links = [], [], []
    fidelity_max = defaultdict(float)
    primary_lookup = {(r["base_seed"], r["fold_index"], r["stage"], r["branch"]): r
                      for r in stage_cells}
    splits = read(root / "data/processed/brats2023_gli_crop4/splits.json")
    # Verify every case/channel artifact in one pass rather than reading GB twice.
    def stream(path, expected):
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for line in source:
                digest.update(line)
                yield json.loads(line)
        actual = digest.hexdigest()
        if actual != expected:
            raise ValueError(f"Changed normalization artifact: {path}")
        hashes[str(path.relative_to(root))] = actual

    for job in plan["jobs"]:
        status = read(out / "queue" / (job["job"] + ".json"))
        if status["phase"] != "complete":
            raise ValueError("Incomplete normalization job")
        base = root / status["attempt_dir"]
        complete = read(base / "complete.json")
        if complete["status"] != "complete" or (complete["base_seed"], complete["fold"]) != (job["base_seed"], job["fold"]):
            raise ValueError("Normalization completion identity mismatch")
        checkpoint = str((root / job["fold_dir"] / "checkpoints/latest.pt").relative_to(root))
        if hashes.get(checkpoint) != complete["checkpoint_sha256"]:
            raise ValueError("Normalization and stage replication checkpoint hashes disagree")
        patients = {c.rsplit("-", 1)[0] for c in splits[job["fold"]]["val"]}
        selected = set(sorted(patients, key=lambda p: hashlib.sha256(("kernel_normalization_audit_v1:" + p).encode()).hexdigest())[:2])
        selected_studies = {c for c in splits[job["fold"]]["val"] if c.rsplit("-", 1)[0] in selected}
        if set(complete["sensitivity_studies"]) != selected_studies:
            raise ValueError("Precision subset differs from outcome-independent hash selection")
        if aggregate["job_completion_hashes"][job["job"]] != sha(base / "complete.json"):
            raise ValueError("Aggregate references a different normalization attempt")
        for filename, expected in complete["artifacts"].items():
            if filename.endswith(".jsonl"):
                continue
            if sha(base / filename) != expected:
                raise ValueError("Changed normalization artifact: " + filename)
            read(base / filename)
        fidelity = json.loads((base / "fidelity.json").read_text())
        if fidelity["status"] != "passed":
            raise ValueError("Observation hook fidelity failed")
        for row in fidelity["checks"] + json.loads((base / "learned_fidelity.json").read_text())["checks"]:
            for key in ("repeat_logit_max", "hook_logit_max", "repeat_dice_max", "hook_dice_max", "historical_dice_max"):
                fidelity_max[key] = max(fidelity_max[key], row.get(key, 0))
        job_cells = json.loads((base / "cells.json").read_text())
        # The evaluator records a flat array of endpoint cells.
        if isinstance(job_cells, dict):
            job_cells = job_cells["cells"]
        if len(job_cells) != 36 or len({(r["mode"], r["stage"], r["branch"]) for r in job_cells}) != 36:
            raise ValueError("Incomplete or duplicate normalization endpoint cells")
        grouped_channels = defaultdict(lambda: dict(n=0, post_max=0., prediction_max=0., pre_max=0., min_variance_over_epsilon=float("inf"), below_floor=0))
        ncases = nchannels = 0
        for mode in ("primary_bf16", "sensitivity_bf16", "sensitivity_fp32"):
            expected_ids = set(splits[job["fold"]]["val"]) if mode == "primary_bf16" else set(complete["sensitivity_studies"])
            case_rows = list(stream(base / (mode + "_cases.jsonl"), complete["artifacts"][mode + "_cases.jsonl"]))
            identities = {(r["case_id"], r["stage"], r["branch"]) for r in case_rows}
            if len(identities) != len(case_rows) or identities != {(i, s, b) for i in expected_ids for s in (1, 2, 3) for b in ("k3", "k5", "dilated_k3", "anisotropic")}:
                raise ValueError("Incomplete normalization case grid")
            ncases += len(case_rows)
            for cell in (r for r in job_cells if r["mode"] == mode):
                rr = [r for r in case_rows if r["stage"] == cell["stage"] and r["branch"] == cell["branch"]]
                scores = {}
                for policy in ("drop", "renorm"):
                    region_scores = [np.mean([r[policy + "_dice_" + region] for r in rr if r["gt_positive_" + region]])
                                     for region in ("wt", "tc", "et")]
                    scores[policy] = float(np.mean(region_scores))
                    if not np.isclose(scores[policy], cell[policy + "_region_macro_dice"], rtol=0, atol=1e-12):
                        raise ValueError("Case reconstruction differs from normalization cell")
                if mode == "primary_bf16":
                    reference = primary_lookup[(job["base_seed"], job["fold"], cell["stage"], cell["branch"])]
                    for policy, effect in (("drop", "drop_effect"), ("renorm", "renorm_effect")):
                        if not np.isclose(scores[policy], reference["learned"] - reference[effect], rtol=0, atol=1e-12):
                            raise ValueError("Diagnostic changed original full-cohort endpoint")
                cells.append({**cell, "base_seed": job["base_seed"], "fold": job["fold"],
                              "renorm_minus_drop": scores["renorm"] - scores["drop"]})
            seen = set()
            for row in stream(base / (mode + "_channels.jsonl"), complete["artifacts"][mode + "_channels.jsonl"]):
                identity = (row["case_id"], row["stage"], row["branch"], row["channel"])
                if identity in seen or identity[:3] not in identities or not 0 <= row["channel"] < {1: 128, 2: 64, 3: 32}[row["stage"]]:
                    raise ValueError("Duplicate or invalid normalization channel")
                seen.add(identity)
                key = (mode, row["stage"])
                count = channel_counts[key]
                count["n"] += 1
                count["zero_mass"] += int(row["zero_mass"])
                count["below_floor"] += int(row["below_floor"])
                for metric in metrics:
                    value = row[metric]
                    if value is not None:
                        if not np.isfinite(value):
                            raise ValueError("Nonfinite channel diagnostic")
                        channel_values[key][metric].append(value)
                local = grouped_channels[(mode, *identity[:3])]
                local["n"] += 1
                for name, source in (("post_max", "post_difference_rmse"), ("prediction_max", "finite_epsilon_prediction_rmse"), ("pre_max", "pre_scale_relative")):
                    local[name] = max(local[name], row[source] or 0.)
                local["min_variance_over_epsilon"] = min(local["min_variance_over_epsilon"], row["variance_over_epsilon"])
                local["below_floor"] += int(row["below_floor"])
            if len(seen) != len(expected_ids) * 4 * (128 + 64 + 32):
                raise ValueError("Incomplete normalization channel grid")
            nchannels += len(seen)
            # Retain case-level connections, not just extrema across different cases.
            for row in case_rows:
                present = [r for r in ("wt", "tc", "et") if row["gt_positive_" + r]]
                delta = float(np.mean([row["renorm_dice_" + r] - row["drop_dice_" + r] for r in present])) if present else None
                case_links.append(dict(base_seed=job["base_seed"], fold=job["fold"], mode=mode,
                    case_id=row["case_id"], stage=row["stage"], branch=row["branch"],
                    renorm_minus_drop_available_region_dice=delta,
                    **grouped_channels[(mode, row["case_id"], row["stage"], row["branch"]) ]))
        if ncases != complete["paired_case_rows"] or nchannels != complete["channel_rows"]:
            raise ValueError("Completion row counts disagree")
        jobs.append(dict(base_seed=job["base_seed"], fold=job["fold"],
                         attempt=str(base.relative_to(root)), checkpoint_sha256=complete["checkpoint_sha256"],
                         sensitivity_studies=complete["sensitivity_studies"], gpu=complete["gpu"], torch=complete["torch"]))
        print(f"Verified normalization seed={job['base_seed']} fold={job['fold']}", flush=True)
    if len(cells) != 540:
        raise ValueError("Expected 180 endpoint cells in each of three modes")
    distributions = []
    for (mode, stage), count in sorted(channel_counts.items()):
        summary = {}
        for metric, values in channel_values[(mode, stage)].items():
            q = np.quantile(np.frombuffer(values, dtype=np.float64), [0, .5, .95, 1])
            summary[metric] = dict(zip(("min", "median", "p95", "max"), q.tolist()))
        distributions.append(dict(mode=mode, stage=stage, **count, metrics=summary))
    endpoint_summary = []
    for mode in ("primary_bf16", "sensitivity_bf16", "sensitivity_fp32"):
        for stage in (1, 2, 3):
            subset = [c for c in cells if c["mode"] == mode and c["stage"] == stage]
            largest = max(subset, key=lambda c: abs(c["renorm_minus_drop"]))
            endpoint_summary.append(dict(mode=mode, stage=stage, cells=len(subset),
                median_absolute_difference=float(np.median([abs(c["renorm_minus_drop"]) for c in subset])),
                maximum_absolute_difference=abs(largest["renorm_minus_drop"]),
                exceedances=exceedance_summary(subset),
                largest_cell=largest))
    case_links.sort(key=lambda r: abs(r["renorm_minus_drop_available_region_dice"] or 0), reverse=True)
    return dict(models=15, cells=cells, endpoints=endpoint_summary, channel_distributions=distributions,
                jobs=jobs, fidelity_max=dict(fidelity_max), largest_case_connections=case_links[:30],
                boundary="Full-cohort bf16 primary; matched batch-one bf16/fp32 sensitivity on two fixed patients per fold. No isolated-channel causal attribution or normalization-placement retraining.")
