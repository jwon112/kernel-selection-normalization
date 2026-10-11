from __future__ import annotations


import argparse


import csv


import hashlib


import json


from pathlib import Path


import numpy as np


from build_evidence_tables import FIVE, REGIONS, ROOT, sha


def patient_id(case_id: str) -> str:
    if not case_id.startswith("BraTS-GLI-") or case_id.count("-") < 3:
        raise ValueError(f"Unexpected BraTS study ID: {case_id}")
    return case_id.rsplit("-", 1)[0]


def cluster_matrix(by_case: dict, patients: list[str]) -> np.ndarray:
    locations = {patient: i for i, patient in enumerate(patients)}
    matrix = np.zeros((len(patients), len(REGIONS), 2), dtype=np.float64)
    for case_id, row in by_case.items():
        i = locations[patient_id(case_id)]
        for j, region in enumerate(REGIONS):
            flag = row[f"gt_positive_{region}"]
            if flag not in ("True", "False"):
                raise ValueError(f"Unexpected GT flag in {case_id}: {flag!r}")
            if flag == "True":
                score = float(row[f"dice_{region}"])
                if not np.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError(f"Invalid Dice in {case_id}: {score}")
                matrix[i, j, 0] += score
                matrix[i, j, 1] += 1
    return matrix


def fold_scores(matrix: np.ndarray, indices: np.ndarray) -> tuple[float, np.ndarray]:
    point = matrix.sum(axis=0)
    if np.any(point[:, 1] == 0):
        raise ValueError("Region with no GT-positive studies")
    point_score = float(np.mean(point[:, 0] / point[:, 1]))
    sampled = matrix[indices].sum(axis=1)
    if np.any(sampled[:, :, 1] == 0):
        raise ValueError("Bootstrap draw with no GT-positive studies")
    draw_scores = np.mean(sampled[:, :, 0] / sampled[:, :, 1], axis=1)
    return point_score, draw_scores

