"""Locked input conditions and patient-grouped inner split for Branch P."""

from __future__ import annotations

import hashlib

import torch
from torch.utils.data import Dataset


CONDITIONS = ("clean", "missing_t2f", "missing_t1c")
MISSING_CHANNEL = {"missing_t2f": 0, "missing_t1c": 2}


def patient_id(case_id: str) -> str:
    return case_id.rsplit("-", 1)[0]


def inner_case_ids(case_ids: list[str], *, fold: int, fraction: float = 0.1,
                   salt: str = "branch_p_v1_20260923") -> tuple[list[str], list[str]]:
    """Reserve patients, not studies, for gate fitting within outer train."""
    if not 0 < fraction < 1:
        raise ValueError("fraction must be strictly between zero and one")
    patients = sorted({patient_id(case) for case in case_ids})
    ranked = sorted(patients, key=lambda patient: hashlib.sha256(
        f"{salt}:{fold}:{patient}".encode()).digest())
    holdout = set(ranked[:max(1, round(len(ranked) * fraction))])
    expert = [case for case in case_ids if patient_id(case) not in holdout]
    gate = [case for case in case_ids if patient_id(case) in holdout]
    if not expert or not gate or set(expert) & set(gate):
        raise RuntimeError("Invalid Branch-P inner partition")
    return expert, gate


def apply_condition(image: torch.Tensor, condition: str) -> torch.Tensor:
    """Zero one normalized MRI channel after ordinary augmentation."""
    if condition == "clean":
        return image
    if condition in MISSING_CHANNEL:
        output = image.clone()
        output[:, MISSING_CHANNEL[condition]] = 0
        return output
    if condition == "mixed":
        choices = torch.randint(len(CONDITIONS), (image.shape[0],), device=image.device)
        output = image.clone()
        for ordinal, name in enumerate(CONDITIONS[1:], 1):
            selected = choices == ordinal
            if selected.any():
                output[selected, MISSING_CHANNEL[name]] = 0
        return output
    raise ValueError(f"Unknown Branch-P condition: {condition}")


class ConditionedDataset(Dataset):
    def __init__(self, source: Dataset, condition: str):
        if condition not in CONDITIONS:
            raise ValueError(f"Validation condition must be fixed: {condition}")
        self.source = source
        self.condition = condition

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int):
        image, target, case_id = self.source[index]
        return apply_condition(image.unsqueeze(0), self.condition)[0], target, case_id
