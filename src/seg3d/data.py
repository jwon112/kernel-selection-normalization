import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class BraTSVolumeDataset(Dataset):
    def __init__(self, data_dir: str | Path, fold: int, split: str):
        self.data_dir = Path(data_dir)
        with open(self.data_dir / "splits.json") as file:
            folds = json.load(file)
        metadata_path = self.data_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        self.target_encoding = metadata.get("target_encoding", "stored_channels")
        self.case_ids = folds[fold][split]

    def __len__(self): return len(self.case_ids)

    def __getitem__(self, index):
        case_id = self.case_ids[index]
        image = np.load(self.data_dir / "images" / f"{case_id}.npy")
        target = np.load(self.data_dir / "labels" / f"{case_id}.npy")
        if self.target_encoding == "brats_regions_from_atomic_123":
            labels = set(np.unique(target).tolist())
            if not labels.issubset({0, 1, 2, 3}):
                raise ValueError(f"Unexpected atomic BraTS labels for {case_id}: {sorted(labels)}")
            target = np.stack((target > 0, (target == 1) | (target == 3), target == 3)).astype(np.float32)
        elif self.target_encoding != "stored_channels":
            raise ValueError(f"Unsupported target_encoding: {self.target_encoding}")
        return torch.from_numpy(image), torch.from_numpy(target), case_id


def _random_affine_3d(image, target, probability=0.2, max_degrees=30.0, scale_range=(0.7, 1.4)):
    """Per-case 3D rotation and isotropic scale, with nearest-neighbor labels."""
    batch = image.shape[0]; device = image.device
    apply = torch.rand(batch, device=device) < probability
    if not apply.any(): return image, target
    angles = torch.empty(batch, 3, device=device).uniform_(-max_degrees, max_degrees) * (torch.pi / 180.0)
    angles[~apply] = 0
    scale = torch.empty(batch, device=device).uniform_(*scale_range); scale[~apply] = 1
    cx, cy, cz = angles[:, 0].cos(), angles[:, 1].cos(), angles[:, 2].cos()
    sx, sy, sz = angles[:, 0].sin(), angles[:, 1].sin(), angles[:, 2].sin()
    rx = torch.zeros(batch, 3, 3, device=device); rx[:, 0, 0] = 1; rx[:, 1, 1] = cx; rx[:, 1, 2] = -sx; rx[:, 2, 1] = sx; rx[:, 2, 2] = cx
    ry = torch.zeros(batch, 3, 3, device=device); ry[:, 1, 1] = 1; ry[:, 0, 0] = cy; ry[:, 0, 2] = sy; ry[:, 2, 0] = -sy; ry[:, 2, 2] = cy
    rz = torch.zeros(batch, 3, 3, device=device); rz[:, 2, 2] = 1; rz[:, 0, 0] = cz; rz[:, 0, 1] = -sz; rz[:, 1, 0] = sz; rz[:, 1, 1] = cz
    theta = torch.zeros(batch, 3, 4, device=device); theta[:, :, :3] = (rz @ ry @ rx) / scale[:, None, None]
    grid = F.affine_grid(theta, image.shape, align_corners=False)
    original_dtype = image.dtype
    image = F.grid_sample(image.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False).to(original_dtype)
    target = F.grid_sample(target.float(), grid, mode="nearest", padding_mode="zeros", align_corners=False)
    return image, target


def _per_case_select(original, transformed, apply):
    return torch.where(apply[:, None, None, None, None], transformed, original)


def gpu_augment(image, target, profile="light"):
    """GPU augmentation; nnunet_gpu is strong full-volume nnU-Net-inspired augmentation."""
    strong = profile == "nnunet_gpu"
    if strong: image, target = _random_affine_3d(image, target)
    for dimension in (2, 3, 4):
        selected = torch.rand(image.shape[0], device=image.device) < 0.5
        if selected.any():
            image[selected] = torch.flip(image[selected], (dimension,)); target[selected] = torch.flip(target[selected], (dimension,))
    scale = torch.empty((image.shape[0], image.shape[1], 1, 1, 1), device=image.device).uniform_(0.75, 1.25)
    image = image * scale + torch.empty_like(scale).uniform_(-0.1, 0.1)
    if torch.rand((), device=image.device) < 0.15: image = image + torch.randn_like(image) * 0.01
    if strong:
        contrast_mask = torch.rand(image.shape[0], device=image.device) < 0.15
        contrast = torch.empty((image.shape[0], image.shape[1], 1, 1, 1), device=image.device).uniform_(0.75, 1.25)
        mean = image.mean((2, 3, 4), keepdim=True); image = _per_case_select(image, (image - mean) * contrast + mean, contrast_mask)
        gamma_mask = torch.rand(image.shape[0], device=image.device) < 0.15
        gamma = torch.empty((image.shape[0], image.shape[1], 1, 1, 1), device=image.device).uniform_(0.7, 1.5)
        low, high = image.amin((2, 3, 4), keepdim=True), image.amax((2, 3, 4), keepdim=True)
        normalized = (image - low) / (high - low).clamp_min(1e-6)
        image = _per_case_select(image, normalized.pow(gamma) * (high - low) + low, gamma_mask)
        if torch.rand((), device=image.device) < 0.2: image = F.avg_pool3d(image, kernel_size=3, stride=1, padding=1)
        if torch.rand((), device=image.device) < 0.25:
            factor = float(torch.empty((), device=image.device).uniform_(0.5, 1.0))
            small = F.interpolate(image.float(), scale_factor=factor, mode="trilinear", align_corners=False)
            image = F.interpolate(small, size=image.shape[2:], mode="trilinear", align_corners=False).to(image.dtype)
    return image, target


def gpu_nnunet_augment(image, target): return gpu_augment(image, target, profile="nnunet_gpu")

def _random_inplane_affine_ct(
    image,
    target=None,
    probability=0.2,
    max_degrees=15.0,
    scale_range=(0.9, 1.1),
    image_fill=-1.0,
):
    """Apply an XY-plane affine while preserving the anisotropic CT z axis."""
    batch = image.shape[0]
    device = image.device
    apply = torch.rand(batch, device=device) < probability
    if not apply.any():
        return image, target
    angle = torch.empty(batch, device=device).uniform_(
        -max_degrees, max_degrees
    ) * (torch.pi / 180.0)
    angle[~apply] = 0.0
    scale = torch.empty(batch, device=device).uniform_(*scale_range)
    scale[~apply] = 1.0
    cosine = angle.cos() / scale
    sine = angle.sin() / scale
    theta = torch.zeros(batch, 3, 4, device=device)
    theta[:, 0, 0] = cosine
    theta[:, 0, 1] = -sine
    theta[:, 1, 0] = sine
    theta[:, 1, 1] = cosine
    theta[:, 2, 2] = 1.0
    grid = F.affine_grid(theta, image.shape, align_corners=False)
    original_dtype = image.dtype
    shifted = image.float() - float(image_fill)
    image = F.grid_sample(
        shifted, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    ).add_(float(image_fill)).to(original_dtype)
    if target is not None:
        target = F.grid_sample(
            target.float(), grid, mode="nearest", padding_mode="zeros",
            align_corners=False,
        )
    return image, target


def gpu_augment_nsclc_ct(image, target=None, profile="ct_strong"):
    """Target-independent GPU augmentation for anisotropic thoracic CT.

    Random draws depend only on the image batch, so paired tasks receive the
    same transforms when their seeds, case order, and micro-batch sizes match.
    """
    if profile == "none":
        return image, target
    if profile != "ct_strong":
        raise ValueError(f"Unknown NSCLC CT augmentation profile: {profile!r}")

    image, target = _random_inplane_affine_ct(image, target)
    batch = image.shape[0]
    device = image.device

    # Only left-right reflection is anatomically label preserving here.
    selected = torch.rand(batch, device=device) < 0.5
    if selected.any():
        image[selected] = torch.flip(image[selected], (4,))
        if target is not None:
            target[selected] = torch.flip(target[selected], (4,))

    intensity_scale = torch.empty(
        (batch, image.shape[1], 1, 1, 1), device=device
    ).uniform_(0.85, 1.15)
    intensity_shift = torch.empty_like(intensity_scale).uniform_(-0.08, 0.08)
    image = image * intensity_scale + intensity_shift

    noise_mask = torch.rand(batch, device=device) < 0.15
    if noise_mask.any():
        image = _per_case_select(
            image, image + torch.randn_like(image) * 0.015, noise_mask
        )

    contrast_mask = torch.rand(batch, device=device) < 0.15
    contrast = torch.empty(
        (batch, image.shape[1], 1, 1, 1), device=device
    ).uniform_(0.8, 1.2)
    mean = image.mean((2, 3, 4), keepdim=True)
    image = _per_case_select(
        image, (image - mean) * contrast + mean, contrast_mask
    )

    gamma_mask = torch.rand(batch, device=device) < 0.15
    gamma = torch.empty(
        (batch, image.shape[1], 1, 1, 1), device=device
    ).uniform_(0.8, 1.25)
    low = image.amin((2, 3, 4), keepdim=True)
    high = image.amax((2, 3, 4), keepdim=True)
    normalized = (image - low) / (high - low).clamp_min(1e-6)
    image = _per_case_select(
        image, normalized.pow(gamma) * (high - low) + low, gamma_mask
    )

    blur_mask = torch.rand(batch, device=device) < 0.15
    if blur_mask.any():
        padded = F.pad(image, (1, 1, 1, 1, 0, 0), value=-1.0)
        blurred = F.avg_pool3d(padded, kernel_size=(1, 3, 3), stride=1)
        image = _per_case_select(image, blurred, blur_mask)

    lowres_mask = torch.rand(batch, device=device) < 0.20
    if lowres_mask.any():
        factor = float(torch.empty((), device=device).uniform_(0.65, 1.0))
        depth, height, width = image.shape[2:]
        small = F.interpolate(
            image.float(),
            size=(
                depth,
                max(1, round(height * factor)),
                max(1, round(width * factor)),
            ),
            mode="trilinear",
            align_corners=False,
        )
        restored = F.interpolate(
            small, size=(depth, height, width), mode="trilinear",
            align_corners=False,
        ).to(image.dtype)
        image = _per_case_select(image, restored, lowres_mask)

    return image.clamp_(-1.0, 1.0), target
