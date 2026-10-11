import torch
import torch.nn.functional as F


def dice_bce_loss(logits, target, epsilon=1e-5, region_weights=None):
    """Dice+BCE loss, optionally with normalized per-region weights.

    ``region_weights`` changes only the relative WT/TC/ET emphasis. The caller
    must provide one positive weight per output channel; its scale is normalized
    here so role-specific losses retain the same overall magnitude.
    """
    probabilities = logits.sigmoid().float()
    target = target.float()
    axes = (0, 2, 3, 4)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none").mean(axes)
    intersection = (probabilities * target).sum(axes)
    denominator = probabilities.sum(axes) + target.sum(axes)
    dice_error = 1 - (2 * intersection + epsilon) / (denominator + epsilon)
    per_region = bce + dice_error
    if region_weights is None:
        return per_region.mean()
    weights = torch.as_tensor(region_weights, device=logits.device, dtype=per_region.dtype)
    if weights.ndim != 1 or weights.numel() != logits.shape[1] or torch.any(weights <= 0):
        raise ValueError("region_weights must contain one positive value per output channel")
    weights = weights / weights.sum()
    return (per_region * weights).sum()


@torch.no_grad()
def region_dice(logits, target):
    prediction = logits.sigmoid() >= 0.5
    target = target.bool()
    axes = (0, 2, 3, 4)
    intersection = (prediction & target).sum(axes)
    denominator = prediction.sum(axes) + target.sum(axes)
    return 2 * intersection / denominator.clamp_min(1)
