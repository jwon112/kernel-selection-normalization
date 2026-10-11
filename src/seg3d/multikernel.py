from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from typing import Any

import torch
from torch import nn


BRANCH_SPECS = OrderedDict(
    (
        ("k3", {"kernel_size": (3, 3, 3), "dilation": (1, 1, 1)}),
        ("k5", {"kernel_size": (5, 5, 5), "dilation": (1, 1, 1)}),
        ("dilated_k3", {"kernel_size": (3, 3, 3), "dilation": (2, 2, 2)}),
        ("anisotropic", {"kernel_size": (1, 3, 3), "dilation": (1, 1, 1)}),
    )
)
SELECTION_MODES = ("fixed", "uniform", "sk", "soft", "moe")


def _same_padding(kernel_size: tuple[int, int, int], dilation: tuple[int, int, int]):
    return tuple(((kernel - 1) * rate) // 2 for kernel, rate in zip(kernel_size, dilation))


class MultiKernelBranch3D(nn.Module):
    """One named dense spatial branch, without branch-local normalization."""

    def __init__(
        self,
        channels: int,
        kernel_size: tuple[int, int, int],
        dilation: tuple[int, int, int],
    ):
        super().__init__()
        self.spatial = nn.Conv3d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=_same_padding(kernel_size, dilation),
            dilation=dilation,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(x)


class MultiKernelSelector3D(nn.Module):
    """
    Shared block for fixed, uniform, SK, dense-soft, and sparse-MoE experiments.

    Selection is case-level. SK produces channel-wise dense attention.
    Dense-soft and sparse-MoE use scalar branch weights from the shared input.
    Sparse MoE executes only the selected branches for each sample.
    """

    def __init__(
        self,
        channels: int,
        mode: str,
        fixed_branch: str | None = None,
        top_k: int = 2,
        selector_hidden_channels: int | None = None,
        router_jitter: float = 0.01,
        selector_pooling: str = "avg",
    ):
        super().__init__()
        if mode not in SELECTION_MODES:
            raise ValueError(f"Unknown selection mode {mode!r}; expected one of {SELECTION_MODES}")
        if fixed_branch is not None and fixed_branch not in BRANCH_SPECS:
            raise ValueError(
                f"Unknown fixed branch {fixed_branch!r}; expected one of {tuple(BRANCH_SPECS)}"
            )
        if mode == "fixed" and fixed_branch is None:
            raise ValueError("fixed mode requires fixed_branch")
        if mode != "fixed" and fixed_branch is not None:
            raise ValueError("fixed_branch is only valid in fixed mode")

        self.channels = int(channels)
        self.mode = mode
        self.fixed_branch = fixed_branch
        self.router_jitter = float(router_jitter)
        if selector_pooling not in ("avg", "avgmax", "spatial2"):
            raise ValueError(
                f"Unknown selector_pooling {selector_pooling!r}; expected "
                "'avg', 'avgmax', or 'spatial2'"
            )
        self.selector_pooling = selector_pooling
        names = (fixed_branch,) if mode == "fixed" else tuple(BRANCH_SPECS)
        self.branch_names = tuple(name for name in names if name is not None)
        self.num_branches = len(self.branch_names)
        if mode == "moe" and not 1 <= int(top_k) <= self.num_branches:
            raise ValueError(f"top_k must be in [1, {self.num_branches}], got {top_k}")
        self.top_k = int(top_k)

        self.branches = nn.ModuleDict(
            {
                name: MultiKernelBranch3D(self.channels, **BRANCH_SPECS[name])
                for name in self.branch_names
            }
        )
        hidden = (
            max(8, self.channels // 4)
            if selector_hidden_channels is None
            else int(selector_hidden_channels)
        )
        if hidden < 1:
            raise ValueError("selector_hidden_channels must be positive")
        self.selector_hidden_channels = hidden
        selector_input_channels = self.channels * {
            "avg": 1,
            "avgmax": 2,
            "spatial2": 8,
        }[selector_pooling]

        if mode == "sk":
            self.sk_reduce = nn.Sequential(
                nn.Linear(selector_input_channels, hidden),
                nn.ReLU(inplace=True),
            )
            self.sk_heads = nn.ModuleDict(
                {name: nn.Linear(hidden, self.channels) for name in self.branch_names}
            )
            for head in self.sk_heads.values():
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
        elif mode in ("soft", "moe"):
            self.router = nn.Sequential(
                nn.Linear(selector_input_channels, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, self.num_branches),
            )
            nn.init.zeros_(self.router[-1].weight)
            nn.init.zeros_(self.router[-1].bias)

        self.post_norm = nn.InstanceNorm3d(self.channels)
        self.post_activation = nn.LeakyReLU(inplace=True)

        self._auxiliary_loss: torch.Tensor | None = None
        self._last_snapshot: dict[str, Any] | None = None
        self._intervention: dict[str, Any] | None = None

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        if self.selector_pooling == "spatial2":
            return torch.nn.functional.adaptive_avg_pool3d(x, (2, 2, 2)).flatten(1)
        average = x.mean(dim=(2, 3, 4))
        if self.selector_pooling == "avgmax":
            maximum = x.amax(dim=(2, 3, 4))
            return torch.cat((average, maximum), dim=1)
        return average

    @staticmethod
    def _rms_per_case(x: torch.Tensor) -> torch.Tensor:
        return x.detach().float().square().mean(dim=(1, 2, 3, 4)).sqrt()

    def auxiliary_loss(self) -> torch.Tensor:
        if self._auxiliary_loss is None:
            parameter = next(self.parameters())
            return parameter.sum() * 0.0
        return self._auxiliary_loss

    def selection_snapshot(self) -> dict[str, Any] | None:
        if self._last_snapshot is None:
            return None
        return {
            key: value.detach().clone() if isinstance(value, torch.Tensor) else value
            for key, value in self._last_snapshot.items()
        }

    def expert_gradient_snapshot(self) -> dict[str, bool]:
        """Report whether each branch participated in the current backward graph."""
        return {
            name: self.branches[name].spatial.weight.grad is not None
            for name in self.branch_names
        }

    def experiment_spec(self) -> dict[str, Any]:
        return {
            "module": type(self).__name__,
            "insertion": "last_decoder_second_spatial_conv",
            "mode": self.mode,
            "branch_names": list(self.branch_names),
            "fixed_branch": self.fixed_branch,
            "top_k": self.top_k if self.mode == "moe" else None,
            "selection_resolution": "case",
            "selector_hidden_channels": self.selector_hidden_channels,
            "selector_pooling": self.selector_pooling,
            "router_jitter": self.router_jitter,
            "branch_normalization": None,
            "post_fusion_normalization": type(self.post_norm).__name__,
            "post_fusion_activation": type(self.post_activation).__name__,
        }

    def set_intervention(
        self,
        kind: str,
        branch: str | None = None,
        renormalize: bool = True,
    ):
        if kind not in ("force_branch", "force_uniform", "force_hard", "drop_branch"):
            raise ValueError(
                "kind must be force_branch, force_uniform, force_hard, or drop_branch"
            )
        if kind in ("force_branch", "drop_branch"):
            if branch not in self.branch_names:
                raise ValueError(
                    f"branch must be one of {self.branch_names}, got {branch!r}"
                )
        elif branch is not None:
            raise ValueError(f"{kind} does not accept branch")
        self._intervention = {
            "kind": kind,
            "branch": branch,
            "renormalize": bool(renormalize),
        }

    def clear_intervention(self):
        self._intervention = None

    @contextmanager
    def intervention(
        self,
        kind: str,
        branch: str | None = None,
        renormalize: bool = True,
    ):
        previous = self._intervention
        self.set_intervention(kind, branch=branch, renormalize=renormalize)
        try:
            yield self
        finally:
            self._intervention = previous

    def _save_snapshot(
        self,
        weights: torch.Tensor,
        contributions: torch.Tensor,
        dense_probabilities: torch.Tensor | None = None,
        selected_mask: torch.Tensor | None = None,
    ):
        probabilities = weights if dense_probabilities is None else dense_probabilities
        entropy = -(probabilities.clamp_min(1e-8).log() * probabilities).sum(dim=1)
        snapshot: dict[str, Any] = {
            "mode": self.mode,
            "branch_names": self.branch_names,
            "weights": weights.detach(),
            "contribution_rms": contributions.detach(),
            "entropy": entropy.detach(),
        }
        if dense_probabilities is not None:
            snapshot["dense_probabilities"] = dense_probabilities.detach()
        if selected_mask is not None:
            snapshot["selected_mask"] = selected_mask.detach()
        if self._intervention is not None:
            snapshot["intervention"] = dict(self._intervention)
        self._last_snapshot = snapshot

    def _postprocess(self, x: torch.Tensor) -> torch.Tensor:
        return self.post_activation(self.post_norm(x))

    def _drop_dense_weights(self, weights: torch.Tensor) -> torch.Tensor:
        if self._intervention is None or self._intervention["kind"] != "drop_branch":
            return weights
        branch_index = self.branch_names.index(self._intervention["branch"])
        weights = weights.clone()
        weights[:, branch_index] = 0
        if self._intervention["renormalize"]:
            denominator = weights.sum(dim=1, keepdim=True)
            weights = torch.where(
                denominator > 0,
                weights / denominator.clamp_min(1e-8),
                weights,
            )
        return weights

    def _forward_forced_branch(self, x: torch.Tensor, branch: str) -> torch.Tensor:
        branch_index = self.branch_names.index(branch)
        output = self.branches[branch](x)
        weights = x.new_zeros((x.shape[0], self.num_branches))
        weights[:, branch_index] = 1
        contributions = weights.new_zeros(weights.shape)
        contributions[:, branch_index] = self._rms_per_case(output)
        selected = weights.bool()
        self._auxiliary_loss = output.sum() * 0.0
        self._save_snapshot(weights, contributions, selected_mask=selected)
        return self._postprocess(output)

    def _forward_forced_uniform(self, x: torch.Tensor) -> torch.Tensor:
        weights = x.new_full(
            (x.shape[0], self.num_branches),
            1.0 / self.num_branches,
        )
        output = self._forward_dense_scalar(x, weights)
        self._auxiliary_loss = output.sum() * 0.0
        return output

    def _forward_fixed(self, x: torch.Tensor) -> torch.Tensor:
        if self._intervention is not None and self._intervention["kind"] == "drop_branch":
            output = torch.zeros_like(x)
            weights = x.new_zeros((x.shape[0], 1))
            contributions = weights.clone()
            self._auxiliary_loss = output.sum() * 0.0
            self._save_snapshot(
                weights,
                contributions,
                selected_mask=weights.bool(),
            )
            return self._postprocess(output)
        output = self.branches[self.branch_names[0]](x)
        weights = x.new_ones((x.shape[0], 1))
        contributions = self._rms_per_case(output).unsqueeze(1)
        self._auxiliary_loss = output.sum() * 0.0
        self._save_snapshot(weights, contributions)
        return self._postprocess(output)

    def _forward_dense_scalar(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        dense_probabilities: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = torch.zeros_like(x)
        contribution_columns = []
        for branch_index, name in enumerate(self.branch_names):
            branch_output = self.branches[name](x)
            weighted = branch_output * weights[:, branch_index, None, None, None, None]
            output = output + weighted
            contribution_columns.append(self._rms_per_case(weighted))
        contributions = torch.stack(contribution_columns, dim=1)
        self._save_snapshot(weights, contributions, dense_probabilities=dense_probabilities)
        return self._postprocess(output)

    def _forward_uniform(self, x: torch.Tensor) -> torch.Tensor:
        weights = x.new_full(
            (x.shape[0], self.num_branches),
            1.0 / self.num_branches,
        )
        weights = self._drop_dense_weights(weights)
        output = self._forward_dense_scalar(x, weights)
        self._auxiliary_loss = output.sum() * 0.0
        return output

    def _forward_sk(self, x: torch.Tensor) -> torch.Tensor:
        branch_outputs = [self.branches[name](x) for name in self.branch_names]
        fused = branch_outputs[0]
        for branch_output in branch_outputs[1:]:
            fused = fused + branch_output
        fused = fused / self.num_branches
        hidden = self.sk_reduce(self._pool(fused))
        logits = torch.stack(
            [self.sk_heads[name](hidden) for name in self.branch_names],
            dim=1,
        )
        channel_weights = torch.softmax(logits, dim=1)
        channel_weights = self._drop_dense_weights(channel_weights)
        output = torch.zeros_like(x)
        contribution_columns = []
        for branch_index, branch_output in enumerate(branch_outputs):
            weighted = branch_output * channel_weights[:, branch_index, :, None, None, None]
            output = output + weighted
            contribution_columns.append(self._rms_per_case(weighted))
        summary_weights = channel_weights.mean(dim=2)
        contributions = torch.stack(contribution_columns, dim=1)
        self._auxiliary_loss = output.sum() * 0.0
        self._save_snapshot(summary_weights, contributions)
        if self._last_snapshot is not None:
            self._last_snapshot["channel_weights"] = channel_weights.detach()
        return self._postprocess(output)

    def _router_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.router(self._pool(x))
        if self.training and self.router_jitter > 0:
            logits = logits + torch.randn_like(logits) * self.router_jitter
        return torch.softmax(logits, dim=1)

    def _forward_soft(self, x: torch.Tensor) -> torch.Tensor:
        probabilities = self._router_probabilities(x)
        if self._intervention is not None and self._intervention["kind"] == "force_hard":
            indices = probabilities.argmax(dim=1, keepdim=True)
            weights = torch.zeros_like(probabilities).scatter_(1, indices, 1.0)
        else:
            weights = self._drop_dense_weights(probabilities)
        output = self._forward_dense_scalar(
            x,
            weights,
            dense_probabilities=probabilities,
        )
        self._auxiliary_loss = output.sum() * 0.0
        return output

    def _forward_moe(self, x: torch.Tensor) -> torch.Tensor:
        probabilities = self._router_probabilities(x)
        drop_intervention = (
            self._intervention is not None
            and self._intervention["kind"] == "drop_branch"
        )
        reroute = drop_intervention and self._intervention["renormalize"]
        routing_probabilities = probabilities
        if reroute:
            branch_index = self.branch_names.index(self._intervention["branch"])
            routing_probabilities = probabilities.clone()
            routing_probabilities[:, branch_index] = 0
            routing_probabilities = routing_probabilities / routing_probabilities.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-8)
        top_probabilities, top_indices = torch.topk(
            routing_probabilities,
            k=self.top_k,
            dim=1,
        )
        top_weights = top_probabilities / top_probabilities.sum(dim=1, keepdim=True)
        sparse_weights = probabilities.new_zeros(probabilities.shape)
        sparse_weights = sparse_weights.scatter(1, top_indices, top_weights)
        if drop_intervention and not reroute:
            branch_index = self.branch_names.index(self._intervention["branch"])
            sparse_weights = sparse_weights.clone()
            sparse_weights[:, branch_index] = 0
        selected_mask = sparse_weights > 0

        output = torch.zeros_like(x)
        contributions = probabilities.new_zeros(probabilities.shape)
        for branch_index, name in enumerate(self.branch_names):
            sample_indices = selected_mask[:, branch_index].nonzero(as_tuple=False).flatten()
            if sample_indices.numel() == 0:
                continue
            branch_output = self.branches[name](x.index_select(0, sample_indices))
            branch_weights = sparse_weights.index_select(0, sample_indices)[:, branch_index]
            weighted = branch_output * branch_weights.to(branch_output.dtype)[
                :, None, None, None, None
            ]
            output = output.index_add(0, sample_indices, weighted)
            with torch.no_grad():
                contributions[sample_indices, branch_index] = self._rms_per_case(weighted)

        importance = probabilities.mean(dim=0)
        load = (selected_mask.float().mean(dim=0) / self.top_k).detach()
        self._auxiliary_loss = self.num_branches * torch.sum(importance * load)
        self._save_snapshot(
            sparse_weights,
            contributions,
            dense_probabilities=probabilities,
            selected_mask=selected_mask,
        )
        return self._postprocess(output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._intervention is not None:
            kind = self._intervention["kind"]
            if kind == "force_branch":
                return self._forward_forced_branch(x, self._intervention["branch"])
            if kind == "force_uniform":
                return self._forward_forced_uniform(x)
        if self.mode == "fixed":
            return self._forward_fixed(x)
        if self.mode == "uniform":
            return self._forward_uniform(x)
        if self.mode == "sk":
            return self._forward_sk(x)
        if self.mode == "soft":
            return self._forward_soft(x)
        return self._forward_moe(x)
