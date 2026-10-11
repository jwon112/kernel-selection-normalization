import torch


def route_weighted_sparse_macs(
    profiled_active_macs,
    profiled_selected_branches,
    branch_macs,
    execution_rates,
):
    """Decompose a routed profile and reweight branch work by observed routes."""
    common_and_router_macs = float(profiled_active_macs) - sum(
        branch_macs[name] for name in profiled_selected_branches
    )
    if common_and_router_macs < 0:
        raise RuntimeError("Sparse MAC decomposition produced negative common work")
    total_macs = common_and_router_macs + sum(branch_macs.values())
    active_macs = common_and_router_macs + sum(
        execution_rates[name] * branch_macs[name] for name in branch_macs
    )
    return total_macs, active_macs


def _execution_rates(selector_records, branch_names):
    rates = {}
    for name in branch_names:
        rows = [row for row in selector_records if row["branch"] == name]
        if not rows:
            raise ValueError(f"No validation routing records for branch {name!r}")
        rates[name] = sum(float(row["executed"]) for row in rows) / len(rows)
    return rates


def _conv3d_macs(conv, spatial_shape):
    output_voxels = 1
    kernel_voxels = 1
    for size in spatial_shape:
        output_voxels *= int(size)
    for size in conv.kernel_size:
        kernel_voxels *= int(size)
    return (
        output_voxels
        * conv.out_channels
        * (conv.in_channels // conv.groups)
        * kernel_voxels
    )


@torch.no_grad()
def profile_model_complexity(model, input_shape, device, selector_records=None):
    """Profile one full volume; sparse active work uses held-out route frequencies."""
    from thop import profile

    was_training = model.training
    model.eval()
    sample = torch.zeros(
        (1, 4, *input_shape), device=device, dtype=torch.float32
    )
    profiled_active_macs, _ = profile(model, inputs=(sample,), verbose=False)
    active_macs = total_macs = profiled_active_macs
    experiment_spec = getattr(model, "experiment_spec", None)
    spec = experiment_spec() if callable(experiment_spec) else None
    if spec is not None and spec.get("mode") == "moe":
        if not selector_records:
            raise ValueError(
                "Sparse MoE complexity requires held-out selector_records; "
                "a synthetic route is not a valid active-MAC estimate"
            )
        snapshot = model.selection_snapshot()
        if snapshot is None or "selected_mask" not in snapshot:
            raise RuntimeError("Sparse profiling did not produce a selected-mask snapshot")
        branch_names = tuple(spec["branch_names"])
        selected = snapshot["selected_mask"][0].cpu().tolist()
        profiled_selected = [
            name for name, is_selected in zip(branch_names, selected) if is_selected
        ]
        selector = model.multi_kernel_selector
        branch_macs = {
            name: _conv3d_macs(selector.branches[name].spatial, input_shape)
            for name in branch_names
        }
        execution_rates = _execution_rates(selector_records, branch_names)
        expected_executions = sum(execution_rates.values())
        if abs(expected_executions - spec["top_k"]) > 1e-5:
            raise ValueError(
                "Held-out execution rates sum to "
                f"{expected_executions}, expected top_k={spec['top_k']}"
            )
        total_macs, active_macs = route_weighted_sparse_macs(
            profiled_active_macs,
            profiled_selected,
            branch_macs,
            execution_rates,
        )
    del sample
    if was_training:
        model.train()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    return {
        "parameters": int(parameters),
        "trainable_parameters": int(trainable),
        "macs_per_volume": int(round(total_macs)),
        "flops_per_volume": int(round(2 * total_macs)),
        "active_macs_per_volume": int(round(active_macs)),
        "active_flops_per_volume": int(round(2 * active_macs)),
        "flops_input_shape": "x".join(map(str, input_shape)),
        "flops_method": (
            "THOP; FLOPs=2xMACs; batch=1; sparse branch MACs decomposed "
            "analytically; active branch work weighted by held-out routes"
            if spec is not None and spec.get("mode") == "moe"
            else "THOP; FLOPs=2xMACs; batch=1; active=total"
        ),
    }
