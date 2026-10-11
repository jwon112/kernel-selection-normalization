from collections.abc import Callable

import torch
from torch import nn
import torch.nn.functional as F

from seg3d.models import (
    ResidualUNet3D,
    CanonicalUNet3D,
    BandLimitedCanonicalUNet3D,
    CanonicalUNet3DMultiKernel,
    CanonicalUNet3DBottleneckMultiKernel,
    CanonicalEncoderClassifier,
    CanonicalEncoderMultiKernelClassifier,
    CanonicalEncoderRFExpertClassifier,
    CanonicalEncoderStageExpertClassifier,
)


class PadToMultiple(nn.Module):
    """Right-pad arbitrary full volumes and crop logits back to the input shape."""

    def __init__(self, model: nn.Module, multiple: int):
        super().__init__()
        self.model = model
        self.multiple = multiple

    def forward(self, x):
        shape = x.shape[-3:]
        needed = [(self.multiple - value % self.multiple) % self.multiple
                  for value in shape]
        if any(needed):
            x = F.pad(x, (0, needed[2], 0, needed[1], 0, needed[0]))
        logits = self.model(x)
        return logits[..., :shape[0], :shape[1], :shape[2]]


def _nnunet_plain(in_channels, out_channels, **_):
    from dynamic_network_architectures.architectures.unet import PlainConvUNet

    model = PlainConvUNet(
        input_channels=in_channels,
        n_stages=6,
        features_per_stage=(32, 64, 128, 256, 320, 320),
        conv_op=nn.Conv3d,
        kernel_sizes=((3, 3, 3),) * 6,
        strides=((1, 1, 1), (2, 2, 2), (2, 2, 2),
                 (2, 2, 2), (2, 2, 2), (2, 2, 2)),
        n_conv_per_stage=(2, 2, 2, 2, 2, 2),
        num_classes=out_channels,
        n_conv_per_stage_decoder=(2, 2, 2, 2, 2),
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=False,
    )
    return PadToMultiple(model, 32)


def _canonical_unet(in_channels, out_channels, **_):
    return CanonicalUNet3D(in_channels, out_channels)


def _band_limited_canonical_unet(
    in_channels, out_channels, band="low", fine_kernel=3, low_kernel=9, **_
):
    return BandLimitedCanonicalUNet3D(
        in_channels=in_channels, out_channels=out_channels, band=band,
        fine_kernel=fine_kernel, low_kernel=low_kernel,
    )


def _canonical_multikernel(
    in_channels,
    out_channels,
    selection_mode,
    fixed_branch=None,
    selector_fixed_branches=None,
    top_k=2,
    selector_hidden_channels=None,
    router_jitter=0.0,
    selector_pooling="avg",
    selector_decoder_indices=(3,),
    **_,
):
    return CanonicalUNet3DMultiKernel(
        in_channels=in_channels,
        out_channels=out_channels,
        selection_mode=selection_mode,
        fixed_branch=fixed_branch,
        selector_fixed_branches=selector_fixed_branches,
        top_k=top_k,
        selector_hidden_channels=selector_hidden_channels,
        router_jitter=router_jitter,
        selector_pooling=selector_pooling,
        selector_decoder_indices=selector_decoder_indices,
    )


def _mk_fixed(branch):
    return lambda in_channels, out_channels, **kwargs: _canonical_multikernel(
        in_channels,
        out_channels,
        selection_mode="fixed",
        fixed_branch=branch,
        **kwargs,
    )


def _encoder_multikernel_classifier(
    in_channels,
    out_channels,
    selection_mode,
    fixed_branch=None,
    top_k=2,
    selector_hidden_channels=None,
    router_jitter=0.0,
    selector_pooling="avg",
    backbone_out_channels=3,
    **_,
):
    return CanonicalEncoderMultiKernelClassifier(
        in_channels=in_channels,
        num_classes=out_channels,
        selection_mode=selection_mode,
        fixed_branch=fixed_branch,
        top_k=top_k,
        selector_hidden_channels=selector_hidden_channels,
        router_jitter=router_jitter,
        selector_pooling=selector_pooling,
        backbone_out_channels=backbone_out_channels,
    )

def _bottleneck_multikernel(
    in_channels,
    out_channels,
    selection_mode,
    fixed_branch=None,
    top_k=2,
    selector_hidden_channels=None,
    router_jitter=0.0,
    selector_pooling="spatial2",
    **_,
):
    return CanonicalUNet3DBottleneckMultiKernel(
        in_channels=in_channels,
        out_channels=out_channels,
        selection_mode=selection_mode,
        fixed_branch=fixed_branch,
        top_k=top_k,
        selector_hidden_channels=selector_hidden_channels,
        router_jitter=router_jitter,
        selector_pooling=selector_pooling,
    )


def _canonical_encoder_classifier(
    in_channels, out_channels, backbone_out_channels=3, **_
):
    return CanonicalEncoderClassifier(
        in_channels=in_channels,
        num_classes=out_channels,
        backbone_out_channels=backbone_out_channels,
    )


def _canonical_encoder_rf_expert_classifier(
    in_channels,
    out_channels,
    tap_stage=2,
    expert_dilation=1,
    backbone_out_channels=3,
    **_,
):
    return CanonicalEncoderRFExpertClassifier(
        in_channels=in_channels,
        num_classes=out_channels,
        tap_stage=tap_stage,
        expert_dilation=expert_dilation,
        backbone_out_channels=backbone_out_channels,
    )


def _canonical_encoder_stage_expert_classifier(
    in_channels, out_channels, tap_stage=1, common_channels=64,
    backbone_out_channels=3, **_,
):
    return CanonicalEncoderStageExpertClassifier(
        in_channels=in_channels, num_classes=out_channels,
        tap_stage=tap_stage, common_channels=common_channels,
        backbone_out_channels=backbone_out_channels,
    )


def _resunet(in_channels, out_channels, base_channels=16, **_):
    model = ResidualUNet3D(
        in_channels=in_channels, out_channels=out_channels,
        base_channels=base_channels, gradient_checkpointing=True,
    )
    return PadToMultiple(model, 16)


def _vsnext(in_channels, out_channels, **_):
    from .vendor.convnextv2_unet3d import ConvNeXtV2UNet3DS2PreconvConcat

    model = ConvNeXtV2UNet3DS2PreconvConcat(
        n_channels=in_channels,
        n_classes=out_channels,
        depths=(2, 2, 6, 1),
        dims=(64, 128, 256, 320),
        head_channels=32,
        decoder_depths=(1, 1, 1),
        drop_path_rate=0.1,
        deep_supervision=False,
    )
    return PadToMultiple(model, 16)


def _mednext(in_channels, out_channels, **_):
    from .vendor.mednextv1 import create_mednextv1_B

    model = create_mednextv1_B(
        num_input_channels=in_channels, num_classes=out_channels,
        kernel_size=3, ds=False, grn=False,
    )
    return PadToMultiple(model, 16)


def _unetr(in_channels, out_channels, img_size=(64, 112, 112), **_):
    from monai.networks.nets import UNETR
    return UNETR(
        in_channels=in_channels, out_channels=out_channels, img_size=img_size,
        feature_size=16, hidden_size=768, mlp_dim=3072, num_heads=12,
        proj_type='conv', norm_name='instance', conv_block=True, res_block=True,
        dropout_rate=0.0, spatial_dims=3,
    )


def _swinunetr(in_channels, out_channels, **_):
    """SwinUNETR with internal 32-multiple padding for 64x112x112 EchoNet clips."""
    from monai.networks.nets import SwinUNETR
    model = SwinUNETR(
        in_channels=in_channels, out_channels=out_channels,
        feature_size=24, spatial_dims=3,
    )
    # MONAI SwinUNETR requires every spatial axis to be divisible by 32.
    # PadToMultiple restores the original output coordinates before loss/evaluation.
    return PadToMultiple(model, 32)


def _mamba2(in_channels, out_channels, base_channels=16, **_):
    from .vendor.mamba_unet3d import MambaUNet3D

    model = MambaUNet3D(
        n_channels=in_channels, n_classes=out_channels,
        base_channels=base_channels,
    )
    return PadToMultiple(model, 8)


_MODELS: dict[str, Callable[..., nn.Module]] = {
    "nnunet_plain": _nnunet_plain,
    "canonical_unet3d": _canonical_unet,
    "canonical_unet3d_fixed_band": _band_limited_canonical_unet,
    "canonical_unet3d_mk_fixed_k3": _mk_fixed("k3"),
    "canonical_unet3d_mk_fixed_k5": _mk_fixed("k5"),
    "canonical_unet3d_mk_fixed_dilated_k3": _mk_fixed("dilated_k3"),
    "canonical_unet3d_mk_fixed_anisotropic": _mk_fixed("anisotropic"),
    "canonical_unet3d_mk_fixed_schedule": lambda in_channels, out_channels, **kwargs: _canonical_multikernel(
        in_channels, out_channels, selection_mode="fixed", **kwargs
    ),
    "canonical_unet3d_mk_uniform": lambda in_channels, out_channels, **kwargs: _canonical_multikernel(
        in_channels, out_channels, selection_mode="uniform", **kwargs
    ),
    "canonical_unet3d_mk_sk": lambda in_channels, out_channels, **kwargs: _canonical_multikernel(
        in_channels, out_channels, selection_mode="sk", **kwargs
    ),
    "canonical_unet3d_mk_soft_control": lambda in_channels, out_channels, **kwargs: _canonical_multikernel(
        in_channels, out_channels, selection_mode="soft", **kwargs
    ),
    "canonical_unet3d_mk_moe_top2": lambda in_channels, out_channels, **kwargs: _canonical_multikernel(
        in_channels, out_channels, selection_mode="moe", top_k=2, **kwargs
    ),
    "canonical_unet3d_mk_moe_top1": lambda in_channels, out_channels, **kwargs: _canonical_multikernel(
        in_channels, out_channels, selection_mode="moe", top_k=1, **kwargs
    ),
    "canonical_unet3d_bottleneck_mk_fixed_k3": lambda in_channels, out_channels, **kwargs: _bottleneck_multikernel(in_channels, out_channels, selection_mode="fixed", fixed_branch="k3", **kwargs),
    "canonical_unet3d_bottleneck_mk_fixed_k5": lambda in_channels, out_channels, **kwargs: _bottleneck_multikernel(in_channels, out_channels, selection_mode="fixed", fixed_branch="k5", **kwargs),
    "canonical_unet3d_bottleneck_mk_fixed_dilated_k3": lambda in_channels, out_channels, **kwargs: _bottleneck_multikernel(in_channels, out_channels, selection_mode="fixed", fixed_branch="dilated_k3", **kwargs),
    "canonical_unet3d_bottleneck_mk_fixed_anisotropic": lambda in_channels, out_channels, **kwargs: _bottleneck_multikernel(in_channels, out_channels, selection_mode="fixed", fixed_branch="anisotropic", **kwargs),
    "canonical_unet3d_bottleneck_mk_uniform": lambda in_channels, out_channels, **kwargs: _bottleneck_multikernel(in_channels, out_channels, selection_mode="uniform", **kwargs),
    "canonical_unet3d_bottleneck_mk_soft": lambda in_channels, out_channels, **kwargs: _bottleneck_multikernel(in_channels, out_channels, selection_mode="soft", **kwargs),
    "canonical_encoder_classifier": _canonical_encoder_classifier,
    "canonical_encoder_rf_expert_classifier": _canonical_encoder_rf_expert_classifier,
    "canonical_encoder_stage_expert_classifier": _canonical_encoder_stage_expert_classifier,
    "canonical_encoder_bottleneck_mk_fixed_k3_classifier": lambda in_channels, out_channels, **kwargs: _encoder_multikernel_classifier(in_channels, out_channels, selection_mode="fixed", fixed_branch="k3", **kwargs),
    "canonical_encoder_bottleneck_mk_fixed_k5_classifier": lambda in_channels, out_channels, **kwargs: _encoder_multikernel_classifier(in_channels, out_channels, selection_mode="fixed", fixed_branch="k5", **kwargs),
    "canonical_encoder_bottleneck_mk_fixed_dilated_k3_classifier": lambda in_channels, out_channels, **kwargs: _encoder_multikernel_classifier(in_channels, out_channels, selection_mode="fixed", fixed_branch="dilated_k3", **kwargs),
    "canonical_encoder_bottleneck_mk_fixed_anisotropic_classifier": lambda in_channels, out_channels, **kwargs: _encoder_multikernel_classifier(in_channels, out_channels, selection_mode="fixed", fixed_branch="anisotropic", **kwargs),
    "canonical_encoder_bottleneck_mk_uniform_classifier": lambda in_channels, out_channels, **kwargs: _encoder_multikernel_classifier(in_channels, out_channels, selection_mode="uniform", **kwargs),
    "canonical_encoder_bottleneck_mk_soft_classifier": lambda in_channels, out_channels, **kwargs: _encoder_multikernel_classifier(in_channels, out_channels, selection_mode="soft", **kwargs),
    "canonical_encoder_mk_uniform_classifier": lambda in_channels, out_channels, **kwargs: _encoder_multikernel_classifier(
        in_channels, out_channels, selection_mode="uniform", **kwargs
    ),
    "canonical_encoder_mk_sk_classifier": lambda in_channels, out_channels, **kwargs: _encoder_multikernel_classifier(
        in_channels, out_channels, selection_mode="sk", **kwargs
    ),
    "canonical_encoder_mk_moe_top2_classifier": lambda in_channels, out_channels, **kwargs: _encoder_multikernel_classifier(
        in_channels, out_channels, selection_mode="moe", top_k=2, **kwargs
    ),
    "resunet": _resunet,
    "vsnext_preconv3x3": _vsnext,
    "mednext_b_k3": _mednext,
    "unetr_vit_base": _unetr,
    "swinunetr": _swinunetr,
    "mamba2_unet": _mamba2,
}


def available_models():
    return tuple(_MODELS)


def build_model(name, in_channels=4, out_channels=3, **kwargs):
    try:
        factory = _MODELS[name]
    except KeyError as error:
        raise ValueError(
            f"Unknown model {name!r}. Available: {', '.join(available_models())}"
        ) from error
    return factory(in_channels=in_channels, out_channels=out_channels, **kwargs)
