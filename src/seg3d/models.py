import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from seg3d.multikernel import MultiKernelSelector3D


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        groups = min(8, out_channels)
        self.body = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, 1, bias=False)
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.activation(self.body(x) + self.skip(x))


class ResidualUNet3D(nn.Module):
    """Five-level residual 3D U-Net with GroupNorm."""

    def __init__(self, in_channels=4, out_channels=3, base_channels=16,
                 gradient_checkpointing=True):
        super().__init__()
        widths = [base_channels * 2**i for i in range(5)]
        self.gradient_checkpointing = gradient_checkpointing
        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        current = in_channels
        for index, width in enumerate(widths):
            self.encoders.append(ResidualBlock(current, width))
            if index < len(widths) - 1:
                self.downsamples.append(
                    nn.Conv3d(width, widths[index + 1], 2, stride=2, bias=False)
                )
                current = widths[index + 1]
            else:
                current = width
        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for width in reversed(widths[:-1]):
            self.upsamples.append(
                nn.ConvTranspose3d(width * 2, width, 2, stride=2, bias=False)
            )
            self.decoders.append(ResidualBlock(width * 2, width))
        self.output = nn.Conv3d(widths[0], out_channels, 1)

    def _block(self, module, x):
        if self.gradient_checkpointing and self.training:
            return checkpoint(module, x, use_reentrant=False)
        return module(x)

    def forward(self, x):
        skips = []
        for index, encoder in enumerate(self.encoders):
            x = self._block(encoder, x)
            if index < len(self.downsamples):
                skips.append(x)
                x = self.downsamples[index](x)
        for upsample, decoder, skip in zip(
            self.upsamples, self.decoders, reversed(skips)
        ):
            x = upsample(x)
            x = self._block(decoder, torch.cat((x, skip), dim=1))
        return self.output(x)


class CanonicalUNet3D(nn.Module):
    """Standard five-level 3-D U-Net, trained from scratch."""
    def __init__(self, in_channels=4, out_channels=3, features=(32,64,128,256,512)):
        super().__init__()
        self.encoders=nn.ModuleList(); self.pools=nn.ModuleList()
        current=in_channels
        for f in features:
            self.encoders.append(nn.Sequential(nn.Conv3d(current,f,3,padding=1,bias=False),nn.InstanceNorm3d(f),nn.LeakyReLU(inplace=True),nn.Conv3d(f,f,3,padding=1,bias=False),nn.InstanceNorm3d(f),nn.LeakyReLU(inplace=True)))
            current=f
            if f != features[-1]: self.pools.append(nn.MaxPool3d(2))
        self.upconvs=nn.ModuleList(); self.decoders=nn.ModuleList()
        for f in reversed(features[:-1]):
            self.upconvs.append(nn.ConvTranspose3d(current,f,2,stride=2)); self.decoders.append(nn.Sequential(nn.Conv3d(f*2,f,3,padding=1,bias=False),nn.InstanceNorm3d(f),nn.LeakyReLU(inplace=True),nn.Conv3d(f,f,3,padding=1,bias=False),nn.InstanceNorm3d(f),nn.LeakyReLU(inplace=True))); current=f
        self.output=nn.Conv3d(features[0],out_channels,1)
    def forward(self,x):
        skips=[]
        for i,enc in enumerate(self.encoders):
            x=enc(x)
            if i<len(self.pools): skips.append(x); x=self.pools[i](x)
        for up,dec,skip in zip(self.upconvs,self.decoders,reversed(skips)):
            x=up(x)
            if x.shape[-3:] != skip.shape[-3:]: x=nn.functional.interpolate(x,size=skip.shape[-3:],mode="trilinear",align_corners=False)
            x=dec(torch.cat((x,skip),dim=1))
        return self.output(x)


class BandLimitedCanonicalUNet3D(nn.Module):
    """Canonical 3-D U-Net receiving one fixed complementary scale band."""
    def __init__(self, in_channels=4, out_channels=3,
                 features=(32, 64, 128, 256, 512), band="low",
                 fine_kernel=3, low_kernel=9):
        super().__init__()
        if band not in ("low", "mid", "high"):
            raise ValueError("band must be low, mid, or high")
        if fine_kernel % 2 != 1 or low_kernel % 2 != 1 or fine_kernel >= low_kernel:
            raise ValueError("Require odd fine_kernel < low_kernel")
        self.band = band
        self.fine_kernel = int(fine_kernel)
        self.low_kernel = int(low_kernel)
        self.unet = CanonicalUNet3D(in_channels, out_channels, features)

    @staticmethod
    def _blur(x, kernel):
        padding = kernel // 2
        return nn.functional.avg_pool3d(
            nn.functional.pad(x.float(), (padding,) * 6, mode="replicate"),
            kernel_size=kernel, stride=1,
        ).to(x.dtype)

    def fixed_view(self, x):
        low = self._blur(x, self.low_kernel)
        if self.band == "low":
            return low
        fine = self._blur(x, self.fine_kernel)
        if self.band == "mid":
            return fine - low
        return x - fine

    def forward(self, x):
        return self.unet(self.fixed_view(x))

    def experiment_spec(self):
        return {
            "model": type(self).__name__,
            "backbone": "CanonicalUNet3D",
            "input_role": self.band,
            "decomposition": "low=boxblur9; mid=boxblur3-boxblur9; high=input-boxblur3",
            "exact_reconstruction": "low+mid+high=input",
            "full_information_bypass": False,
            "role_parameters_matched": True,
        }


class CanonicalUNet3DMultiKernel(CanonicalUNet3D):
    """
    CanonicalUNet3D with selected decoder blocks' second 3x3 convolutions
    replaced by controlled multi-kernel selectors.
    """

    def __init__(
        self,
        in_channels=4,
        out_channels=3,
        features=(32, 64, 128, 256, 512),
        selection_mode="uniform",
        fixed_branch=None,
        selector_fixed_branches=None,
        top_k=2,
        selector_hidden_channels=None,
        router_jitter=0.0,
        selector_pooling="avg",
        selector_decoder_indices=(3,),
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            features=features,
        )
        indices = tuple(int(index) for index in selector_decoder_indices)
        if not indices or len(set(indices)) != len(indices):
            raise ValueError("selector_decoder_indices must be non-empty and unique")
        if any(index < 0 or index >= len(self.decoders) for index in indices):
            raise ValueError(
                f"selector_decoder_indices={indices} outside [0, {len(self.decoders) - 1}]"
            )
        self.selector_decoder_indices = indices
        if selector_fixed_branches is not None:
            if selection_mode != "fixed":
                raise ValueError(
                    "selector_fixed_branches is valid only for fixed selection mode"
                )
            stage_fixed_branches = tuple(selector_fixed_branches)
            if len(stage_fixed_branches) != len(indices):
                raise ValueError(
                    "selector_fixed_branches must match selector_decoder_indices"
                )
        else:
            stage_fixed_branches = (fixed_branch,) * len(indices)
        self.selector_fixed_branches = stage_fixed_branches
        decoder_channels = tuple(reversed(features[:-1]))
        for index, stage_fixed_branch in zip(indices, stage_fixed_branches):
            decoder = self.decoders[index]
            self.decoders[index] = nn.Sequential(
                decoder[0],
                decoder[1],
                decoder[2],
                MultiKernelSelector3D(
                    channels=int(decoder_channels[index]),
                    mode=selection_mode,
                    fixed_branch=stage_fixed_branch,
                    top_k=top_k,
                    selector_hidden_channels=selector_hidden_channels,
                    router_jitter=router_jitter,
                    selector_pooling=selector_pooling,
                ),
            )

    @property
    def multi_kernel_selector(self):
        return self.multi_kernel_selectors[-1]

    @property
    def multi_kernel_selectors(self):
        return [self.decoders[index][-1] for index in self.selector_decoder_indices]

    def auxiliary_loss(self):
        return sum(
            (selector.auxiliary_loss() for selector in self.multi_kernel_selectors),
            start=next(self.parameters()).sum() * 0.0,
        )

    def selection_snapshot(self):
        snapshots = [selector.selection_snapshot() for selector in self.multi_kernel_selectors]
        if any(snapshot is None for snapshot in snapshots):
            return None
        snapshots = [snapshot for snapshot in snapshots if snapshot is not None]
        result = dict(snapshots[-1])
        result["stage_snapshots"] = snapshots
        result["stage_indices"] = self.selector_decoder_indices
        for key in ("weights", "dense_probabilities", "contribution_rms", "entropy"):
            values = [snapshot.get(key) for snapshot in snapshots]
            if all(isinstance(value, torch.Tensor) for value in values):
                result[key] = torch.stack(values).mean(dim=0)
        selected = [snapshot.get("selected_mask") for snapshot in snapshots]
        if all(isinstance(value, torch.Tensor) for value in selected):
            result["selected_mask"] = torch.stack(selected).any(dim=0)
        else:
            result["selected_mask"] = None
        return result

    def expert_gradient_snapshot(self):
        stage_snapshots = [
            selector.expert_gradient_snapshot()
            for selector in self.multi_kernel_selectors
        ]
        branch_names = sorted({
            branch
            for snapshot in stage_snapshots
            for branch in snapshot
        })
        return {
            branch: any(snapshot.get(branch, False) for snapshot in stage_snapshots)
            for branch in branch_names
        }

    def experiment_spec(self):
        spec = self.multi_kernel_selector.experiment_spec()
        spec["model"] = type(self).__name__
        spec["backbone"] = "CanonicalUNet3D"
        spec["selector_decoder_indices"] = list(self.selector_decoder_indices)
        spec["selector_fixed_branches"] = list(self.selector_fixed_branches)
        spec["selector_stages"] = len(self.selector_decoder_indices)
        spec["insertion"] = "decoder_second_spatial_conv"
        return spec

    def set_intervention(self, kind, branch=None, renormalize=True):
        for selector in self.multi_kernel_selectors:
            selector.set_intervention(
                kind,
                branch=branch,
                renormalize=renormalize,
            )

    def clear_intervention(self):
        for selector in self.multi_kernel_selectors:
            selector.clear_intervention()


class CanonicalEncoderMultiKernelClassifier(nn.Module):
    """Canonical 3-D U-Net encoder with a bottleneck selector and case head."""

    def __init__(
        self,
        in_channels=4,
        num_classes=2,
        features=(32, 64, 128, 256, 512),
        selection_mode="uniform",
        fixed_branch=None,
        top_k=2,
        selector_hidden_channels=None,
        router_jitter=0.0,
        selector_pooling="avg",
        backbone_out_channels=3,
    ):
        super().__init__()
        backbone = CanonicalUNet3D(
            in_channels=in_channels,
            out_channels=backbone_out_channels,
            features=features,
        )
        self.encoders = backbone.encoders
        self.pools = backbone.pools
        bottleneck = self.encoders[-1]
        self.selector = MultiKernelSelector3D(
            channels=int(features[-1]),
            mode=selection_mode,
            fixed_branch=fixed_branch,
            top_k=top_k,
            selector_hidden_channels=selector_hidden_channels,
            router_jitter=router_jitter,
            selector_pooling=selector_pooling,
        )
        self.encoders[-1] = nn.Sequential(
            bottleneck[0], bottleneck[1], bottleneck[2], self.selector
        )
        self.classifier = nn.Linear(int(features[-1]), int(num_classes))

    def forward(self, x):
        for index, encoder in enumerate(self.encoders):
            x = encoder(x)
            if index < len(self.pools):
                x = self.pools[index](x)
        return self.classifier(x.mean(dim=(2, 3, 4)))

    def auxiliary_loss(self):
        return self.selector.auxiliary_loss()

    def selection_snapshot(self):
        return self.selector.selection_snapshot()

    def expert_gradient_snapshot(self):
        return self.selector.expert_gradient_snapshot()

    def experiment_spec(self):
        spec = self.selector.experiment_spec()
        spec.update({
            "model": type(self).__name__,
            "backbone": "CanonicalUNet3DEncoder",
            "task": "case_classification",
            "insertion": "encoder_bottleneck_second_spatial_conv",
        })
        return spec

    def set_intervention(self, kind, branch=None, renormalize=True):
        self.selector.set_intervention(
            kind, branch=branch, renormalize=renormalize
        )

    def clear_intervention(self):
        self.selector.clear_intervention()


class CanonicalUNet3DBottleneckMultiKernel(CanonicalUNet3D):
    """Canonical U-Net with the controlled selector in the encoder bottleneck."""

    def __init__(
        self,
        in_channels=4,
        out_channels=3,
        features=(32, 64, 128, 256, 512),
        selection_mode="uniform",
        fixed_branch=None,
        top_k=2,
        selector_hidden_channels=None,
        router_jitter=0.0,
        selector_pooling="spatial2",
    ):
        super().__init__(in_channels=in_channels, out_channels=out_channels, features=features)
        bottleneck = self.encoders[-1]
        self.selector = MultiKernelSelector3D(
            channels=int(features[-1]),
            mode=selection_mode,
            fixed_branch=fixed_branch,
            top_k=top_k,
            selector_hidden_channels=selector_hidden_channels,
            router_jitter=router_jitter,
            selector_pooling=selector_pooling,
        )
        self.encoders[-1] = nn.Sequential(
            bottleneck[0], bottleneck[1], bottleneck[2], self.selector
        )

    def auxiliary_loss(self):
        return self.selector.auxiliary_loss()

    def selection_snapshot(self):
        return self.selector.selection_snapshot()

    def expert_gradient_snapshot(self):
        return self.selector.expert_gradient_snapshot()

    def experiment_spec(self):
        spec = self.selector.experiment_spec()
        spec.update({
            "model": type(self).__name__,
            "backbone": "CanonicalUNet3D",
            "task": "segmentation",
            "insertion": "encoder_bottleneck_second_spatial_conv",
        })
        return spec

    def set_intervention(self, kind, branch=None, renormalize=True):
        self.selector.set_intervention(kind, branch=branch, renormalize=renormalize)

    def clear_intervention(self):
        self.selector.clear_intervention()


class CanonicalEncoderClassifier(nn.Module):
    """Selector-free canonical encoder baseline for case classification."""

    def __init__(
        self,
        in_channels=4,
        num_classes=2,
        features=(32, 64, 128, 256, 512),
        backbone_out_channels=3,
    ):
        super().__init__()
        backbone = CanonicalUNet3D(
            in_channels=in_channels,
            out_channels=backbone_out_channels,
            features=features,
        )
        self.encoders = backbone.encoders
        self.pools = backbone.pools
        self.classifier = nn.Linear(int(features[-1]), int(num_classes))

    def forward(self, x):
        for index, encoder in enumerate(self.encoders):
            x = encoder(x)
            if index < len(self.pools):
                x = self.pools[index](x)
        return self.classifier(x.mean(dim=(2, 3, 4)))

    def experiment_spec(self):
        return {
            "model": type(self).__name__,
            "backbone": "CanonicalUNet3DEncoder",
            "task": "case_classification",
            "insertion": None,
            "mode": "canonical",
            "branch_names": [],
        }


class CanonicalEncoderRFExpertClassifier(nn.Module):
    """Capacity-matched intermediate-resolution RF expert for case prediction.

    Every family sees the same frozen canonical encoder prefix and differs only
    in dilation of one 3x3x3 spatial convolution.  The classifier consumes a
    spatial 2x2x2 summary so coarse location is retained while the expert's
    pre-pooling receptive field remains explicitly controlled.
    """

    def __init__(
        self,
        in_channels=4,
        num_classes=2,
        features=(32, 64, 128, 256, 512),
        tap_stage=2,
        expert_dilation=1,
        backbone_out_channels=3,
    ):
        super().__init__()
        tap_stage = int(tap_stage)
        expert_dilation = int(expert_dilation)
        if tap_stage < 0 or tap_stage >= len(features) - 1:
            raise ValueError("tap_stage must address a non-bottleneck encoder stage")
        if expert_dilation < 1:
            raise ValueError("expert_dilation must be positive")
        backbone = CanonicalUNet3D(
            in_channels=in_channels,
            out_channels=backbone_out_channels,
            features=features,
        )
        self.tap_stage = tap_stage
        self.expert_dilation = expert_dilation
        self.encoders = nn.ModuleList(list(backbone.encoders[: tap_stage + 1]))
        self.pools = nn.ModuleList(list(backbone.pools[:tap_stage]))
        channels = int(features[tap_stage])
        self.expert = nn.Sequential(
            nn.Conv3d(
                channels,
                channels,
                kernel_size=3,
                padding=expert_dilation,
                dilation=expert_dilation,
                bias=False,
            ),
            nn.InstanceNorm3d(channels),
            nn.LeakyReLU(inplace=True),
        )
        self.classifier = nn.Linear(channels * 8, int(num_classes))

    def forward(self, x):
        for index, encoder in enumerate(self.encoders):
            x = encoder(x)
            if index < len(self.pools):
                x = self.pools[index](x)
        x = self.expert(x)
        x = nn.functional.adaptive_avg_pool3d(x, (2, 2, 2)).flatten(1)
        return self.classifier(x)

    def experiment_spec(self):
        jump = 2 ** self.tap_stage
        canonical_prefix_rf = 1
        prefix_jump = 1
        for stage in range(self.tap_stage + 1):
            canonical_prefix_rf += 4 * prefix_jump
            if stage < self.tap_stage:
                canonical_prefix_rf += prefix_jump
                prefix_jump *= 2
        prepool_rf = canonical_prefix_rf + 2 * self.expert_dilation * jump
        return {
            "model": type(self).__name__,
            "backbone": "frozen_CanonicalUNet3D_encoder_prefix",
            "task": "case_classification",
            "tap_stage": self.tap_stage,
            "feature_stride_voxels": jump,
            "expert_kernel": [3, 3, 3],
            "expert_dilation": [self.expert_dilation] * 3,
            "estimated_prepool_receptive_field_voxels": prepool_rf,
            "pooling": "adaptive_avg_2x2x2",
            "capacity_matching": "same channels, kernel parameters, head, and pooling across dilations",
        }


class CanonicalEncoderStageExpertClassifier(nn.Module):
    """Capacity-matched readout from a fixed encoder resolution stage.

    Every variant computes the same frozen encoder through stage 3. The tapped
    feature is average-pooled to stage-3 resolution, projected to 64 channels
    with a grouped 1x1 convolution containing exactly 4096 weights at every
    stage, and passed through the same adapter and classifier.
    """
    def __init__(self, in_channels=4, num_classes=2,
                 features=(32, 64, 128, 256, 512), tap_stage=1,
                 common_channels=64, backbone_out_channels=3):
        super().__init__()
        tap_stage=int(tap_stage); common_channels=int(common_channels)
        if tap_stage not in (1, 2, 3):
            raise ValueError("stage expert tap_stage must be 1, 2, or 3")
        source_channels=int(features[tap_stage])
        if source_channels % common_channels or common_channels % (source_channels // common_channels):
            raise ValueError("channels do not support parameter-matched grouped projection")
        backbone=CanonicalUNet3D(in_channels=in_channels,out_channels=backbone_out_channels,features=features)
        self.encoders=nn.ModuleList(list(backbone.encoders[:4]))
        self.pools=nn.ModuleList(list(backbone.pools[:3]))
        self.tap_stage=tap_stage
        groups=source_channels // common_channels
        self.projection=nn.Sequential(
            nn.Conv3d(source_channels,common_channels,1,groups=groups,bias=False),
            nn.InstanceNorm3d(common_channels),nn.LeakyReLU(inplace=True),
        )
        self.expert=nn.Sequential(
            nn.Conv3d(common_channels,common_channels,3,padding=1,bias=False),
            nn.InstanceNorm3d(common_channels),nn.LeakyReLU(inplace=True),
        )
        self.classifier=nn.Linear(common_channels*8,int(num_classes))

    def forward(self,x):
        tapped=None
        for index,encoder in enumerate(self.encoders):
            x=encoder(x)
            if index==self.tap_stage:
                factor=2 ** (3-index)
                tapped=nn.functional.avg_pool3d(x,kernel_size=factor,stride=factor) if factor>1 else x
            if index<len(self.pools): x=self.pools[index](x)
        if tapped is None: raise RuntimeError("stage tap was not reached")
        tapped=self.expert(self.projection(tapped))
        tapped=nn.functional.adaptive_avg_pool3d(tapped,(2,2,2)).flatten(1)
        return self.classifier(tapped)

    def experiment_spec(self):
        rf={1:18,2:40,3:84}[self.tap_stage]
        return {"model":type(self).__name__,"backbone":"frozen_CanonicalUNet3D_encoder_0_to_3",
                "task":"case_classification","tap_stage":self.tap_stage,
                "source_feature_stride_voxels":2**self.tap_stage,
                "common_readout_stride_voxels":8,"estimated_adapter_rf_voxels":rf,
                "projection":"grouped_1x1_exact_4096_weights","common_channels":64,
                "pooling":"adaptive_avg_2x2x2",
                "capacity_matching":"same frozen encoder compute, adapter parameters, head and pooling"}
