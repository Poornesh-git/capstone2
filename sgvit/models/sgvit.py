from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch
from torch import Tensor, nn

from .hcmga import HCMGA
from .swin_skeleton import SwinSkeletonEncoder
from .token_pruning import PoseAwareTokenPruner
from .video_backbone import VideoTransformerBackbone


@dataclass
class SGViTConfig:
    num_classes: int = 60
    img_size: int = 224
    video_frames: int = 64
    video_patch_size: int = 16
    tubelet_size: int = 2
    video_embed_dim: int = 384
    video_depth: int = 12
    video_heads: int = 6
    skeleton_embed_dim: int = 96
    skeleton_depths: Sequence[int] = field(default_factory=lambda: (2, 2))
    skeleton_heads: Sequence[int] = field(default_factory=lambda: (3, 6))
    skeleton_patch_size: int = 2
    keep_ratio: float = 0.6
    hcmga_heads: int = 6
    hcmga_layers: Sequence[int] = field(default_factory=lambda: (4, 8, 11))
    dropout: float = 0.1
    late_fusion: bool = False


class SGViT(nn.Module):
    def __init__(self, config: SGViTConfig) -> None:
        super().__init__()
        self.config = config
        self.num_classes = config.num_classes

        self.skeleton_encoder = SwinSkeletonEncoder(
            embed_dim=config.skeleton_embed_dim,
            depths=list(config.skeleton_depths),
            num_heads=list(config.skeleton_heads),
            patch_size=config.skeleton_patch_size,
            dropout=config.dropout,
        )
        self.video_backbone = VideoTransformerBackbone(
            img_size=config.img_size,
            patch_size=config.video_patch_size,
            tubelet_size=config.tubelet_size,
            embed_dim=config.video_embed_dim,
            depth=config.video_depth,
            num_heads=config.video_heads,
            dropout=config.dropout,
            max_frames=config.video_frames,
        )
        self.pruner = PoseAwareTokenPruner(keep_ratio=config.keep_ratio)

        self.skel_to_video = nn.Linear(self.skeleton_encoder.out_dim, config.video_embed_dim)
        self.hcmga_layers = set(config.hcmga_layers)
        self.hcmga = nn.ModuleDict(
            {str(i): HCMGA(config.video_embed_dim, config.hcmga_heads, dropout=config.dropout) for i in self.hcmga_layers}
        )

        head_in_dim = config.video_embed_dim * (2 if config.late_fusion else 1)
        self.classifier = nn.Linear(head_in_dim, config.num_classes)

    def forward(self, skeleton: Tensor, video: Tensor) -> dict[str, Tensor]:
        skeleton_tokens, skeleton_global = self.skeleton_encoder(skeleton)
        skeleton_tokens = self.skel_to_video(skeleton_tokens)
        skeleton_global = self.skel_to_video(skeleton_global)

        video_tokens, pos = self.video_backbone.embed(video)
        b, n, d = video_tokens.shape
        num_tubes = video.shape[1] // self.config.tubelet_size
        pruned_tokens, pruned_pos, keep_idx = self.pruner(
            skeleton=skeleton,
            video_tokens=video_tokens,
            pos_tokens=pos,
            num_tubes=num_tubes,
            spatial_per_tube=self.video_backbone.spatial_per_tube,
            tubelet_size=self.config.tubelet_size,
        )

        x = self.video_backbone.pos_drop(pruned_tokens + pruned_pos)
        for i, blk in enumerate(self.video_backbone.blocks):
            x = blk(x)
            if i in self.hcmga_layers:
                x = self.hcmga[str(i)](x, skeleton_tokens)

        x = self.video_backbone.norm(x)
        video_tokens_out = x
        video_global = x.mean(dim=1)

        if self.config.late_fusion:
            fused_global = torch.cat([video_global, skeleton_global], dim=-1)
        else:
            fused_global = video_global

        logits = self.classifier(fused_global)
        return {
            "logits": logits,
            "video_tokens": video_tokens_out,
            "video_global": video_global,
            "skeleton_tokens": skeleton_tokens,
            "skeleton_global": skeleton_global,
            "kept_token_indices": keep_idx,
        }
