from __future__ import annotations

from typing import List, Tuple

import torch
from torch import Tensor, nn

from .blocks import PatchMerging2D, SwinBlock


class SwinSkeletonEncoder(nn.Module):
    """Hierarchical Swin-style encoder for skeleton sequences.

    Input: (B, T, V, 3)
    Output:
        skeleton_tokens: (B, Ns, D)
        skeleton_global: (B, D)
    """

    def __init__(
        self,
        embed_dim: int = 96,
        depths: List[int] | Tuple[int, ...] = (2, 2),
        num_heads: List[int] | Tuple[int, ...] = (3, 6),
        patch_size: int = 2,
        window_size: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_drop = nn.Dropout(dropout)

        self.stages = nn.ModuleList()
        self.mergers = nn.ModuleList()
        dim = embed_dim

        for stage_idx, depth in enumerate(depths):
            blocks = nn.ModuleList(
                [
                    SwinBlock(
                        dim=dim,
                        num_heads=num_heads[stage_idx],
                        window_size=window_size,
                        shift_size=0 if i % 2 == 0 else window_size // 2,
                        dropout=dropout,
                    )
                    for i in range(depth)
                ]
            )
            self.stages.append(blocks)
            if stage_idx < len(depths) - 1:
                self.mergers.append(PatchMerging2D(dim))
                dim *= 2

        self.out_dim = dim
        self.norm = nn.LayerNorm(self.out_dim)

    def forward(self, skeleton: Tensor) -> tuple[Tensor, Tensor]:
        b, t, v, c = skeleton.shape
        assert c == 3, "Skeleton last dimension must be 3 (x,y,z)."

        x = skeleton.permute(0, 3, 1, 2).contiguous()  # (B,3,T,V)
        x = self.patch_embed(x)  # (B,D,H,W)
        h, w = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2).contiguous()  # (B, H*W, D)
        x = self.pos_drop(x)

        for stage_idx, blocks in enumerate(self.stages):
            for blk in blocks:
                x = blk(x, h, w)
            if stage_idx < len(self.stages) - 1:
                x, h, w = self.mergers[stage_idx](x, h, w)

        x = self.norm(x)
        skeleton_tokens = x
        skeleton_global = x.mean(dim=1)
        return skeleton_tokens, skeleton_global
