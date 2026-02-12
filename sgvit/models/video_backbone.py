from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from .blocks import VideoTransformerBlock


class VideoTransformerBackbone(nn.Module):
    """Video transformer with tubelet embedding and token-level outputs."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        tubelet_size: int = 2,
        in_chans: int = 3,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_frames: int = 64,
    ) -> None:
        super().__init__()
        assert img_size % patch_size == 0
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.embed_dim = embed_dim
        self.grid_h = img_size // patch_size
        self.grid_w = img_size // patch_size
        self.spatial_per_tube = self.grid_h * self.grid_w
        self.max_tubes = max_frames // tubelet_size

        self.tubelet_embed = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_tubes * self.spatial_per_tube, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [VideoTransformerBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def embed(self, video: Tensor) -> tuple[Tensor, Tensor]:
        """Create tubelet tokens and corresponding positional encodings.

        video: (B,T,3,H,W)
        returns tokens, pos: (B,N,D), (B,N,D)
        """
        x = video.permute(0, 2, 1, 3, 4).contiguous()  # (B,C,T,H,W)
        x = self.tubelet_embed(x)  # (B,D,T',H',W')
        b, d, tt, hh, ww = x.shape
        n = tt * hh * ww
        tokens = x.flatten(2).transpose(1, 2).contiguous()
        pos = self.pos_embed[:, :n, :].expand(b, -1, -1)
        return tokens, pos

    def forward_tokens(self, tokens: Tensor, pos: Optional[Tensor] = None, start_block: int = 0) -> Tensor:
        if pos is not None:
            tokens = tokens + pos
        tokens = self.pos_drop(tokens)
        for blk in self.blocks[start_block:]:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        return tokens

    def forward(self, video: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        tokens, pos = self.embed(video)
        tokens = self.forward_tokens(tokens, pos=pos)
        global_feat = tokens.mean(dim=1)
        return tokens, global_feat, pos
