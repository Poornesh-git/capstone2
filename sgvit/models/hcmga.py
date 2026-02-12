from __future__ import annotations

import torch
from torch import Tensor, nn


class HCMGA(nn.Module):
    """Hierarchical Cross-Modal Guided Attention (video queries skeleton)."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.gamma = nn.Parameter(torch.zeros(1))
        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(self, video_tokens: Tensor, skeleton_tokens: Tensor) -> Tensor:
        q = self.norm_q(video_tokens)
        kv = self.norm_kv(skeleton_tokens)
        cross, _ = self.attn(q, kv, kv, need_weights=False)
        fused = video_tokens + self.gamma * cross
        return self.out_norm(fused)
