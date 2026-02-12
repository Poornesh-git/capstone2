from __future__ import annotations

import torch
from torch import Tensor, nn


class PoseAwareTokenPruner(nn.Module):
    """Pose-aware dynamic token pruning based on skeleton motion saliency."""

    def __init__(self, keep_ratio: float = 0.5) -> None:
        super().__init__()
        if not (0.0 < keep_ratio <= 1.0):
            raise ValueError("keep_ratio must be in (0,1].")
        self.keep_ratio = keep_ratio

    @staticmethod
    def compute_frame_saliency(skeleton: Tensor) -> Tensor:
        """Compute per-frame saliency from joint velocity.

        skeleton: (B,T,V,3)
        returns: (B,T)
        """
        velocity = skeleton[:, 1:] - skeleton[:, :-1]
        speed = torch.linalg.norm(velocity, dim=-1).mean(dim=-1)  # (B,T-1)
        first = speed[:, :1]
        frame_sal = torch.cat([first, speed], dim=1)
        frame_sal = frame_sal / (frame_sal.amax(dim=1, keepdim=True) + 1e-6)
        return frame_sal

    def forward(
        self,
        skeleton: Tensor,
        video_tokens: Tensor,
        pos_tokens: Tensor,
        num_tubes: int,
        spatial_per_tube: int,
        tubelet_size: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Prune tokens by skeleton-guided saliency.

        Returns pruned_tokens, pruned_pos, kept_indices.
        """
        b, n, d = video_tokens.shape
        frame_sal = self.compute_frame_saliency(skeleton)  # (B,T)
        total_frames = frame_sal.shape[1]

        usable = num_tubes * tubelet_size
        if usable > total_frames:
            pad = usable - total_frames
            frame_sal = torch.cat([frame_sal, frame_sal[:, -1:].expand(-1, pad)], dim=1)
        else:
            frame_sal = frame_sal[:, :usable]

        tube_sal = frame_sal.view(b, num_tubes, tubelet_size).mean(dim=-1)  # (B,num_tubes)
        token_sal = tube_sal.unsqueeze(-1).expand(-1, -1, spatial_per_tube).reshape(b, -1)
        assert token_sal.shape[1] == n

        keep_n = max(1, int(n * self.keep_ratio))
        keep_idx = torch.topk(token_sal, k=keep_n, dim=1, largest=True, sorted=True).indices

        gather_idx = keep_idx.unsqueeze(-1).expand(-1, -1, d)
        pruned_tokens = torch.gather(video_tokens, dim=1, index=gather_idx)
        pruned_pos = torch.gather(pos_tokens, dim=1, index=gather_idx)
        return pruned_tokens, pruned_pos, keep_idx
