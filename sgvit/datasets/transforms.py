from __future__ import annotations

from typing import List

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T


def build_video_transform(train: bool, image_size: int) -> T.Compose:
    if train:
        return T.Compose(
            [
                T.Resize((image_size + 32, image_size + 32)),
                T.RandomCrop((image_size, image_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def uniform_indices(length: int, target: int) -> np.ndarray:
    if length <= 0:
        return np.zeros(target, dtype=np.int64)
    if length >= target:
        return np.linspace(0, length - 1, target, dtype=np.int64)
    reps = int(np.ceil(target / length))
    idx = np.tile(np.arange(length), reps)[:target]
    return idx


def load_and_transform_frames(paths: List[str], target_len: int, transform: T.Compose) -> torch.Tensor:
    idx = uniform_indices(len(paths), target_len)
    frames: list[torch.Tensor] = []
    for i in idx:
        if len(paths) == 0:
            img_t = torch.zeros(3, 224, 224)
        else:
            img = Image.open(paths[i]).convert("RGB")
            img_t = transform(img)
        frames.append(img_t)
    return torch.stack(frames, dim=0)
