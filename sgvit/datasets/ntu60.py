from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .transforms import build_video_transform, load_and_transform_frames, uniform_indices


@dataclass(frozen=True)
class SampleRecord:
    skeleton_path: str
    frame_dir: str
    label: int


class NTU60Dataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """NTU60 RGB+Skeleton dataset.

    Expected split file format (CSV with header):
    skeleton_path,frame_dir,label
    """

    def __init__(
        self,
        root: str,
        split_file: str,
        num_frames: int = 64,
        num_joints: int = 25,
        image_size: int = 224,
        train: bool = True,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.num_frames = num_frames
        self.num_joints = num_joints
        self.video_transform = build_video_transform(train=train, image_size=image_size)
        self.samples = self._read_split(self.root / split_file)

    def _read_split(self, split_path: Path) -> list[SampleRecord]:
        records: list[SampleRecord] = []
        with split_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(
                    SampleRecord(
                        skeleton_path=str(self.root / row["skeleton_path"]),
                        frame_dir=str(self.root / row["frame_dir"]),
                        label=int(row["label"]),
                    )
                )
        return records

    def __len__(self) -> int:
        return len(self.samples)

    def _load_skeleton(self, path: str) -> Tensor:
        arr = np.load(path)
        # Accept (T,V,3) or (T,M,V,3)
        if arr.ndim == 4:
            energy = np.linalg.norm(arr, axis=-1).sum(axis=(0, 2))
            pid = int(np.argmax(energy))
            arr = arr[:, pid]
        if arr.shape[-1] != 3:
            raise ValueError(f"Invalid skeleton shape at {path}: {arr.shape}")
        if arr.shape[1] != self.num_joints:
            arr = arr[:, : self.num_joints]
            if arr.shape[1] < self.num_joints:
                pad = np.zeros((arr.shape[0], self.num_joints - arr.shape[1], 3), dtype=arr.dtype)
                arr = np.concatenate([arr, pad], axis=1)

        idx = uniform_indices(arr.shape[0], self.num_frames)
        arr = arr[idx].astype(np.float32)

        mean = arr.mean(axis=(0, 1), keepdims=True)
        std = arr.std(axis=(0, 1), keepdims=True) + 1e-6
        arr = (arr - mean) / std
        return torch.from_numpy(arr)

    def _load_video(self, frame_dir: str) -> Tensor:
        frame_paths = sorted([str(p) for p in Path(frame_dir).glob("*.jpg")])
        if not frame_paths:
            frame_paths = sorted([str(p) for p in Path(frame_dir).glob("*.png")])
        return load_and_transform_frames(frame_paths, self.num_frames, self.video_transform)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        record = self.samples[index]
        skeleton = self._load_skeleton(record.skeleton_path)
        video = self._load_video(record.frame_dir)
        label = torch.tensor(record.label, dtype=torch.long)
        return skeleton, video, label
