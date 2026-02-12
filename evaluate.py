from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from sgvit.datasets import NTU60Dataset
from sgvit.models import SGViT
from sgvit.training.metrics import topk_accuracy
from sgvit.utils import load_checkpoint, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SG-ViT")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_ds = NTU60Dataset(
        root=cfg.data_root,
        split_file=cfg.test_split,
        num_frames=cfg.model.video_frames,
        image_size=cfg.model.img_size,
        train=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    model = SGViT(cfg.model).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device.type)
    model.eval()

    top1_sum, top5_sum, n = 0.0, 0.0, 0
    with torch.no_grad():
        for skeleton, video, labels in test_loader:
            skeleton = skeleton.to(device, non_blocking=True)
            video = video.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(skeleton, video)["logits"]
            acc1, acc5 = topk_accuracy(logits, labels, topk=(1, 5))
            bsz = labels.size(0)
            top1_sum += acc1 * bsz
            top5_sum += acc5 * bsz
            n += bsz

    print(f"Top-1: {top1_sum / max(1, n):.2f}")
    print(f"Top-5: {top5_sum / max(1, n):.2f}")


if __name__ == "__main__":
    main()
