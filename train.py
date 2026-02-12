from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from sgvit.datasets import NTU60Dataset
from sgvit.models import SGViT
from sgvit.training import Trainer
from sgvit.utils import load_config, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SG-ViT")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    train_ds = NTU60Dataset(
        root=cfg.data_root,
        split_file=cfg.train_split,
        num_frames=cfg.model.video_frames,
        image_size=cfg.model.img_size,
        train=True,
    )
    val_ds = NTU60Dataset(
        root=cfg.data_root,
        split_file=cfg.val_split,
        num_frames=cfg.model.video_frames,
        image_size=cfg.model.img_size,
        train=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    model = SGViT(cfg.model)
    trainer = Trainer(model, cfg, train_loader, val_loader)
    trainer.fit()


if __name__ == "__main__":
    main()
