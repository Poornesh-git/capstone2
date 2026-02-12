from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from sgvit.training.metrics import ConfusionMatrix
from sgvit.utils.checkpoint import load_checkpoint, save_checkpoint
from sgvit.utils.config import TrainConfig


@dataclass
class EpochResult:
    loss: float
    acc1: float


class Trainer:
    def __init__(self, model: nn.Module, cfg: TrainConfig, train_loader: DataLoader, val_loader: DataLoader) -> None:
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr)
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = torch.amp.GradScaler(enabled=cfg.amp and self.device.type == "cuda")

        self.start_epoch = 0
        self.best_acc = 0.0
        self.ckpt_path = str(Path(cfg.output_dir) / "best.pt")

        if cfg.resume:
            self.start_epoch, self.best_acc = load_checkpoint(
                cfg.resume, self.model, self.optimizer, self.scheduler, map_location=self.device.type
            )

    def _run_epoch(self, train: bool) -> EpochResult:
        loader = self.train_loader if train else self.val_loader
        self.model.train(train)

        total_loss = 0.0
        correct = 0
        seen = 0

        for skeleton, video, labels in loader:
            skeleton = skeleton.to(self.device, non_blocking=True)
            video = video.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with torch.set_grad_enabled(train), torch.autocast(device_type=self.device.type, enabled=self.scaler.is_enabled()):
                out = self.model(skeleton, video)
                logits = out["logits"]
                loss = self.criterion(logits, labels)

            if train:
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()

            total_loss += loss.item() * labels.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            seen += labels.size(0)

        return EpochResult(loss=total_loss / max(1, seen), acc1=100.0 * correct / max(1, seen))

    @torch.no_grad()
    def evaluate(self, loader: DataLoader | None = None) -> dict[str, float | torch.Tensor]:
        loader = loader or self.val_loader
        self.model.eval()
        cm = ConfusionMatrix(self.cfg.model.num_classes, self.device)
        total_loss = 0.0
        total_n = 0
        for skeleton, video, labels in loader:
            skeleton = skeleton.to(self.device, non_blocking=True)
            video = video.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            logits = self.model(skeleton, video)["logits"]
            loss = self.criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            total_n += labels.size(0)
            cm.update(logits.argmax(dim=1), labels)

        out = cm.compute()
        out["loss"] = total_loss / max(1, total_n)
        return out

    def fit(self) -> None:
        for epoch in range(self.start_epoch, self.cfg.epochs):
            train_res = self._run_epoch(train=True)
            val_metrics = self.evaluate()
            self.scheduler.step()

            acc = float(val_metrics["accuracy"]) * 100.0
            print(
                f"Epoch {epoch + 1}/{self.cfg.epochs} | "
                f"train_loss={train_res.loss:.4f} train_acc={train_res.acc1:.2f} | "
                f"val_loss={val_metrics['loss']:.4f} val_acc={acc:.2f}"
            )
            if acc > self.best_acc:
                self.best_acc = acc
                save_checkpoint(self.ckpt_path, self.model, self.optimizer, self.scheduler, epoch, self.best_acc)

        print(f"Best validation accuracy: {self.best_acc:.2f}")
