from __future__ import annotations

import torch
from torch import Tensor


def topk_accuracy(logits: Tensor, targets: Tensor, topk: tuple[int, ...] = (1, 5)) -> list[float]:
    with torch.no_grad():
        maxk = max(topk)
        _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1))
        out: list[float] = []
        for k in topk:
            ck = correct[:k].reshape(-1).float().sum().item()
            out.append(100.0 * ck / targets.size(0))
        return out


class ConfusionMatrix:
    def __init__(self, num_classes: int, device: torch.device) -> None:
        self.num_classes = num_classes
        self.matrix = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)

    def update(self, preds: Tensor, targets: Tensor) -> None:
        preds = preds.view(-1)
        targets = targets.view(-1)
        k = (targets >= 0) & (targets < self.num_classes)
        inds = self.num_classes * targets[k] + preds[k]
        self.matrix += torch.bincount(inds, minlength=self.num_classes**2).reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict[str, Tensor | float]:
        cm = self.matrix.float()
        tp = torch.diag(cm)
        precision = tp / (cm.sum(0) + 1e-8)
        recall = tp / (cm.sum(1) + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        acc = tp.sum() / (cm.sum() + 1e-8)
        return {
            "confusion_matrix": cm,
            "accuracy": float(acc.item()),
            "precision": float(precision.mean().item()),
            "recall": float(recall.mean().item()),
            "f1": float(f1.mean().item()),
        }
