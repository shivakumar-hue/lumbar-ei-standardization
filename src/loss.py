"""
loss.py
-------
Combined Binary Cross-Entropy + Dice loss for muscle segmentation.

Formulation:
    L_total = 0.5 * L_BCE + 0.5 * L_Dice

    L_Dice  = 1 - (2|P∩T| + 1) / (|P| + |T| + 1)

Why combined?
    BCE  : stable per-pixel probability calibration
    Dice : directly optimises the spatial overlap metric (IoU proxy)
    Together: faster convergence, better performance on imbalanced
              tasks where background pixels far outnumber muscle pixels.
"""

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Soft Dice loss over a batch of predicted logits."""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                smooth: float = 1.0) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice  = (2.0 * inter + smooth) / (union + smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """
    Weighted combination of BCE and Dice losses.

    Args:
        weight : fraction allocated to BCE (1-weight goes to Dice)
    """

    def __init__(self, weight: float = 0.5):
        super().__init__()
        self.w    = weight
        self.bce  = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        return self.w * self.bce(logits, targets) + \
               (1.0 - self.w) * self.dice(logits, targets)
