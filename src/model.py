"""
model.py
--------
U-Net architecture for Lumbar Multifidus muscle segmentation.

Architecture:
    Encoder  : 4 × DoubleConv blocks (64→128→256→512 ch) + MaxPool 2×2
    Bottleneck: DoubleConv 1024 ch  with Dropout(0.2)
    Decoder  : 4 × ConvTranspose2d 2×2  + DoubleConv with skip connections
    Output   : Conv 1×1 → raw logits (apply sigmoid externally)

Input  : (B, 1, 256, 256)  — single-channel grayscale
Output : (B, 1, 256, 256)  — raw logits; threshold at 0.5 after sigmoid
Params : ~31 M
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two consecutive Conv-BN-ReLU blocks with optional dropout."""

    def __init__(self, in_ch: int, out_ch: int, drop: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(drop),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNET(nn.Module):
    """
    Standard U-Net with 4-level encoder-decoder and skip connections.

    Args:
        in_ch  : input channels (1 for grayscale)
        out_ch : output channels (1 for binary segmentation)
        feats  : channel widths at each encoder level
    """

    def __init__(self, in_ch: int = 1, out_ch: int = 1,
                 feats: list = [64, 128, 256, 512]):
        super().__init__()
        self.downs      = nn.ModuleList()
        self.ups        = nn.ModuleList()
        self.pool       = nn.MaxPool2d(2, 2)

        # Encoder
        for f in feats:
            self.downs.append(DoubleConv(in_ch, f))
            in_ch = f

        # Bottleneck
        self.bottleneck = DoubleConv(feats[-1], feats[-1] * 2, drop=0.2)

        # Decoder
        for f in reversed(feats):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(f * 2, f))

        self.final = nn.Conv2d(feats[0], out_ch, kernel_size=1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            s = skips[-(i // 2 + 1)]
            # Handle size mismatch from odd input dimensions
            if x.shape != s.shape:
                x = F.interpolate(x, size=s.shape[2:])
            x = torch.cat([s, x], dim=1)
            x = self.ups[i + 1](x)

        return self.final(x)


def build_model(device: torch.device) -> UNET:
    """Instantiate U-Net and print parameter count."""
    model = UNET().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"U-Net params: {n_params:,}")
    return model
