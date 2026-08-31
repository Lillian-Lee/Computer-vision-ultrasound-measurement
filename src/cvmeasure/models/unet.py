"""U-Net for eye-muscle / fat segmentation of B-mode ultrasound.

A compact encoder-decoder with skip connections (Ronneberger et al., 2015). Ultrasound
frames are single-channel and structures are large relative to the frame, so a shallow
U-Net with 16-128 channels is sufficient and trains in minutes on CPU; ``base`` scales it.

Optionally the encoder can be swapped for an ImageNet-pretrained ResNet-18
(``encoder="resnet18"``) - useful when only a few hundred real annotated frames exist.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, cin, cout, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_ch: int = 1, n_classes: int = 3, base: int = 16, depth: int = 4, dropout: float = 0.1):
        super().__init__()
        chs = [base * 2 ** i for i in range(depth + 1)]      # e.g. 16,32,64,128,256
        self.inc = DoubleConv(in_ch, chs[0])
        self.downs = nn.ModuleList([DoubleConv(chs[i], chs[i + 1], dropout) for i in range(depth)])
        self.ups = nn.ModuleList([nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2) for i in reversed(range(depth))])
        self.upconvs = nn.ModuleList([DoubleConv(chs[i] * 2, chs[i], dropout) for i in reversed(range(depth))])
        self.outc = nn.Conv2d(chs[0], n_classes, 1)

    def forward(self, x):
        feats = [self.inc(x)]
        for d in self.downs:
            feats.append(d(F.max_pool2d(feats[-1], 2)))
        y = feats[-1]
        for up, conv, skip in zip(self.ups, self.upconvs, reversed(feats[:-1])):
            y = up(y)
            if y.shape[-2:] != skip.shape[-2:]:
                y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = conv(torch.cat([skip, y], 1))
        return self.outc(y)


class ResNetUNet(nn.Module):
    """U-Net with a torchvision ResNet-18 encoder (ImageNet weights optional)."""

    def __init__(self, n_classes: int = 3, pretrained: bool = True):
        super().__init__()
        import torchvision
        weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        r = torchvision.models.resnet18(weights=weights)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)      # /2, 64
        self.pool = r.maxpool                                    # /4
        self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4   # 64,128,256,512
        self.up4 = DoubleConv(512 + 256, 256)
        self.up3 = DoubleConv(256 + 128, 128)
        self.up2 = DoubleConv(128 + 64, 64)
        self.up1 = DoubleConv(64 + 64, 32)
        self.up0 = DoubleConv(32, 16)
        self.outc = nn.Conv2d(16, n_classes, 1)

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        s0 = self.stem(x)                 # /2
        s1 = self.l1(self.pool(s0))       # /4
        s2 = self.l2(s1)                  # /8
        s3 = self.l3(s2)                  # /16
        s4 = self.l4(s3)                  # /32
        up = lambda t, ref: F.interpolate(t, size=ref.shape[-2:], mode="bilinear", align_corners=False)  # noqa
        y = self.up4(torch.cat([up(s4, s3), s3], 1))
        y = self.up3(torch.cat([up(y, s2), s2], 1))
        y = self.up2(torch.cat([up(y, s1), s1], 1))
        y = self.up1(torch.cat([up(y, s0), s0], 1))
        y = self.up0(up(y, x))
        return self.outc(y)


def build_segmentation_model(name: str = "unet", **kw) -> nn.Module:
    if name == "unet":
        return UNet(**kw)
    if name == "resnet18_unet":
        return ResNetUNet(**kw)
    raise ValueError(name)


class DiceCELoss(nn.Module):
    """Cross-entropy + soft multi-class Dice (foreground classes only). Standard for
    ultrasound segmentation where the muscle/fat classes are small vs background."""

    def __init__(self, ce_weight: float = 1.0, dice_weight: float = 1.0, eps: float = 1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.ce_weight, self.dice_weight, self.eps = ce_weight, dice_weight, eps

    def forward(self, logits, target):
        ce = self.ce(logits, target)
        probs = logits.softmax(1)
        onehot = F.one_hot(target, logits.shape[1]).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        inter = (probs * onehot).sum(dims)[1:]
        denom = (probs + onehot).sum(dims)[1:]
        dice = 1 - ((2 * inter + self.eps) / (denom + self.eps)).mean()
        return self.ce_weight * ce + self.dice_weight * dice
