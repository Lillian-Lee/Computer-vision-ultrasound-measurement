"""Direct-regression CNN baseline: image -> [EMA, EMD, EMW, fat C] with no explicit mask.

This is the "black box" alternative to segmentation-then-measure. It is included so the
results can answer the obvious question: *is segmentation needed at all, or can a
CNN just read the numbers off the image?*

Trade-offs discussed in docs/03-results.md: direct regression is simpler and fast, but
it cannot show a technician *where* it measured, it cannot flag out-of-field frames, and
it silently learns the pixel spacing of the training scanner (so it breaks if the depth
setting changes) - whereas the segmentation route takes mm/px as an explicit input.
"""
from __future__ import annotations

import torch
from torch import nn


def conv_bn(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class ResBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = conv_bn(cin, cout, stride)
        self.c2 = nn.Sequential(nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout))
        self.skip = nn.Identity() if (cin == cout and stride == 1) else nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        return torch.relu(self.c2(self.c1(x)) + self.skip(x))


class MeasurementRegressor(nn.Module):
    def __init__(self, in_ch: int = 1, n_outputs: int = 4, widths=(16, 32, 64, 128, 192), dropout: float = 0.2):
        super().__init__()
        layers = [conv_bn(in_ch, widths[0])]
        for i in range(1, len(widths)):
            layers.append(ResBlock(widths[i - 1], widths[i], stride=2))
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout),
                                  nn.Linear(widths[-1], 128), nn.ReLU(inplace=True), nn.Linear(128, n_outputs))

    def forward(self, x):
        return self.head(self.features(x))
