"""victim_resnet.py — second victim model (Wave 6, G2): 2-channel ResNet.

Why a second victim: every published attack number so far was crafted against
the AUTHORS' OWN model in the SAME codebase — a test suite that has only ever
attacked its author's model proves nothing about anyone else's model. This
victim is architecturally different from the InceptionTime dual-stream CNN:

  * early fusion: (mag, ifr) stacked as 2 input channels (vs late fusion)
  * residual blocks with pre-pool downsampling (vs inception multi-kernel)
  * ~2.6x parameters (225k vs 86k)
  * same receiver front-end (STFT + z-score) — the RECEIVER is held fixed so
    the model, not the receiver, is the changed variable (disclosed)

It exposes the same forward(mag, ifr) / forward_wave(wave, frontend)
interface as the other victims, so the attack chain (attack_mask.waveform_pgd)
runs on it unchanged.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from waveforms import NUM_CLASSES


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        z = F.relu(self.bn1(self.conv1(x)))
        z = self.bn2(self.conv2(z))
        return F.relu(x + z)


class ResNetVictim(nn.Module):
    """2-channel early-fusion ResNet on the STFT streams."""

    def __init__(self, num_classes=NUM_CLASSES, width=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(2, width, 5, padding=2, bias=False),
            nn.BatchNorm2d(width), nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(ResBlock(width), ResBlock(width))
        self.trans1 = nn.Sequential(
            nn.Conv2d(width, width * 2, 1, bias=False),
            nn.BatchNorm2d(width * 2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.stage2 = nn.Sequential(ResBlock(width * 2), ResBlock(width * 2))
        self.trans2 = nn.Sequential(
            nn.Conv2d(width * 2, width * 4, 1, bias=False),
            nn.BatchNorm2d(width * 4), nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.stage3 = ResBlock(width * 4)
        self.head = nn.Linear(width * 4, num_classes)

    def forward(self, mag, ifr=None):
        x = torch.cat([mag, ifr], dim=1) if ifr is not None else mag
        x = self.stem(x)
        x = self.stage1(x)
        x = self.trans1(x)
        x = self.stage2(x)
        x = self.trans2(x)
        x = self.stage3(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.head(x)

    def forward_wave(self, wave, frontend):
        mag, ifr = frontend(wave)
        return self.forward(mag, ifr)
