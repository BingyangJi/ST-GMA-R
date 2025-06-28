import torch
import torch.nn as nn
from typing import List


def conv_bn_act(in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, activation=nn.ReLU):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
              nn.BatchNorm2d(out_channels)]
    if activation:
        layers.append(activation(inplace=True))
    return nn.Sequential(*layers)


class ScaleBranch(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size, stride=1, groups=1):
        super().__init__()
        self.branch = nn.Sequential(
            conv_bn_act(in_channels, mid_channels, kernel_size=1),
            conv_bn_act(mid_channels, mid_channels, kernel_size, stride, padding=kernel_size//2, groups=groups),
            conv_bn_act(mid_channels, out_channels, kernel_size=1, activation=None)
        )

    def forward(self, x):
        return self.branch(x)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, planes, stride=1, scales=2, groups=1, width_per_group=4, downsample=None):
        super().__init__()
        if planes % scales != 0:
            raise ValueError('"planes" must be divisible by "scales"')

        mid_channels = int(planes * (width_per_group / 128.)) * groups
        out_channels = planes * self.expansion // scales
        self.downsample = downsample

        kernels = [3, 5]
        self.branches = nn.ModuleList([
            ScaleBranch(
                in_channels // scales,
                mid_channels,
                out_channels,
                kernel,
                stride,
                groups
            ) for kernel in kernels[:scales]
        ])

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.downsample(x) if self.downsample else x
        parts = torch.chunk(x, len(self.branches), dim=1)
        out = torch.cat([branch(part) for branch, part in zip(self.branches, parts)], dim=1)
        return self.relu(out + identity)


class ResFusionNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        depth_blocks: List[int] = [3, 4, 6, 3],
        num_classes: int = 2,
        include_top: bool = True,
        groups: int = 16,
        width_per_group: int = 4
    ):
        super().__init__()
        self.include_top = include_top
        self.stem = nn.Sequential(
            conv_bn_act(input_dim, 64, kernel_size=1),
            nn.MaxPool2d(3, stride=1, padding=1)
        )

        self.in_channels = 64
        self.layers = nn.Sequential(*[
            self._make_stage(planes, blocks, stride)
            for (planes, blocks, stride) in zip([64, 128, 256, 512], depth_blocks, [1, 2, 2, 2])
        ])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if include_top:
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512 * Bottleneck.expansion, num_classes),
                nn.Sigmoid()
            )

    def _make_stage(self, planes, blocks, stride):
        downsample = None
        out_channels = planes * Bottleneck.expansion
        if stride != 1 or self.in_channels != out_channels:
            downsample = conv_bn_act(self.in_channels, out_channels, kernel_size=1, stride=stride, activation=None)

        layers = [
            Bottleneck(
                self.in_channels,
                planes,
                stride,
                groups=self.width_per_group if hasattr(self, 'width_per_group') else 1,
                width_per_group=self.width_per_group,
                downsample=downsample
            )
        ]
        self.in_channels = out_channels
        layers += [
            Bottleneck(self.in_channels, planes, groups=self.width_per_group, width_per_group=self.width_per_group)
            for _ in range(1, blocks)
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layers(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x) if self.include_top else x


def build_resfusionnet(input_dim, num_classes=2, include_top=True):
    return ResFusionNet(input_dim, include_top=include_top, num_classes=num_classes)
