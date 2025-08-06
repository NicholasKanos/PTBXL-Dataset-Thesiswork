# models/xresnet1d.py

import torch
import torch.nn as nn


class ConvLayer(nn.Sequential):
    def __init__(self, ni, nf, ks=3, stride=1, padding=None, bias=None, act_cls=nn.ReLU):
        if padding is None:
            padding = (ks - 1) // 2
        conv = nn.Conv1d(ni, nf, kernel_size=ks, stride=stride, padding=padding, bias=bias or False)
        bn = nn.BatchNorm1d(nf)
        layers = [conv, bn]
        if act_cls is not None:
            layers.append(act_cls())
        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, ni, nf, stride=1):
        super().__init__()
        self.convs = nn.Sequential(
            ConvLayer(ni, nf, stride=stride),
            ConvLayer(nf, nf, act_cls=None)
        )
        self.idconv = nn.Identity() if ni == nf and stride == 1 else ConvLayer(ni, nf, ks=1, stride=stride,
                                                                               act_cls=None)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.convs(x) + self.idconv(x))


class XResNet1D(nn.Sequential):
    def __init__(self, layers, input_channels=12, num_classes=10):
        block_sizes = [64, 128, 256, 512]
        stem = [ConvLayer(input_channels, 64, ks=7, stride=2),
                nn.MaxPool1d(kernel_size=3, stride=2, padding=1)]

        blocks = []
        ni = 64
        for i, n_blocks in enumerate(layers):
            nf = block_sizes[i]
            for j in range(n_blocks):
                stride = 2 if j == 0 and i != 0 else 1
                blocks.append(ResBlock(ni, nf, stride=stride))
                ni = nf

        head = [nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(ni, num_classes)]
        super().__init__(*stem, *blocks, *head)


def build_model(input_shape, num_classes, layers=[2, 2, 2, 2], **kwargs):
    """
    Constructs an XResNet1D model.
    Args:
        input_shape (tuple): (channels, sequence_length)
        num_classes (int): number of output classes
        layers (list): list indicating number of blocks per stage
    """
    input_channels = input_shape[0]
    return XResNet1D(layers=layers, input_channels=input_channels, num_classes=num_classes)
