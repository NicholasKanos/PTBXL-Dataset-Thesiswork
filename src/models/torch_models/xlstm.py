# CNN → Downsample → BiLSTM → Attention Pooling → Linear
# Returns raw logits suitable for BCEWithLogitsLoss (multi-label)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.depth = nn.Conv1d(
            in_ch, in_ch, kernel_size,
            stride=stride, padding=padding, groups=in_ch, bias=False
        )
        self.point = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depth(x)
        x = self.point(x)
        x = self.bn(x)
        return self.act(x)


class SEBlock(nn.Module):
    def __init__(self, ch, reduction=8):
        super().__init__()
        self.fc1 = nn.Conv1d(ch, ch // reduction, kernel_size=1)
        self.fc2 = nn.Conv1d(ch // reduction, ch, kernel_size=1)

    def forward(self, x):
        w = x.mean(dim=-1, keepdim=True)
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w


class ResBlock(nn.Module):
    def __init__(self, ch, stride=1, kernel_size=5, use_se=True, p_dropout=0.1):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv1d(
            ch, ch, kernel_size=kernel_size, stride=stride
        )
        self.conv2 = DepthwiseSeparableConv1d(
            ch, ch, kernel_size=kernel_size, stride=1
        )
        self.se = SEBlock(ch) if use_se else nn.Identity()
        self.drop = nn.Dropout(p_dropout)
        # FIX: make identity path halve with ceil on odd lengths to match conv path
        self.down = (
            nn.AvgPool1d(kernel_size=2, stride=2, ceil_mode=True)
            if stride == 2 else nn.Identity()
        )

    def forward(self, x):
        identity = self.down(x) if not isinstance(self.down, nn.Identity) else x
        out = self.conv1(x)
        out = self.drop(out)
        out = self.conv2(out)
        out = self.se(out)

        # FIX: align temporal length if still off by one before residual add
        if out.size(-1) != identity.size(-1):
            T = min(out.size(-1), identity.size(-1))
            out = out[..., :T]
            identity = identity[..., :T]

        return F.relu(out + identity)


class AttnPool1D(nn.Module):
    """Single-head additive attention pooling over time."""
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):  # x: [B, T, D]
        scores = self.proj(x).squeeze(-1)          # [B, T]
        weights = torch.softmax(scores, dim=1)     # [B, T]
        pooled = torch.einsum('bt, btd -> bd', weights, x)
        return pooled, weights


class xLSTMECG_Improved(nn.Module):
    def __init__(
        self,
        input_channels=12,
        num_classes=10,
        lstm_hidden=256,
        lstm_layers=2,
        cnn_width=128,
        down_blocks=(2, 2, 2),  # number of ResBlocks per stage
        p_dropout=0.1
    ):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(
                input_channels, cnn_width, kernel_size=7,
                stride=2, padding=3, bias=False
            ),
            nn.BatchNorm1d(cnn_width),
            nn.ReLU(inplace=True),
        )

        # Stages with progressive downsampling (reduce T by 2 each stage)
        stages = []
        ch = cnn_width
        for _, n_blocks in enumerate(down_blocks):
            # first block in stage downsamples
            blocks = [ResBlock(ch, stride=2, p_dropout=p_dropout)]
            # remaining keep same length
            for _ in range(n_blocks - 1):
                blocks.append(ResBlock(ch, stride=1, p_dropout=p_dropout))
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.ModuleList(stages)

        # Project to LSTM input size
        self.to_lstm = nn.Conv1d(ch, lstm_hidden, kernel_size=1, bias=False)
        self.ln = nn.LayerNorm(lstm_hidden)

        self.lstm = nn.LSTM(
            input_size=lstm_hidden,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=p_dropout if lstm_layers > 1 else 0.0,
        )

        self.pool = AttnPool1D(dim=2 * lstm_hidden, hidden=lstm_hidden)

        self.head = nn.Sequential(
            nn.Dropout(p_dropout),
            nn.Linear(2 * lstm_hidden, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):  # x: [B, C, T]
        x = self.stem(x)                 # [B, W, T/2]
        for stage in self.stages:
            x = stage(x)                 # progressively downsample T
        x = self.to_lstm(x)              # [B, H, T_reduced]
        x = x.permute(0, 2, 1)           # [B, T_reduced, H]
        x = self.ln(x)
        x, _ = self.lstm(x)              # [B, T_reduced, 2H]
        x, _ = self.pool(x)              # [B, 2H]
        return self.head(x)              # logits [B, num_classes]


def build_model(input_shape, num_classes, lstm_hidden=256, lstm_layers=2, **kwargs):
    """
    Constructs the improved CNN + BiLSTM with attention pooling.

    Args:
        input_shape (tuple): (channels, sequence_length)
        num_classes (int): number of output labels (multi-label OK)
        lstm_hidden (int): hidden width for LSTM and pre-proj
        lstm_layers (int): number of LSTM layers
    """
    input_channels, _ = input_shape
    return xLSTMECG_Improved(
        input_channels=input_channels,
        num_classes=num_classes,
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
        **kwargs,
    )
