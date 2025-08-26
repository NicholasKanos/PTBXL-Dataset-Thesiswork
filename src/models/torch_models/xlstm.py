# models/xlstm.py

import torch
import torch.nn as nn

class xLSTMECG(nn.Module):
    def __init__(self, input_channels=12, num_classes=10, hidden_size=128, num_layers=2):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # x: [B, C, T]
        x = self.relu(self.bn1(self.conv1(x)))  # [B, 32, T]
        x = self.relu(self.bn2(self.conv2(x)))  # [B, 64, T]
        x = x.permute(0, 2, 1)                  # [B, T, 64]
        x, _ = self.lstm(x)                     # [B, T, 2*H]
        x = x[:, -1, :]                         # [B, 2*H]
        return self.fc(x)                       # [B, num_classes]

def build_model(input_shape, num_classes, hidden_size=128, num_layers=2, **kwargs):
    """
    Constructs a CNN + LSTM model.
    Args:
        input_shape (tuple): (channels, sequence_length)
        num_classes (int): output class count
        hidden_size (int): hidden units in LSTM
        num_layers (int): number of LSTM layers
    """
    input_channels, _ = input_shape
    return xLSTMECG(input_channels=input_channels,
                    num_classes=num_classes,
                    hidden_size=hidden_size,
                    num_layers=num_layers)
