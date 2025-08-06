# models/cnn_transformer.py

import torch
import torch.nn as nn


class ECG_Transformer(nn.Module):
    def __init__(self, seq_len=5000, num_features=12, d_model=32, nhead=2, num_layers=2, num_classes=8):
        super(ECG_Transformer, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=d_model, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, C, T) → (B, Channels, SeqLen)
        x = self.cnn(x)  # (B, d_model, reduced_seq_len)
        x = x.permute(0, 2, 1)  # (B, SeqLen, d_model)
        x = self.transformer_encoder(x)
        x = x.permute(0, 2, 1)  # (B, d_model, SeqLen)
        x = self.global_avg_pool(x).squeeze(-1)  # (B, d_model)
        return self.fc(x)  # (B, num_classes)


def build_model(input_shape, num_classes, d_model=32, nhead=2, num_layers=2, **kwargs):
    """
    Constructs a CNN + Transformer model.
    Args:
        input_shape (tuple): (channels, sequence_length)
        num_classes (int): output class count
        d_model (int): transformer dimension
        nhead (int): number of attention heads
        num_layers (int): transformer encoder layers
    """
    channels, seq_len = input_shape
    return ECG_Transformer(seq_len=seq_len,
                           num_features=channels,
                           d_model=d_model,
                           nhead=nhead,
                           num_layers=num_layers,
                           num_classes=num_classes)
