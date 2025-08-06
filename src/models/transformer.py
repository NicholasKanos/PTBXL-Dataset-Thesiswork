# models/transformer.py

import torch
import torch.nn as nn

class ECG_Transformer(nn.Module):
    def __init__(self, seq_len=5000, num_features=12, d_model=32, nhead=2, num_layers=2, num_classes=8):
        super(ECG_Transformer, self).__init__()
        self.input_linear = nn.Linear(num_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (B, SeqLen, Channels)
        x = self.input_linear(x)                # → (B, SeqLen, d_model)
        x = self.transformer_encoder(x)         # → (B, SeqLen, d_model)
        x = x.permute(0, 2, 1)                  # → (B, d_model, SeqLen)
        x = self.global_avg_pool(x).squeeze(-1) # → (B, d_model)
        return self.fc(x)                       # → (B, num_classes)

def build_model(input_shape, num_classes, d_model=32, nhead=2, num_layers=2, **kwargs):
    """
    Constructs the ECG Transformer model.
    Args:
        input_shape (tuple): (channels, sequence_length)
        num_classes (int): number of output classes
        d_model (int): transformer model dimension
        nhead (int): number of attention heads
        num_layers (int): number of encoder layers
    """
    num_features, seq_len = input_shape
    return ECG_Transformer(seq_len=seq_len,
                           num_features=num_features,
                           d_model=d_model,
                           nhead=nhead,
                           num_layers=num_layers,
                           num_classes=num_classes)
