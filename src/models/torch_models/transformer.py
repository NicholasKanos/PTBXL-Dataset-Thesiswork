import torch
import torch.nn as nn

class ECG_Transformer(nn.Module):
    """Transformer for multilead ECG.

    Expects input shaped (B, T, C):
      B = batch size, T = time steps, C = channels/leads (e.g., 12)
    Returns logits shaped (B, num_classes) suitable for BCEWithLogitsLoss.
    """
    def __init__(
        self,
        seq_len: int = 5000,
        num_features: int = 12,
        d_model: int = 32,
        nhead: int = 2,
        num_layers: int = 2,
        num_classes: int = 8,
    ) -> None:
        super().__init__()
        self.input_linear = nn.Linear(num_features, d_model)

        # Learned positional embedding so the encoder has temporal information
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        Args:
            x: Tensor of shape (B, T, C)
        Returns:
            Tensor of shape (B, num_classes) — raw logits
        """
        # (B, T, C) -> (B, T, d_model)
        x = self.input_linear(x)
        # Add positional info (trim if sequence shorter than seq_len)
        x = x + self.pos_embed[:, : x.size(1)]
        # (B, T, d_model)
        x = self.transformer_encoder(x)
        # (B, d_model, T)
        x = x.transpose(1, 2)
        # (B, d_model)
        x = self.global_avg_pool(x).squeeze(-1)
        # (B, num_classes)
        return self.fc(x)


def build_model(
    input_shape: tuple,
    num_classes: int,
    d_model: int = 32,
    nhead: int = 2,
    num_layers: int = 2,
    **kwargs,
) -> nn.Module:
    """Factory to match your selector signature.

    Args:
        input_shape: (channels, sequence_length)
        num_classes: number of output classes
    """
    num_features, seq_len = input_shape
    return ECG_Transformer(
        seq_len=seq_len,
        num_features=num_features,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        num_classes=num_classes,
    )
