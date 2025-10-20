# models/transformer.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Small helpers ------------------------------------------------------------

class SE1D(nn.Module):
    """Squeeze-Excite for 1D feature maps: emphasize informative leads early."""
    def __init__(self, ch: int, r: int = 8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, ch // r),
            nn.GELU(),
            nn.Linear(ch // r, ch),
            nn.Sigmoid(),
        )

    def forward(self, x):           # x: (B, C, T)
        b, c, _ = x.shape
        w = self.pool(x).view(b, c) # (B, C)
        w = self.fc(w).view(b, c, 1)
        return x * w


class PatchEmbed(nn.Module):
    """
    Learn patch tokens using a lightweight conv stem.
    Input:  (B, C, T)
    Output: (B, L, d_model) where L ~ T/stride
    """
    def __init__(self, in_ch=12, d_model=192, patch=32, stride=16, use_se=True):
        super().__init__()
        mid = max(64, d_model // 2)
        layers = [
            nn.Conv1d(in_ch, mid, kernel_size=patch, stride=stride, padding=patch // 2, bias=False),
            nn.BatchNorm1d(mid),
            nn.GELU(),
            nn.Conv1d(mid, d_model, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        ]
        self.proj = nn.Sequential(*layers)
        self.se = SE1D(d_model, r=8) if use_se else nn.Identity()

    def forward(self, x):  # (B, C, T)
        x = self.proj(x)   # (B, d_model, L)
        x = self.se(x)
        return x.transpose(1, 2)  # (B, L, d_model)


class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positions with a learnable global scale (often helps).
    """
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d)
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):  # x: (B, L, d)
        L = x.size(1)
        return x + self.scale * self.pe[:, :L, :]


# --- Model --------------------------------------------------------------------

class ECG_Transformer(nn.Module):
    """
    Transformer for multilead ECG with conv patch embedding and CLS token.

    Accepts (B, T, C) or (B, C, T).
    Returns logits (B, num_classes) suitable for BCEWithLogitsLoss.
    """
    def __init__(
        self,
        seq_len: int = 5000,        # raw sequence length at 500 Hz
        num_features: int = 12,     # ECG leads
        num_classes: int = 8,
        # Transformer width/depth
        d_model: int = 320,
        nhead: int = 8,
        num_layers: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.30,
        # Patch embedding
        patch: int = 32,
        stride: int = 16,
        use_se: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.d_model = d_model

        # 1) Patchify with conv stem (reduces token length & encodes locality)
        self.patch = PatchEmbed(in_ch=num_features, d_model=d_model, patch=patch, stride=stride, use_se=use_se)

        # Approx token length after patching (for buffer init and comments only)
        self.estimated_tokens = math.ceil(seq_len / stride)

        # 2) CLS token
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))

        # 3) Positional encoding (sinusoidal + learnable scale)
        # (max_len generously set; you can trim based on seq_len//stride + 1 for CLS)
        self.posenc = SinusoidalPositionalEncoding(d_model=d_model, max_len=8192)

        # 4) Encoder (PyTorch's stock encoder; windowed attention would require a custom block)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=int(d_model * max(1.0, 4 * mlp_ratio)),  # keep FFN reasonably wide
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # typically better for transformers on signals
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 5) Head: concat(CLS, mean) → LN → MLP → logits
        head_in = 2 * d_model
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

        # Init cls with small noise (helps early learning)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def _ensure_bct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize input to (B, C, T). Accepts (B, T, C) or (B, C, T).
        """
        assert x.ndim == 3, f"Expected 3D tensor, got {x.shape}"
        B, A, Bdim = x.shape
        # Heuristic: if the middle dim is very large (≈ time), treat as (B, T, C)
        # but we’ll be explicit: if last dim ≤ 16, we assume it's channels.
        if x.shape[-1] <= 16:
            # (B, T, C) -> (B, C, T)
            x = x.transpose(1, 2)
        # else it's already (B, C, T)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C) or (B, C, T)
        Returns:
            logits: (B, num_classes)
        """
        # Normalize to (B, C, T)
        x = self._ensure_bct(x)

        # Patchify: (B, C, T) -> (B, L, d)
        x = self.patch(x)

        # Prepend CLS: (B, 1+L, d)
        B = x.size(0)
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        # Add positions
        x = self.posenc(x)

        # Encode
        x = self.transformer_encoder(x)  # (B, 1+L, d)

        # Pool: concat(CLS, mean)
        cls_out = x[:, 0]            # (B, d)
        mean_out = x[:, 1:].mean(dim=1)  # (B, d)
        feat = torch.cat([cls_out, mean_out], dim=1)  # (B, 2d)

        # Head
        logits = self.head(feat)     # (B, num_classes)
        return logits


# Factory to match your selector signature
def build_model(
    input_shape: tuple,
    num_classes: int,
    d_model: int = 160,
    nhead: int = 5,
    num_layers: int = 4,
    patch: int = 32,
    stride: int = 16,
    dropout: float = 0.30,
    use_se: bool = True,
    **kwargs,
) -> nn.Module:
    """
    Args:
        input_shape: (channels, sequence_length)
        num_classes: number of output classes
    """
    num_features, seq_len = input_shape
    return ECG_Transformer(
        seq_len=seq_len,
        num_features=num_features,
        num_classes=num_classes,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        patch=patch,
        stride=stride,
        dropout=dropout,
        use_se=use_se,
        **kwargs,
    )