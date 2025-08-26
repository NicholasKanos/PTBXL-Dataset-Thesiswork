# models/cnn_transformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvStem1D(nn.Module):
    """Aggressive but stable downsampling + morphology capture."""
    def __init__(self, in_ch: int, hid: int, out_ch: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, hid, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(hid),
            nn.GELU(),
            nn.Conv1d(hid, hid, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(hid),
            nn.GELU(),
            nn.Conv1d(hid, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
        )  # overall /8 in time

    def forward(self, x):  # x: (B, C, T)
        return self.stem(x)  # (B, out_ch, T/8)


class CNNTransformer(nn.Module):
    def __init__(
        self,
        seq_len=5000,
        num_features=12,
        d_model=128,      # bumped default width
        nhead=4,          # must divide d_model
        num_layers=4,     # a bit deeper
        num_classes=8,
        stem_hidden=64,
        dropout=0.1,
        emb_dropout=0.1,
        pool="avg",       # "avg" or "cls"
    ):
        super().__init__()

        # --- 1) Conv stem with stride downsampling (/8) ---
        self.stem = ConvStem1D(in_ch=num_features, hid=stem_hidden, out_ch=d_model)

        # --- 2) 1x1 conv to ensure exact d_model alignment (no-op if already d_model) ---
        self.proj = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)

        # --- 3) Learnable positional embedding (created lazily per sequence length) ---
        self.pos_emb = None
        self.emb_dropout = nn.Dropout(emb_dropout)

        # Optional CLS token if you prefer (disabled by default with pool="avg")
        self.use_cls = (pool == "cls")
        if self.use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.cls_token, std=0.02)

        # --- 4) Transformer encoder (norm_first is nicer for stability) ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- 5) Head ---
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        # Init head
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def _pos_embed(self, x_tokens):
        """Create/resize learnable pos emb to current length L."""
        B, L, D = x_tokens.shape
        if (self.pos_emb is None) or (self.pos_emb.shape[1] != L):
            # fresh parameter sized to current token length
            pe = torch.zeros(1, L, D, device=x_tokens.device)
            nn.init.normal_(pe, std=0.01)
            self.pos_emb = nn.Parameter(pe)
        return x_tokens + self.pos_emb

    def forward(self, x):
        # Expect (B, C, T). If you feed (B, T, C) from elsewhere, permute before calling.
        # --- conv stem ---
        x = self.stem(x)           # (B, d_model, T')
        x = self.proj(x)           # (B, d_model, T')
        x = x.transpose(1, 2)      # (B, T', d_model) = tokens

        # --- optional CLS token ---
        if self.use_cls:
            cls = self.cls_token.expand(x.size(0), -1, -1)  # (B, 1, D)
            x = torch.cat([cls, x], dim=1)                  # (B, 1+T', D)

        # --- learnable pos emb + dropout ---
        x = self._pos_embed(x)
        x = self.emb_dropout(x)

        # --- encoder ---
        x = self.transformer_encoder(x)   # (B, L, D)

        # --- pooling ---
        if self.use_cls:
            x = x[:, 0]                   # (B, D)
        else:
            x = x.mean(dim=1)             # global avg pool (B, D)

        x = self.norm(x)
        x = self.head(x)                  # (B, num_classes)
        return x


def build_model(input_shape, num_classes, d_model=128, nhead=4, num_layers=4, **kwargs):
    """
    Constructs a stronger CNN + Transformer model.
    input_shape: (channels, sequence_length)
    """
    channels, seq_len = input_shape
    return CNNTransformer(
        seq_len=seq_len,
        num_features=channels,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        num_classes=num_classes,
        **kwargs
    )
