# Filename: src/models/components/lit_generic.py
# Updated LitGenericModel with BCEWithLogitsLoss, AdamW scheduler fixes,


import pytorch_lightning as pl
import torch
import torch.nn as nn

from torchmetrics.classification import MultilabelAccuracy


class LitGenericModel(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        pos_weight: torch.Tensor = None,   # shape [num_classes], optional
        loss_fn: nn.Module = None,         # if provided, overrides BCEWithLogitsLoss
        optimizer_class=torch.optim.AdamW, # AdamW by default
        use_scheduler: bool = True,
    ):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=["model", "loss_fn", "pos_weight"])

        if loss_fn is not None:
            self.loss_fn = loss_fn
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight) if pos_weight is not None else nn.BCEWithLogitsLoss()

        self.optimizer_class = optimizer_class
        self.use_scheduler = use_scheduler

        # Training-only accuracy (lazy initialized once labels are seen)
        self._train_acc = None
        self._num_labels = None

    def forward(self, x):
        # If the underlying model is Conv1d-based and input is (B, T, C),
        # permute to (B, C, T) before forwarding.
        if x.dim() == 3:
            first_conv = next((m for m in self.model.modules() if isinstance(m, nn.Conv1d)), None)
            if first_conv is not None:
                # If last dim matches in_channels and middle doesn't, we likely have (B, T, C)
                if x.shape[-1] == first_conv.in_channels and x.shape[1] != first_conv.in_channels:
                    x = x.permute(0, 2, 1)  # (B, T, C) -> (B, C, T)
        return self.model(x)

    def _ensure_train_metrics(self, y: torch.Tensor, device: torch.device):
        if self._train_acc is None:
            self._num_labels = int(y.shape[1])
            self._train_acc = MultilabelAccuracy(num_labels=self._num_labels, average="macro").to(device)

    def _step(self, batch, stage: str):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log(
            f"{stage}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=(stage in ("train", "val")),
            batch_size=y.size(0),
        )
        return loss, logits, y

    def training_step(self, batch, batch_idx):
        loss, logits, y = self._step(batch, "train")
        # Update training-only accuracy (ensure metric device matches tensors)
        probs = torch.sigmoid(logits)
        self._ensure_train_metrics(y, device=probs.device)
        target = y.to(probs.device).int()
        self._train_acc.update(probs, target)
        return loss

    def on_train_epoch_end(self):
        if self._train_acc is not None:
            self.log("train/acc", self._train_acc.compute(), prog_bar=True)
            self._train_acc.reset()

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")  # loss-only on val

    def test_step(self, batch, batch_idx):
        self._step(batch, "test")  # loss-only on test

    def configure_optimizers(self):
        opt = self.optimizer_class(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        if not self.use_scheduler:
            return opt
        # Remove unsupported 'verbose' for compatibility
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=0.7,
            patience=5,
            threshold=1e-4,
            threshold_mode="rel",
            cooldown=1,
            min_lr=1e-6,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sch,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
