import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import MultilabelAUROC, MultilabelAveragePrecision

try:
    from models.xlstm_improved import xLSTMECG_Improved as xLSTMECG
except Exception:
    from models.xlstm import xLSTMECG


class LitxLSTM(pl.LightningModule):
    def __init__(
        self,
        model_args: dict,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        warmup_pct: float = 0.05,
        use_adamw: bool = True,
        pos_weight: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["pos_weight"])  # avoids storing tensors in hparams

        # Build backbone
        self.model = xLSTMECG(**model_args)

        # --- Robust num_labels inference ---
        if "num_classes" in model_args and isinstance(model_args["num_classes"], int):
            self.num_labels = model_args["num_classes"]
        else:
            last_linear = None
            if hasattr(self.model, "fc") and isinstance(self.model.fc, nn.Linear):
                last_linear = self.model.fc
            elif hasattr(self.model, "head"):
                if isinstance(self.model.head, nn.Linear):
                    last_linear = self.model.head
                elif isinstance(self.model.head, nn.Sequential):
                    for m in reversed(self.model.head):
                        if isinstance(m, nn.Linear):
                            last_linear = m
                            break
            if last_linear is None:
                raise ValueError("Could not infer num_labels; please include 'num_classes' in model_args.")
            self.num_labels = last_linear.out_features

        # Register pos_weight as buffer (moves with device & checkpointed)
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.float())
        else:
            self.pos_weight = None

        # Metrics (macro over labels)
        self.auroc = MultilabelAUROC(num_labels=self.num_labels, average="macro")
        self.auprc = MultilabelAveragePrecision(num_labels=self.num_labels, average="macro")

        # Optim settings
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_pct = warmup_pct
        self.use_adamw = use_adamw

    def forward(self, x):
        return self.model(x)

    def _bce_loss(self, logits, targets):
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self._bce_loss(logits, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self._bce_loss(logits, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        probs = torch.sigmoid(logits)
        self.auroc.update(probs, y.int())
        self.auprc.update(probs, y.int())
        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        auroc = self.auroc.compute()
        auprc = self.auprc.compute()
        self.log("val_macro_auroc", auroc, prog_bar=True)
        self.log("val_macro_auprc", auprc, prog_bar=True)
        self.auroc.reset()
        self.auprc.reset()

    # Optional: log LR each step (nice for W&B/TB)
    def on_train_batch_end(self, *args, **kwargs):
        if self.trainer is not None and self.trainer.optimizers:
            opt = self.trainer.optimizers[0]
            if opt.param_groups:
                self.log("lr", opt.param_groups[0]["lr"], on_step=True, on_epoch=False, prog_bar=False)

    # ---- Optimizer & Scheduler (Cosine decay w/ warmup) ----
    def configure_optimizers(self):
        if self.use_adamw:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        total_steps = self._get_total_steps()
        warmup_steps = max(1, int(self.warmup_pct * total_steps))

        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def _get_total_steps(self):
        # Works in most PL setups after trainer is attached
        if self.trainer is None or self.trainer.estimated_stepping_batches is None:
            return 1000  # fallback
        return int(self.trainer.estimated_stepping_batches)
