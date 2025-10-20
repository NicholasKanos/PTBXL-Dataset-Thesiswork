import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import MultilabelAccuracy
# NEW: add richer multilabel metrics
from torchmetrics.classification import MultilabelAUROC, MultilabelAveragePrecision


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
        label_smoothing: float = 0.0,      # optional label smoothing (0.0 disables)
        class_names=None                   # optional; useful for logs
    ):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=["model", "loss_fn", "pos_weight"])

        # CHANGED: keep a reference to a *buffer* pos_weight so it moves with .to(device)
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.float())
        else:
            self.pos_weight = None

        # CHANGED: do NOT prebuild BCEWithLogitsLoss with a CPU pos_weight tensor
        
        self._external_loss = loss_fn  

        self.optimizer_class = optimizer_class
        self.use_scheduler = use_scheduler
        self.label_smoothing = float(label_smoothing)
        self.class_names = class_names

        # Metrics (lazy init after seeing label shape)
        self._train_acc = None
        self._val_acc = None
        self._num_labels = None
        # NEW: richer multilabel metrics 
        self._val_auroc = None
        self._val_auprc = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            first_conv = next((m for m in self.model.modules() if isinstance(m, nn.Conv1d)), None)
            if first_conv is not None:
                if x.shape[-1] == first_conv.in_channels and x.shape[1] != first_conv.in_channels:
                    x = x.permute(0, 2, 1)  # (B, T, C) -> (B, C, T)
        return self.model(x)

    def _ensure_train_metrics(self, y: torch.Tensor, device: torch.device):
        if self._train_acc is None:
            self._num_labels = int(y.shape[1])
            self._train_acc = MultilabelAccuracy(
                num_labels=self._num_labels, average="macro", threshold=0.5
            ).to(device)

    def _ensure_val_metrics(self, y: torch.Tensor, device: torch.device):
        if self._val_acc is None:
            self._num_labels = int(y.shape[1])
            self._val_acc = MultilabelAccuracy(
                num_labels=self._num_labels, average="macro", threshold=0.5
            ).to(device)
        # NEW: init AUROC/AUPRC once we know C
        if self._val_auroc is None:
            self._val_auroc = MultilabelAUROC(
                num_labels=self._num_labels, average="macro"
            ).to(device)
        if self._val_auprc is None:
            self._val_auprc = MultilabelAveragePrecision(
                num_labels=self._num_labels, average="macro"
            ).to(device)

    def _apply_label_smoothing(self, y: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing <= 0.0:
            return y
        eps = self.label_smoothing
        return y * (1.0 - eps) + 0.5 * eps

    # NEW: build BCEWithLogitsLoss 
    def _bce_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self._external_loss is not None:
            return self._external_loss(logits, targets)
        # If pos_weight is set, ensure it’s on the same device
        if self.pos_weight is not None:
            return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight)
        return F.binary_cross_entropy_with_logits(logits, targets)

    def _step(self, batch, stage: str):
        x, y = batch
        logits = self(x)

        y_for_loss = self._apply_label_smoothing(y)
        loss = self._bce_loss(logits, y_for_loss)

        self.log(
            f"{stage}_loss", loss,
            on_step=False, on_epoch=True,
            prog_bar=(stage in ("train", "val")),
            batch_size=y.size(0),
        )
        return loss, logits, y

    # ----- Training -----
    def training_step(self, batch, batch_idx):
        loss, logits, y = self._step(batch, "train")
        probs = torch.sigmoid(logits)
        self._ensure_train_metrics(y, device=probs.device)
        self._train_acc.update(probs, y.int())
        return loss

    def on_train_epoch_end(self):
        if self._train_acc is not None:
            self.log("train/acc", self._train_acc.compute(), prog_bar=True)
            self._train_acc.reset()

    # ----- Validation: unweighted BCE + richer metrics -----
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        # Keep your unweighted BCE for early stopping
        y_for_loss = self._apply_label_smoothing(y)
        loss_unw = F.binary_cross_entropy_with_logits(logits, y_for_loss)
        self.log("val_loss", loss_unw, prog_bar=True, on_epoch=True, batch_size=y.size(0))

        # Also log the weighted objective (informational only)
        if self.pos_weight is not None:
            loss_w = F.binary_cross_entropy_with_logits(logits, y_for_loss, pos_weight=self.pos_weight)
            self.log("val_loss_weighted", loss_w, on_epoch=True, batch_size=y.size(0))

        # Metrics
        probs = torch.sigmoid(logits)
        self._ensure_val_metrics(y, device=probs.device)
        target = y.to(probs.device).int()
        self._val_acc.update(probs, target)
        # NEW: AUROC / AUPRC (macro)
        self._val_auroc.update(probs, target)
        self._val_auprc.update(probs, target)

    def on_validation_epoch_end(self):
        if self._val_acc is not None:
            self.log("val_acc", self._val_acc.compute(), prog_bar=True, on_epoch=True)
            self._val_acc.reset()
        # NEW: finalize & log AUROC/AUPRC
        if self._val_auroc is not None:
            self.log("val_macro_auroc", self._val_auroc.compute(), prog_bar=True, on_epoch=True)
            self._val_auroc.reset()
        if self._val_auprc is not None:
            self.log("val_macro_auprc", self._val_auprc.compute(), prog_bar=True, on_epoch=True)
            self._val_auprc.reset()

    # ----- Test: loss-only -----
    def test_step(self, batch, batch_idx):
        self._step(batch, "test")

    def configure_optimizers(self):
        opt = self.optimizer_class(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        if not self.use_scheduler:
            return opt

        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=0.5,
            patience=3,
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
