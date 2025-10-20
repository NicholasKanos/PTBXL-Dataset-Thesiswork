import math
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from models.transformer import ECG_Transformer

# Metrics
from torchmetrics.classification import (
    MultilabelAUROC,
    MultilabelAveragePrecision,
    MultilabelF1Score,
)
from sklearn.metrics import f1_score as _sk_f1


# ---- Simple EMA of weights (helps lower/steady val_loss) --------------------
class _EMA:
    def __init__(self, model, decay: float = 0.999):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    def apply_shadow(self, model):
        self.backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.backup[n] = p.detach().clone()
                p.data.copy_(self.shadow[n].data)

    def restore(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.backup[n].data)
        self.backup = {}


class LitTransformer(pl.LightningModule):
    """
    Minimal surgical upgrades:
    - Label smoothing (tends to lower val BCE)
    - Optional pos_weight for imbalance (logs both weighted/unweighted losses)
    - EMA evaluation (steadier/lower val_loss)
    - Real metrics: macro AUROC, PR-AUC, F1@0.5 and F1@tuned thresholds
    - Saves per-class tuned thresholds in checkpoint buffer

    Optimizer remains Adam to keep your training script unchanged.
    """

    def __init__(
        self,
        model_args: dict,
        lr: float = 1e-3,
        label_smoothing: float = 0.05,  # helps reduce over-confidence -> lower val_loss
        use_ema: bool = True,  # evaluate with EMA weights
        use_pos_weight: bool = False,  # set True to enable class weighting
        compute_pos_weight_from_loader: bool = True,  # requires a DataModule
        pos_weight: torch.Tensor | None = None,  # or pass your own (num_classes,)
    ):
        super().__init__()
        self.model = ECG_Transformer(**model_args)
        self.lr = lr
        self.label_smoothing = float(label_smoothing)
        self.use_ema = bool(use_ema)
        self.use_pos_weight = bool(use_pos_weight)
        self.compute_pos_weight_from_loader = bool(compute_pos_weight_from_loader)
        self.pos_weight = pos_weight  # optionally provided

        # Discover classes for metrics
        self.num_classes = getattr(self.model, "num_classes", model_args.get("num_classes"))
        assert isinstance(self.num_classes, int) and self.num_classes > 0, "num_classes must be set"

        # Metrics
        self.train_auroc = MultilabelAUROC(num_labels=self.num_classes, average="macro")
        self.val_auroc = MultilabelAUROC(num_labels=self.num_classes, average="macro")
        self.val_ap = MultilabelAveragePrecision(num_labels=self.num_classes, average="macro")
        self.val_f1_default = MultilabelF1Score(num_labels=self.num_classes, average="macro", threshold=0.5)

        # Per-class threshold tuning buffer (saved in ckpt)
        self.register_buffer("per_class_thresh", torch.full((self.num_classes,), 0.5))
        self._val_probs, self._val_targs = [], []

        # EMA holder
        self._ema = None

    # ---------------- Base plumbing ----------------
    def forward(self, x):
        return self.model(x)

    def setup(self, stage=None):
        # Optionally estimate pos_weight from the DataModule's train loader (fast scan)
        if (
            stage == "fit"
            and self.use_pos_weight
            and self.pos_weight is None
            and self.compute_pos_weight_from_loader
            and self.trainer is not None
            and getattr(self.trainer, "datamodule", None) is not None
        ):
            dl = self.trainer.datamodule.train_dataloader()
            pos = torch.zeros(self.num_classes, dtype=torch.float64)
            neg = torch.zeros(self.num_classes, dtype=torch.float64)
            with torch.no_grad():
                for i, (_, y) in enumerate(dl):
                    y = y.float()
                    pos += y.sum(0).double()
                    neg += (1.0 - y).sum(0).double()
                    if i >= 255:  # cap for speed
                        break
            pos.clamp_(min=1.0)
            neg.clamp_(min=1.0)
            self.pos_weight = (neg / pos).float()

        if self.use_ema and self._ema is None:
            self._ema = _EMA(self.model, decay=0.999)

    # Label-smoothed BCE; can be weighted or unweighted
    def _bce(self, logits, targets, weighted: bool):
        if self.label_smoothing and self.label_smoothing > 0:
            eps = self.label_smoothing
            targets = targets * (1.0 - eps) + 0.5 * eps
        pw = (
            self.pos_weight.to(logits.device)
            if (weighted and self.use_pos_weight and self.pos_weight is not None)
            else None
        )
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)

    # ---------------- Training ----------------
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        # If you enable pos_weight, train with weighted BCE; otherwise unweighted (pure apples-to-apples)
        loss = self._bce(logits, y.float(), weighted=self.use_pos_weight)

        # Optional train AUROC (noisy but useful)
        with torch.no_grad():
            self.train_auroc.update(logits.sigmoid(), y.int())

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=x.size(0))
        return loss

    def on_train_epoch_end(self):
        self.log("train_macro_auroc", self.train_auroc.compute(), prog_bar=True)
        self.train_auroc.reset()

    # ---------------- Validation ----------------
    def on_validation_start(self):
        # evaluate with EMA weights applied (usually lowers val_loss a bit)
        if self.use_ema and self._ema is not None:
            self._ema.apply_shadow(self.model)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        y = y.float()

        # Always log UNWEIGHTED val_loss for comparability
        loss_unw = self._bce(logits, y, weighted=False)
        self.log("val_loss", loss_unw, prog_bar=True, on_epoch=True, batch_size=x.size(0))

        # If using pos_weight, also log the weighted objective (may be numerically higher)
        if self.use_pos_weight and self.pos_weight is not None:
            loss_w = self._bce(logits, y, weighted=True)
            self.log("val_loss_weighted", loss_w, on_epoch=True, batch_size=x.size(0))

        probs = logits.sigmoid()
        self.val_auroc.update(probs, y.int())
        self.val_ap.update(probs, y.int())
        self.val_f1_default.update(probs, y.int())

        # Collect for threshold tuning
        self._val_probs.append(probs.detach().cpu())
        self._val_targs.append(y.detach().cpu())
        return loss_unw

    def on_validation_epoch_end(self):
        # Macro metrics with default threshold
        self.log("val_macro_auroc", self.val_auroc.compute(), prog_bar=True)
        self.log("val_macro_ap", self.val_ap.compute(), prog_bar=True)
        self.log("val_macro_f1@0.5", self.val_f1_default.compute(), prog_bar=True)

        # Per-class threshold tuning on the validation set (saved into buffer)
        if self._val_probs:
            probs = torch.cat(self._val_probs, 0).numpy()
            targs = torch.cat(self._val_targs, 0).numpy().astype(int)
            ts = np.linspace(0.05, 0.95, 19)
            best_t = []
            for k in range(self.num_classes):
                yk = targs[:, k]
                pk = probs[:, k]
                if yk.max() == yk.min():  # degenerate class in this fold
                    best_t.append(float(self.per_class_thresh[k].item()))
                    continue
                scores = [_sk_f1(yk, (pk >= t).astype(int), average="binary", zero_division=0) for t in ts]
                best_t.append(float(ts[int(np.argmax(scores))]))
            best_t = torch.tensor(best_t, dtype=self.per_class_thresh.dtype, device=self.per_class_thresh.device)
            self.per_class_thresh.copy_(best_t)

            preds_tuned = (probs >= best_t.cpu().numpy()[None, :]).astype(int)
            macro_f1_tuned = _sk_f1(targs, preds_tuned, average="macro", zero_division=0)
            self.log("val_macro_f1@tuned_t", macro_f1_tuned, prog_bar=True)

        # reset accumulators
        self._val_probs.clear()
        self._val_targs.clear()
        self.val_auroc.reset()
        self.val_ap.reset()
        self.val_f1_default.reset()

    def on_validation_end(self):
        if self.use_ema and self._ema is not None:
            self._ema.restore(self.model)

    # Update EMA each optimizer step
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure, **kwargs):
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
        if self.use_ema and self._ema is not None:
            self._ema.update(self.model)

    # ---------------- Optimizer ----------------
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
