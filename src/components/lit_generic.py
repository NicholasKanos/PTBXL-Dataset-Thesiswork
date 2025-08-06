import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

class LitGenericModel(pl.LightningModule):
    def __init__(self, model: nn.Module, loss_fn=None, optimizer_class=torch.optim.Adam, lr=1e-3, metrics=None):
        """
        Args:
            model (nn.Module): Any torch model (from torch_models)
            loss_fn (callable): Loss function (default: nn.CrossEntropyLoss)
            optimizer_class: Optimizer class (default: Adam)
            lr (float): Learning rate
            metrics (dict): Dict of metric_name: metric_fn(batch, preds, targets)
        """
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()
        self.optimizer_class = optimizer_class
        self.lr = lr
        self.metrics = metrics or {}

    def forward(self, x):
        return self.model(x)

    def step(self, batch, batch_idx, stage):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log(f'{stage}_loss', loss, on_step=False, on_epoch=True)
        for name, metric_fn in self.metrics.items():
            metric_value = metric_fn(logits, y)
            self.log(f'{stage}_{name}', metric_value, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        self.step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        self.step(batch, batch_idx, 'test')

    def configure_optimizers(self):
        return self.optimizer_class(self.parameters(), lr=self.lr)