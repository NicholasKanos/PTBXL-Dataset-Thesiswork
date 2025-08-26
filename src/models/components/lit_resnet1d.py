import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from models.resnet1d import ResNet1d

class LitResNet1D(pl.LightningModule):
    def __init__(self, model_args, lr=1e-3):
        super().__init__()
        self.model = ResNet1d(**model_args)
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
