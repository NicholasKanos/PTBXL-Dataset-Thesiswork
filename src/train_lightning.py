# Filename: src/train_lightning.py (W&B-enabled)
# Purpose : Train ECG model with BCEWithLogitsLoss(+pos_weight), AdamW, proper checkpointing & W&B logging

import argparse
import ast
import os
import pandas as pd
import torch
import wfdb

from torch.utils.data import DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb

from models.components.lit_generic import LitGenericModel
from models.torch_models.model_selector import get_model


class PTBXL_Dataset(torch.utils.data.Dataset):
    """PTB-XL dataset that returns (T, C) tensors per record with per-lead z-score normalization."""
    def __init__(self, records, labels, signal_path, sr=500):
        self.records = records
        self.labels = labels
        self.signal_path = signal_path
        self.sr = sr

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        label = self.labels[idx]
        signal, _ = wfdb.rdsamp(f"{self.signal_path}/{rec}")  # (samples, channels)
        seq_len = 5000 if self.sr == 500 else 1000
        signal = signal[:seq_len]  # (T, C)

        # Per-lead z-score (record-wise) — must match eval.py
        mean = signal.mean(axis=0, keepdims=True)
        std = signal.std(axis=0, keepdims=True) + 1e-6
        signal = (signal - mean) / std

        signal = torch.tensor(signal, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        return signal, label


def main():
    parser = argparse.ArgumentParser(description='Train a model with PyTorch Lightning (W&B enabled)')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name (cnn_transformer, resnet1d, transformer, xlstm, xresnet1d)')
    parser.add_argument('--input-channels', type=int, default=12)
    parser.add_argument('--seq-len', type=int, default=5000)
    parser.add_argument('--num-classes', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--signal-path', type=str, required=True)
    parser.add_argument('--csv-path', type=str, required=True)
    parser.add_argument('--sr', type=int, default=500, choices=[100, 500],
                        help='Sampling rate per lead used for sequence length (100->1000, 500->5000)')

    # Weights & Biases flags
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--project', type=str, default='ptbxl-ecg', help='W&B project name')
    parser.add_argument('--entity', type=str, default=None, help='W&B entity (user or team)')
    parser.add_argument('--run-name', type=str, default=None, help='W&B run name')
    parser.add_argument('--log-artifact', action='store_true', help='Log best checkpoint to W&B as an artifact')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=2, help='OSC suggests <=2 workers')
    args = parser.parse_args()

    
    # Reproducibility
    
    pl.seed_everything(args.seed, workers=True)

    # Ensure checkpoint dir exists (fixed dirpath usage)
    os.makedirs('lightning_logs/checkpoints', exist_ok=True)

   
    # Callbacks
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        save_last=True,
        mode="min",
        dirpath="lightning_logs/checkpoints",
        filename="epoch{epoch}-step{step}-val{val_loss:.4f}",
        every_n_epochs=1,
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

   
    # Data prep
    
    df = pd.read_csv(args.csv_path)
    df['scp_codes'] = df['scp_codes'].apply(ast.literal_eval)
    df['diagnostic_superclass'] = df['scp_codes'].apply(lambda x: list(x.keys()))

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df['diagnostic_superclass'])

    records = df['filename_hr'].str.replace('.hea', '', regex=False).values

    try:
        train_rec, val_rec, y_train, y_val = train_test_split(
            records, y, test_size=0.2, random_state=args.seed, stratify=y
        )
    except ValueError:
        train_rec, val_rec, y_train, y_val = train_test_split(
            records, y, test_size=0.2, random_state=args.seed
        )

    train_ds = PTBXL_Dataset(train_rec, y_train, args.signal_path, sr=args.sr)
    val_ds = PTBXL_Dataset(val_rec, y_val, args.signal_path, sr=args.sr)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            num_workers=args.num_workers, pin_memory=pin)

    # --- compute class imbalance weights on the TRAIN split ---
    # pos_weight = (num_negative / num_positive) per class
    pos = torch.tensor(y_train.sum(axis=0), dtype=torch.float32) + 1e-6
    neg = torch.tensor(y_train.shape[0] - y_train.sum(axis=0), dtype=torch.float32) + 1e-6
    pos_weight = neg / pos  # shape [num_classes]

  
    # Model
    
    model = get_model(
        args.model,
        input_shape=(args.input_channels, args.seq_len),
        num_classes=len(mlb.classes_),
    )
    lit_model = LitGenericModel(
        model=model,
        lr=args.lr,
        weight_decay=1e-2,
        pos_weight=pos_weight,
        optimizer_class=torch.optim.AdamW,
        use_scheduler=True,
        
    )

   
    # Logger (W&B)
  
    logger = None
    if args.wandb:
        logger = WandbLogger(
            project=args.project,
            entity=args.entity,
            name=args.run_name,
            log_model=False,
            save_dir="wandb"
        )
        # make metrics visible in W&B UI
        logger.experiment.define_metric("train_loss", summary="min")
        logger.experiment.define_metric("val_loss", summary="min")
        logger.experiment.define_metric("lr", summary="last")
        # Log hyperparams & class names
        logger.experiment.config.update({
            **vars(args),
            "model_name": args.model,
            "num_classes": len(mlb.classes_),
            "classes": list(mlb.classes_)
        }, allow_val_change=True)
        try:
            logger.watch(lit_model, log="gradients", log_freq=200)
        except Exception:
            pass

   
    # Trainer
   
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback, lr_monitor],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger,
        default_root_dir="lightning_logs",
        deterministic=True,
        log_every_n_steps=10,
    )

    # Train
    trainer.fit(lit_model, train_loader, val_loader)

    # Explicitly finish W&B run to push logs promptly
    if args.wandb:
        try:
            wandb.finish()
        except Exception:
            pass

    # Optional: log best checkpoint as a W&B artifact
    if args.wandb and args.log_artifact and checkpoint_callback.best_model_path:
        art = wandb.Artifact(
            name=f"{args.model}-best",
            type="model",
            metadata={
                "best_model_score": float(checkpoint_callback.best_model_score.cpu().item()) if checkpoint_callback.best_model_score is not None else None,
                "classes": list(mlb.classes_),
                **{k: v for k, v in vars(args).items() if k not in {"entity"}}
            }
        )
        art.add_file(checkpoint_callback.best_model_path)
        logger.experiment.log_artifact(art)


if __name__ == '__main__':
    main()
