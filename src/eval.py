# Filename: src/eval.py

import argparse
import glob
import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.serialization import add_safe_globals
import wfdb
from torch.utils.data import DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torchmetrics.functional.classification import multilabel_accuracy as tm_multilabel_accuracy

from models.components.lit_generic import LitGenericModel
from models.torch_models.model_selector import get_model

# Allowlist optimizer classes that may appear inside Lightning checkpoints when loading with weights_only
add_safe_globals([torch.optim.AdamW])

# Evaluate only these 8 classes if present in the CSV label space
TARGET_CLASSES = ['NORM', 'AFIB', 'PVC', 'LVH', 'IMI', 'ASMI', 'LAFB', 'IRBBB']


class PTBXL_Dataset(torch.utils.data.Dataset):
    """Returns (T, C) tensors with per-lead z-score normalization."""
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

        # Per-lead z-score (record-wise)
        mean = signal.mean(axis=0, keepdims=True)
        std = signal.std(axis=0, keepdims=True) + 1e-6
        signal = (signal - mean) / std

        signal = torch.tensor(signal, dtype=torch.float32)   # (T, C)
        label = torch.tensor(label, dtype=torch.float32)     # (num_classes,)
        return signal, label


def pick_checkpoint(args):
    """Return a checkpoint path from args.checkpoint or newest in args.checkpoint_dir."""
    if args.checkpoint:
        return args.checkpoint
    ckpt_dir = args.checkpoint_dir or "lightning_logs/checkpoints"
    candidates = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt")))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in: {ckpt_dir}")
    return candidates[-1]  # newest by lexicographic sort


def main():
    parser = argparse.ArgumentParser(description="Evaluate a model and print VAL classification report + TRAIN accuracy")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name (cnn_transformer, resnet1d, transformer, xlstm, xresnet1d, or 'auto')")
    parser.add_argument("--input-channels", type=int, default=12)
    parser.add_argument("--seq-len", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--signal-path", type=str, required=True)
    parser.add_argument("--csv-path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a specific .ckpt file")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Directory containing .ckpt files (auto-choose newest)")
    parser.add_argument("--sr", type=int, default=500, choices=[100, 500])
    args = parser.parse_args()

    
    # Dataframe & labels
    
    df = pd.read_csv(args.csv_path)
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)
    df["diagnostic_superclass"] = df["scp_codes"].apply(lambda x: list(x.keys()))

    # Fit on ALL classes present in the CSV so the output dimension matches the checkpoint
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df["diagnostic_superclass"])  # (N, num_all_classes)

    # Map from our 8 target class names to their indices in the full class list
    name_to_idx = {name: i for i, name in enumerate(mlb.classes_)}
    present_targets = [c for c in TARGET_CLASSES if c in name_to_idx]
    if not present_targets:
        raise RuntimeError("None of the requested TARGET_CLASSES exist in the fitted label space.")
    target_idxs = [name_to_idx[c] for c in present_targets]

    records = df["filename_hr"].str.replace(".hea", "", regex=False).values

    # Same split seed as training for reproducibility
    train_rec, val_rec, y_train, y_val = train_test_split(records, y, test_size=0.2, random_state=42)

    train_ds = PTBXL_Dataset(train_rec, y_train, args.signal_path, sr=args.sr)
    val_ds   = PTBXL_Dataset(val_rec, y_val, args.signal_path, sr=args.sr)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size)

    
    # Build model & load checkpoint
    
    ckpt_path = pick_checkpoint(args)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    ckpt_sd = ckpt["state_dict"]

    has_tx  = any(("transformer" in k.lower() or "encoder" in k.lower() or "input_linear" in k) for k in ckpt_sd.keys())
    has_cnn = any(("cnn." in k.lower() or "conv" in k.lower()) for k in ckpt_sd.keys())

    detected_model = args.model
    if args.model == "auto":
        if has_tx and has_cnn:
            detected_model = "cnn_transformer"
        elif has_tx:
            detected_model = "transformer"

    # Infer number of classes from checkpoint if possible
    ckpt_num_classes = None
    for k in ["model.fc.weight", "model.classifier.weight", "model.head.weight"]:
        if k in ckpt_sd:
            ckpt_num_classes = ckpt_sd[k].shape[0]
            break
    if ckpt_num_classes is None:
        ckpt_num_classes = len(mlb.classes_)

    # Instantiate backbone
    model = get_model(
        detected_model,
        input_shape=(args.input_channels, args.seq_len),
        num_classes=ckpt_num_classes,
    )

    # Build LightningModule and load state dict non-strictly
    lit_model = LitGenericModel(model)
    lit_model.load_state_dict(ckpt_sd, strict=False)
    lit_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lit_model.to(device)

    
    # VAL inference for classification report
    
    y_true, y_pred = [], []
    with torch.no_grad():
        for signals, labels in val_loader:
            signals = signals.to(device)
            outputs = lit_model(signals)
            y_true.append(labels.numpy())
            y_pred.append(torch.sigmoid(outputs).cpu().numpy())

    y_true_all = np.vstack(y_true)
    y_pred_all = np.vstack(y_pred)

    # Slice to our present targets
    y_true = y_true_all[:, target_idxs]
    y_pred = y_pred_all[:, target_idxs]

    # Print validation classification report
    y_pred_bin = (y_pred >= 0.5).astype(int)

    print("\nClassification Report:")
    print(classification_report(
        y_true, y_pred_bin, target_names=present_targets, zero_division=0
    ))

    
    # TRAIN accuracy (TorchMetrics multilabel_accuracy, micro, thr=0.5)
    
    t_true, t_pred = [], []
    with torch.no_grad():
        for signals, labels in train_loader:
            signals = signals.to(device)
            outputs = lit_model(signals)
            t_true.append(labels.numpy())
            t_pred.append(torch.sigmoid(outputs).cpu().numpy())

    t_true = np.vstack(t_true)[:, target_idxs]
    t_pred = np.vstack(t_pred)[:, target_idxs]

    tm_train_acc = tm_multilabel_accuracy(
        torch.tensor(t_pred, dtype=torch.float32),
        torch.tensor(t_true, dtype=torch.int32),
        num_labels=len(target_idxs),
        threshold=0.5,
        average="micro",
    ).item()

    print("\nAccuracy:")
    print(f" {tm_train_acc:.3f}")


if __name__ == "__main__":
    main()