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
from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib.pyplot as plt

from torchmetrics.functional.classification import multilabel_accuracy as tm_multilabel_accuracy

from models.components.lit_generic import LitGenericModel
from models.torch_models.model_selector import get_model

# Allowlist optimizer classes that may appear inside Lightning checkpoints when loading with weights_only
add_safe_globals([torch.optim.AdamW])

# Fixed evaluation set
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

        signal = torch.tensor(signal, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        return signal, label


def pick_checkpoint(args):
    """Return a checkpoint path from args.checkpoint or newest in args.checkpoint_dir."""
    if args.checkpoint:
        return args.checkpoint
    ckpt_dir = args.checkpoint_dir or "lightning_logs/checkpoints"
    candidates = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt")))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in: {ckpt_dir}")
    return candidates[-1]


def main():
    parser = argparse.ArgumentParser(description="Evaluate a model on PTB-XL with 8-class or full-class heads.")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name (cnn_transformer, resnet1d, transformer, xlstm, xresnet1d, or 'auto')")
    parser.add_argument("--input-channels", type=int, default=12)
    parser.add_argument("--seq-len", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--signal-path", type=str, required=True)
    parser.add_argument("--csv-path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--sr", type=int, default=500, choices=[100, 500])
    args = parser.parse_args()

    # ---------- Load CSV ----------
    df = pd.read_csv(args.csv_path)
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)
    df["diagnostic_superclass"] = df["scp_codes"].apply(lambda x: list(x.keys()))
    all_keys = df["diagnostic_superclass"]

    # ---------- Load checkpoint & detect head size ----------
    ckpt_path = pick_checkpoint(args)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = state["state_dict"]

    ckpt_num_classes = None
    for k in ["model.fc.weight", "model.classifier.weight", "model.head.weight"]:
        if k in sd:
            ckpt_num_classes = sd[k].shape[0]
            break
    if ckpt_num_classes is None:
        for k, v in sd.items():
            if k.endswith(".weight") and v.ndim == 2:
                bias_k = k[:-7] + ".bias"
                if bias_k in sd:
                    ckpt_num_classes = int(v.shape[0])
                    break

    # ---------- Label setup ----------
    if ckpt_num_classes == len(TARGET_CLASSES):
        # 8-class path
        df["scp_filtered"] = df["diagnostic_superclass"].apply(
            lambda keys: [k for k in keys if k in TARGET_CLASSES]
        )
        df = df[df["scp_filtered"].map(len) > 0].reset_index(drop=True)
        mlb = MultiLabelBinarizer(classes=TARGET_CLASSES)
        y = mlb.fit_transform(df["scp_filtered"])
        use_full_space = False
        target_names = TARGET_CLASSES
    else:
        # Full label space
        mlb = MultiLabelBinarizer()
        y_full = mlb.fit_transform(all_keys)
        present_targets = [c for c in TARGET_CLASSES if c in mlb.classes_]
        name_to_idx = {n: i for i, n in enumerate(mlb.classes_)}
        target_idxs = [name_to_idx[c] for c in present_targets]
        y = y_full
        use_full_space = True
        target_names = present_targets

    # ---------- Split ----------
    records = df["filename_hr"].str.replace(".hea", "", regex=False).values
    train_rec, val_rec, y_train, y_val = train_test_split(records, y, test_size=0.2, random_state=42)
    train_ds = PTBXL_Dataset(train_rec, y_train, args.signal_path, sr=args.sr)
    val_ds   = PTBXL_Dataset(val_rec,  y_val,   args.signal_path, sr=args.sr)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=pin)

    # ---------- Build model ----------
    detected_model = args.model
    if args.model == "auto":
        has_tx  = any(("transformer" in k.lower() or "encoder" in k.lower()) for k in sd.keys())
        has_cnn = any(("cnn." in k.lower() or "conv" in k.lower()) for k in sd.keys())
        if has_tx and has_cnn:
            detected_model = "cnn_transformer"
        elif has_tx:
            detected_model = "transformer"

    num_classes_for_model = ckpt_num_classes if ckpt_num_classes is not None else len(TARGET_CLASSES)
    model = get_model(
        detected_model,
        input_shape=(args.input_channels, args.seq_len),
        num_classes=num_classes_for_model,
    )
    lit_model = LitGenericModel(model)
    lit_model.load_state_dict(sd, strict=False)
    lit_model.eval().to("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- Inference ----------
    def infer(loader):
        ys, ps = [], []
        with torch.no_grad():
            for x, yb in loader:
                x = x.to(device)
                logits = lit_model(x)
                ps.append(torch.sigmoid(logits).cpu().numpy())
                ys.append(yb.numpy())
        return np.vstack(ps), np.vstack(ys)

    y_pred_all, y_true_all = infer(val_loader)
    if not use_full_space and num_classes_for_model == len(TARGET_CLASSES):
        y_pred, y_true = y_pred_all, y_true_all
    else:
        y_pred, y_true = y_pred_all[:, target_idxs], y_true_all[:, target_idxs]

    # ---------- Console Output ----------
    y_pred_bin = (y_pred >= 0.5).astype(int)
    print("\nClassification Report (targets):")
    print(classification_report(y_true, y_pred_bin, target_names=target_names,
                                zero_division=0, digits=2))

    # Train micro-accuracy
    y_pred_tr, y_true_tr = infer(train_loader)
    if not use_full_space and num_classes_for_model == len(TARGET_CLASSES):
        t_pred, t_true = y_pred_tr, y_true_tr
        n_labels = len(TARGET_CLASSES)
    else:
        t_pred, t_true = y_pred_tr[:, target_idxs], y_true_tr[:, target_idxs]
        n_labels = len(target_names)

    tm_acc = tm_multilabel_accuracy(
        torch.tensor(t_pred, dtype=torch.float32),
        torch.tensor(t_true, dtype=torch.int32),
        num_labels=n_labels,
        threshold=0.5,
        average="micro",
    ).item()
    print(f"\nTrain micro-accuracy @0.5: {tm_acc:.3f}")

    # ---------- Plots ----------
    out_dir = "lightning_logs/metrics"
    os.makedirs(out_dir, exist_ok=True)

    # ROC per label
    plt.figure()
    for k in range(len(target_names)):
        if y_true[:, k].sum() in [0, len(y_true)]:
            continue
        fpr, tpr, _ = roc_curve(y_true[:, k], y_pred[:, k])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{target_names[k]} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC per Label")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roc_per_label.png"), dpi=160)
    plt.close()

    # Macro ROC
    grid = np.linspace(0, 1, 101)
    tprs_interp = []
    valid = False
    for k in range(len(target_names)):
        if y_true[:, k].sum() in [0, len(y_true)]:
            continue
        fpr, tpr, _ = roc_curve(y_true[:, k], y_pred[:, k])
        t_interp = np.interp(grid, fpr, tpr)
        t_interp[0] = 0.0
        tprs_interp.append(t_interp)
        valid = True
    if valid:
        macro_tpr = np.mean(np.vstack(tprs_interp), axis=0)
        macro_auc = auc(grid, macro_tpr)
        plt.figure()
        plt.plot(grid, macro_tpr, label=f"Macro ROC (AUC={macro_auc:.3f})")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Macro-average ROC")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "roc_macro.png"), dpi=160)
        plt.close()


if __name__ == "__main__":
    main()