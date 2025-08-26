# Filename: src/eval.py
# Purpose : Evaluate a trained checkpoint on the PTB-XL validation split.
#           - Auto-detects whether the checkpoint is Transformer or CNN+Transformer
#           - Auto-detects whether positional embeddings (pos_embed) were used
#           - Fits label space on ALL classes in CSV (to match checkpoint head)
#           - Reports Macro ROC AUC + AUROC per class for 8 target classes only

import argparse
import glob
import os
import ast
import re
import numpy as np
import pandas as pd
import torch
from torch.serialization import add_safe_globals
import wfdb
from torch.utils.data import DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve

from models.components.lit_generic import LitGenericModel
from models.torch_models.model_selector import get_model

# Allowlist optimizer classes that may appear inside Lightning checkpoints when loading with weights_only
add_safe_globals([torch.optim.AdamW])

# --- Evaluate only these 8 classes ---
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
    return candidates[-1]  # newest by lexicographic sort (filenames contain epoch/step)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a model with PyTorch Lightning")
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

    # --------------------
    # Dataframe & labels
    # --------------------
    df = pd.read_csv(args.csv_path)
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)
    df["diagnostic_superclass"] = df["scp_codes"].apply(lambda x: list(x.keys()))

    # Fit on ALL classes present in the CSV so the output dimension matches the checkpoint
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df["diagnostic_superclass"])  # (N, num_all_classes)

    # Map from our 8 target class names to their indices in the full class list
    name_to_idx = {name: i for i, name in enumerate(mlb.classes_)}
    missing_targets = [c for c in TARGET_CLASSES if c not in name_to_idx]
    if missing_targets:
        print(f"WARNING: These target classes are not present in this CSV label space and will be skipped: {missing_targets}")
    target_idxs = [name_to_idx[c] for c in TARGET_CLASSES if c in name_to_idx]

    records = df["filename_hr"].str.replace(".hea", "", regex=False).values

    # Use the same split seed as training for reproducibility
    _, val_rec, _, y_val = train_test_split(records, y, test_size=0.2, random_state=42)

    val_ds = PTBXL_Dataset(val_rec, y_val, args.signal_path, sr=args.sr)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # --------------------
    # Build model & load checkpoint (auto-detect arch + pos_embed)
    # --------------------
    ckpt_path = pick_checkpoint(args)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    ckpt_sd = ckpt["state_dict"]
    ckpt_keys = list(ckpt_sd.keys())

    # Heuristics to sniff arch + features from the checkpoint
    has_tx  = any(("transformer" in k.lower() or "encoder" in k.lower() or "input_linear" in k) for k in ckpt_keys)
    has_cnn = any(("cnn." in k.lower() or "conv" in k.lower()) for k in ckpt_keys)
    has_pos = any("pos_embed" in k for k in ckpt_keys)

    detected_model = args.model
    if args.model == "auto":
        if has_tx and has_cnn:
            detected_model = "cnn_transformer"
        elif has_tx:
            detected_model = "transformer"
        else:
            detected_model = args.model

    # If the user passed a model that clearly doesn't match the ckpt, switch to avoid state_dict errors
    if args.model == "cnn_transformer" and has_tx and not has_cnn:
        print("NOTE: Checkpoint looks like a pure Transformer; switching to --model transformer for eval.")
        detected_model = "transformer"
    elif args.model == "transformer" and has_tx and has_cnn:
        print("NOTE: Checkpoint looks like a CNN+Transformer; switching to --model cnn_transformer for eval.")
        detected_model = "cnn_transformer"

    # Infer number of classes from checkpoint if possible (fc layer out_features)
    ckpt_num_classes = None
    for k in ["model.fc.weight", "model.classifier.weight", "model.head.weight"]:
        if k in ckpt_sd:
            ckpt_num_classes = ckpt_sd[k].shape[0]
            break
    if ckpt_num_classes is None:
        ckpt_num_classes = len(mlb.classes_)

    # Try to pass use_pos_embed if the build function supports it
    use_pos_embed = has_pos
    try:
        model = get_model(
            detected_model,
            input_shape=(args.input_channels, args.seq_len),
            num_classes=ckpt_num_classes,
            use_pos_embed=use_pos_embed,
        )
    except TypeError:
        model = get_model(
            detected_model,
            input_shape=(args.input_channels, args.seq_len),
            num_classes=ckpt_num_classes,
        )

    # Prefer a robust load. If checkpoint includes legacy loss_fn keys, drop them.
    legacy_loss_keys = [k for k in ckpt_sd.keys() if k.startswith("loss_fn.")]
    if legacy_loss_keys:
        print(f"NOTE: Dropping legacy loss keys from checkpoint: {legacy_loss_keys}")
        for k in legacy_loss_keys:
            ckpt_sd.pop(k, None)

    # Build LightningModule and load state dict non-strictly to allow minor head/token differences
    lit_model = LitGenericModel(model)
    missing, unexpected = lit_model.load_state_dict(ckpt_sd, strict=False)

    if unexpected:
        print(f"WARNING: Unexpected keys ignored during load: {unexpected}")
    if missing:
        if all("pos_embed" in k for k in missing):
            print("WARNING: pos_embed not found in checkpoint; proceeding with default-initialized pos_embed.")
        else:
            print(f"WARNING: Missing keys during load: {missing}")

    lit_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lit_model.to(device)

    # --------------------
    # Eval loop
    # --------------------
    y_true, y_pred = [], []
    with torch.no_grad():
        for signals, labels in val_loader:
            signals = signals.to(device)
            outputs = lit_model(signals)
            y_true.append(labels.numpy())
            y_pred.append(torch.sigmoid(outputs).cpu().numpy())

    y_true_all = np.vstack(y_true)
    y_pred_all = np.vstack(y_pred)

    # Slice to our 8 target classes
    if len(target_idxs) == 0:
        raise RuntimeError("None of the requested TARGET_CLASSES exist in the fitted label space; cannot compute metrics.")
    y_true = y_true_all[:, target_idxs]
    y_pred = y_pred_all[:, target_idxs]

    # --------------------
    # Metrics
    # --------------------
    macro_auc = roc_auc_score(y_true, y_pred, average="macro")
    print("Macro ROC AUC (8 targets):", macro_auc)

    # AUROC per class
    present_targets = [c for c in TARGET_CLASSES if c in name_to_idx]
    for i, cls in enumerate(present_targets):
        try:
            auc_val = roc_auc_score(y_true[:, i], y_pred[:, i])
            print(f"AUROC for {cls}: {auc_val:.4f}")
        except ValueError:
            print(f"AUROC for {cls}: not defined (only one class present in y_true)")

    # --------------------
    # Baseline threshold=0.5 report
    # --------------------
    y_pred_05 = (y_pred >= 0.5).astype(np.int32)
    print("\nClassification Report (threshold=0.5, 8 targets):")
    print(classification_report(y_true, y_pred_05, target_names=present_targets))

    # --------------------
    # Threshold tuning per class for F1
    # --------------------
    best_thresholds = []
    for c in range(y_true.shape[1]):
        p, r, t = precision_recall_curve(y_true[:, c], y_pred[:, c])
        f1s = (2 * p[1:] * r[1:]) / (p[1:] + r[1:] + 1e-12)
        if len(f1s) == 0:
            best_thresholds.append(0.5)
            continue
        best_idx = np.nanargmax(f1s)
        best_thr = t[best_idx]
        best_thresholds.append(float(best_thr))

    print("Per-class tuned thresholds:", dict(zip(present_targets, [f"{th:.3f}" for th in best_thresholds])))

    y_pred_bin = (y_pred >= np.array(best_thresholds)[None, :]).astype(np.int32)
    print("Classification Report with tuned thresholds (8 targets):")
    print(classification_report(y_true, y_pred_bin, target_names=present_targets))


if __name__ == "__main__":
    main()
