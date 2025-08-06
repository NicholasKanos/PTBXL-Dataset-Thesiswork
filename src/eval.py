import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from components.lit_generic import LitGenericModel
from models.torch_models.model_selector import get_model
import pandas as pd
import ast
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import torch
import wfdb
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score

class PTBXL_Dataset(torch.utils.data.Dataset):
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
        signal, _ = wfdb.rdsamp(f"{self.signal_path}/{rec}")
        signal = signal[:5000] if self.sr == 500 else signal[:1000]
        signal = torch.tensor(signal, dtype=torch.float32)
        signal = signal.transpose(0, 1)
        label = torch.tensor(label, dtype=torch.float32)
        return signal, label

def main():
    parser = argparse.ArgumentParser(description='Evaluate a model with PyTorch Lightning')
    parser.add_argument('--model', type=str, required=True, help='Model name (cnn_transformer, resnet1d, transformer, xlstm, xresnet1d)')
    parser.add_argument('--input-channels', type=int, default=12)
    parser.add_argument('--seq-len', type=int, default=5000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--signal-path', type=str, required=True)
    parser.add_argument('--csv-path', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    df['scp_codes'] = df['scp_codes'].apply(ast.literal_eval)
    df['diagnostic_superclass'] = df['scp_codes'].apply(lambda x: list(x.keys()))
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df['diagnostic_superclass'])
    records = df['filename_hr'].str.replace('.hea', '', regex=False).values
    _, val_rec, _, y_val = train_test_split(records, y, test_size=0.2, random_state=42)

    val_ds = PTBXL_Dataset(val_rec, y_val, args.signal_path)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = get_model(args.model, input_shape=(args.input_channels, args.seq_len), num_classes=len(mlb.classes_))
    lit_model = LitGenericModel(model)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    lit_model.load_state_dict(checkpoint['model_state_dict'])
    lit_model.eval()

    y_true, y_pred = [], []
    for signals, labels in val_loader:
        with torch.no_grad():
            outputs = lit_model(signals)
            y_true.append(labels.numpy())
            y_pred.append(torch.sigmoid(outputs).numpy())

    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)
    macro_auc = roc_auc_score(y_true, y_pred, average='macro')
    print('Macro ROC AUC:', macro_auc)
    print('Classification Report:')
    print(classification_report(y_true > 0.5, y_pred > 0.5, target_names=mlb.classes_))

if __name__ == '__main__':
    main()
