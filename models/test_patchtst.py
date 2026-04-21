import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


FEATURE_COLUMNS = [
    "event_count",
    "group_lag",
    "pending_count",
    "used_memory_bytes",
    "host_cpu_util_pct",
    "host_mem_util_pct",
]

TARGET_COLUMN = "group_lag"


class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse_transform_target(self, y: np.ndarray, target_idx: int) -> np.ndarray:
        return y * self.std[0, target_idx] + self.mean[0, target_idx]


class TimeSeriesWindowDataset(Dataset):
    def __init__(self, features: np.ndarray, target_idx: int, lookback: int, horizon: int):
        self.x = []
        self.y = []

        total_len = len(features)
        end_idx = total_len - lookback - horizon + 1

        for start in range(end_idx):
            end = start + lookback
            target_t = end + horizon - 1

            x_window = features[start:end]
            y_value = features[target_t, target_idx]

            self.x.append(x_window.astype(np.float32))
            self.y.append(np.float32(y_value))

        self.x = np.array(self.x, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class PatchTSTRegressor(nn.Module):
    """
    입력: [B, L, C]
    출력: [B]  (future group_lag scalar)
    """
    def __init__(
        self,
        lookback: int,
        num_features: int,
        patch_len: int,
        stride: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        ff_dim: int,
        dropout: float,
    ):
        super().__init__()

        self.lookback = lookback
        self.num_features = num_features
        self.patch_len = patch_len
        self.stride = stride

        self.num_patches = 1 + (lookback - patch_len) // stride
        if self.num_patches <= 0:
            raise ValueError("Invalid patch_len/stride for given lookback")

        patch_dim = patch_len * num_features

        self.patch_proj = nn.Linear(patch_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=self.num_patches + 4)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def patchify(self, x):
        patches = []
        for start in range(0, self.lookback - self.patch_len + 1, self.stride):
            end = start + self.patch_len
            patch = x[:, start:end, :]
            patch = patch.reshape(x.size(0), -1)
            patches.append(patch)

        patches = torch.stack(patches, dim=1)
        return patches

    def forward(self, x):
        x = self.patchify(x)
        x = self.patch_proj(x)
        x = self.pos_enc(x)
        x = self.dropout(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        y = self.head(x).squeeze(-1)
        return y


def load_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, header=None)
    df.columns = [
        "run_id",
        "replay_time",
        "sample_index",
        "sampling_interval_sec",
        "event_count",
        "group_lag",
        "pending_count",
        "used_memory_bytes",
        "host_cpu_util_pct",
        "host_mem_util_pct",
    ]

    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    return df


def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs_patchtst_test")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.model_path, map_location=args.device, weights_only=False)

    config = checkpoint["config"]
    target_idx = checkpoint["target_idx"]
    feature_columns = checkpoint["feature_columns"]

    lookback = config["lookback"]
    horizon = config["horizon"]

    scaler = StandardScaler()
    scaler.mean = np.array(checkpoint["scaler_mean"], dtype=np.float32)
    scaler.std = np.array(checkpoint["scaler_std"], dtype=np.float32)

    df_test = load_dataframe(args.test_csv)
    test_feat_raw = df_test[feature_columns].values.astype(np.float32)
    test_feat = scaler.transform(test_feat_raw)

    test_ds = TimeSeriesWindowDataset(
        test_feat,
        target_idx=target_idx,
        lookback=lookback,
        horizon=horizon
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = PatchTSTRegressor(
        lookback=lookback,
        num_features=len(feature_columns),
        patch_len=config["patch_len"],
        stride=config["stride"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        num_layers=config["num_layers"],
        ff_dim=config["ff_dim"],
        dropout=config["dropout"],
    ).to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    preds_scaled = []
    trues_scaled = []

    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()

    inference_start = time.perf_counter()

    for x, y in test_loader:
        x = x.to(args.device)
        pred = model(x).detach().cpu().numpy()

        preds_scaled.append(pred)
        trues_scaled.append(y.numpy())

    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()

    inference_end = time.perf_counter()

    preds_scaled = np.concatenate(preds_scaled, axis=0)
    trues_scaled = np.concatenate(trues_scaled, axis=0)

    preds = scaler.inverse_transform_target(preds_scaled, target_idx=target_idx)
    trues = scaler.inverse_transform_target(trues_scaled, target_idx=target_idx)

    test_mse = mse(trues, preds)
    test_rmse = rmse(trues, preds)
    test_mae = mae(trues, preds)

    total_inference_sec = inference_end - inference_start
    num_samples = len(preds)
    mean_inference_ms_per_sample = (total_inference_sec / max(num_samples, 1)) * 1000.0

    metrics = {
        "test_csv": args.test_csv,
        "model_path": args.model_path,
        "num_test_samples": int(num_samples),
        "lookback": int(lookback),
        "horizon": int(horizon),
        "mse": test_mse,
        "rmse": test_rmse,
        "mae": test_mae,
        "total_inference_sec": total_inference_sec,
        "mean_inference_ms_per_sample": mean_inference_ms_per_sample,
        "device": args.device,
    }

    metrics_path = output_dir / "test_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    pred_df = pd.DataFrame({
        "y_true": trues,
        "y_pred": preds,
        "abs_error": np.abs(trues - preds),
    })
    pred_path = output_dir / "test_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    print("[INFO] Test evaluation completed.")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[INFO] metrics saved to: {metrics_path}")
    print(f"[INFO] predictions saved to: {pred_path}")


if __name__ == "__main__":
    main()