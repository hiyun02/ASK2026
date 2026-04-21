import argparse
import json
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


class MovingAvg(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        pad_left = (self.kernel_size - 1) // 2
        pad_right = self.kernel_size - 1 - pad_left

        front = x[:, 0:1, :].repeat(1, pad_left, 1)
        end = x[:, -1:, :].repeat(1, pad_right, 1)
        x = torch.cat([front, x, end], dim=1)

        x = x.permute(0, 2, 1)
        x = self.avg(x)
        x = x.permute(0, 2, 1)
        return x


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size)

    def forward(self, x):
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class DLinearRegressor(nn.Module):
    def __init__(self, lookback: int, num_features: int, target_idx: int, kernel_size: int = 25):
        super().__init__()
        self.lookback = lookback
        self.num_features = num_features
        self.target_idx = target_idx

        self.decomp = SeriesDecomp(kernel_size=kernel_size)
        self.linear_seasonal = nn.Linear(lookback, 1)
        self.linear_trend = nn.Linear(lookback, 1)

    def forward(self, x):
        seasonal, trend = self.decomp(x)
        s = seasonal[:, :, self.target_idx]
        t = trend[:, :, self.target_idx]
        y = self.linear_seasonal(s) + self.linear_trend(t)
        return y.squeeze(-1)


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
    parser.add_argument("--output_dir", type=str, default="./outputs_dlinear_test")
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

    model = DLinearRegressor(
        lookback=lookback,
        num_features=len(feature_columns),
        target_idx=target_idx,
        kernel_size=25
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