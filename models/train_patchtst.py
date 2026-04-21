import argparse
import json
import math
from dataclasses import dataclass, asdict
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


@dataclass
class TrainConfig:
    train_csv: str
    output_dir: str = "./outputs_patchtst"
    lookback: int = 60
    horizon: int = 30
    train_ratio: float = 0.8
    batch_size: int = 64
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-5
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # PatchTST-style settings
    patch_len: int = 12
    stride: int = 6
    d_model: int = 128
    n_heads: int = 4
    num_layers: int = 3
    ff_dim: int = 256
    dropout: float = 0.1


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x: np.ndarray):
        self.mean = x.mean(axis=0, keepdims=True)
        self.std = x.std(axis=0, keepdims=True)
        self.std[self.std < 1e-8] = 1.0

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
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [B, N, D]
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
        # x: [B, L, C]
        patches = []
        for start in range(0, self.lookback - self.patch_len + 1, self.stride):
            end = start + self.patch_len
            patch = x[:, start:end, :]            # [B, patch_len, C]
            patch = patch.reshape(x.size(0), -1) # [B, patch_len*C]
            patches.append(patch)

        patches = torch.stack(patches, dim=1)    # [B, N, patch_len*C]
        return patches

    def forward(self, x):
        # x: [B, L, C]
        x = self.patchify(x)
        x = self.patch_proj(x)                   # [B, N, D]
        x = self.pos_enc(x)
        x = self.dropout(x)
        x = self.encoder(x)                      # [B, N, D]
        x = x.mean(dim=1)                        # global average pooling
        y = self.head(x).squeeze(-1)             # [B]
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


def split_train_val_by_time(df: pd.DataFrame, train_ratio: float):
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()
    return train_df, val_df


def build_dataloaders(config: TrainConfig):
    df = load_dataframe(config.train_csv)
    train_df, val_df = split_train_val_by_time(df, config.train_ratio)

    scaler = StandardScaler()
    scaler.fit(train_df[FEATURE_COLUMNS].values)

    train_feat = scaler.transform(train_df[FEATURE_COLUMNS].values)
    val_feat = scaler.transform(val_df[FEATURE_COLUMNS].values)

    target_idx = FEATURE_COLUMNS.index(TARGET_COLUMN)

    train_ds = TimeSeriesWindowDataset(
        train_feat,
        target_idx=target_idx,
        lookback=config.lookback,
        horizon=config.horizon,
    )
    val_ds = TimeSeriesWindowDataset(
        val_feat,
        target_idx=target_idx,
        lookback=config.lookback,
        horizon=config.horizon,
    )

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    return train_loader, val_loader, scaler, target_idx, len(FEATURE_COLUMNS)


def run_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_count = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_count += bs

    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_count = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        loss = criterion(pred, y)

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_count += bs

    return total_loss / max(total_count, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs_patchtst")
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--patch_len", type=int, default=12)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--ff_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)

    args = parser.parse_args()

    config = TrainConfig(
        train_csv=args.train_csv,
        output_dir=args.output_dir,
        lookback=args.lookback,
        horizon=args.horizon,
        train_ratio=args.train_ratio,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        patch_len=args.patch_len,
        stride=args.stride,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    )

    set_seed(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, scaler, target_idx, num_features = build_dataloaders(config)

    model = PatchTSTRegressor(
        lookback=config.lookback,
        num_features=num_features,
        patch_len=config.patch_len,
        stride=config.stride,
        d_model=config.d_model,
        n_heads=config.n_heads,
        num_layers=config.num_layers,
        ff_dim=config.ff_dim,
        dropout=config.dropout,
    ).to(config.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    history = []

    print(f"[INFO] device={config.device}")
    print(f"[INFO] train_batches={len(train_loader)}, val_batches={len(val_loader)}")
    print(f"[INFO] num_patches={1 + (config.lookback - config.patch_len) // config.stride}")

    for epoch in range(1, config.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, criterion, config.device)
        val_loss = evaluate_epoch(model, val_loader, criterion, config.device)

        history.append({
            "epoch": epoch,
            "train_mse": train_loss,
            "val_mse": val_loss,
        })

        print(f"[Epoch {epoch:03d}] train_mse={train_loss:.6f} val_mse={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                "model_state_dict": model.state_dict(),
                "config": asdict(config),
                "feature_columns": FEATURE_COLUMNS,
                "target_column": TARGET_COLUMN,
                "target_idx": target_idx,
                "scaler_mean": scaler.mean.tolist(),
                "scaler_std": scaler.std.tolist(),
                "best_val_mse": best_val_loss,
            }

    model_path = output_dir / "patchtst_best.pt"
    torch.save(best_state, model_path)

    history_path = output_dir / "train_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"[INFO] best model saved to: {model_path}")
    print(f"[INFO] history saved to: {history_path}")
    print(f"[INFO] best_val_mse={best_val_loss:.6f}")


if __name__ == "__main__":
    main()