"""
Train YieldNet — predicts rendimiento_real (t/ha) from 8 harvest features.

Architecture: MLP 8→32→16→1 with MC Dropout for confidence scores.
Usage: python3 train_yield.py
Output: ../models/yield_net.pt
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from utils import StandardScaler, train, mae, save_checkpoint

FEATURES = [
    "humedad_grano", "velocidad", "fertilizante", "pesticida",
    "temperatura", "humedad_aire", "horas_motor", "temp_motor",
]
TARGET  = "rendimiento_real"
DATA    = Path(__file__).parent.parent / "raw" / "yield_harvest.csv"
OUT     = Path(__file__).parent.parent / "models" / "yield_net.pt"


class YieldNet(nn.Module):
    """
    Shallow MLP with Dropout between every hidden layer.
    Keeping Dropout active at inference time (model.train()) enables
    MC Dropout: run N forward passes → mean = prediction, std = uncertainty.
    """
    def __init__(self, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


def mc_predict(model, x_norm: np.ndarray, n_samples: int = 50):
    """
    Monte-Carlo Dropout inference.
    Returns (mean_prediction, std_prediction) in normalised space.
    """
    model.train()   # keep dropout active
    x_t = torch.tensor(x_norm, dtype=torch.float32)
    with torch.no_grad():
        samples = torch.stack([model(x_t) for _ in range(n_samples)], dim=0)
    return samples.mean(0).numpy(), samples.std(0).numpy()


def confidence_score(mc_std_norm: float, scaler_y: StandardScaler) -> int:
    """Map MC std (in normalised space) to a 40–95% confidence percentage."""
    # Denormalise the std estimate
    std_tha = float(mc_std_norm) * float(scaler_y.std_[0])
    # Linear mapping: std=0 → 95%, std≥0.6 → 40%
    score = 100 - (std_tha / 0.6) * 60
    return int(np.clip(score, 40, 95))


def main():
    print("YieldNet training")
    print("─" * 40)

    df = pd.read_csv(DATA)
    X  = df[FEATURES].values.astype(np.float32)
    y  = df[TARGET].values.astype(np.float32)

    # 80 / 20 split
    rng   = np.random.default_rng(99)
    idx   = rng.permutation(len(X))
    split = int(0.8 * len(X))
    tr, va = idx[:split], idx[split:]

    scaler_X = StandardScaler().fit(X[tr])
    scaler_y = StandardScaler().fit(y[tr].reshape(-1, 1))

    X_tr = scaler_X.transform(X[tr])
    X_va = scaler_X.transform(X[va])
    y_tr = scaler_y.transform(y[tr].reshape(-1, 1)).flatten()
    y_va = scaler_y.transform(y[va].reshape(-1, 1)).flatten()

    model = YieldNet(dropout=0.15)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters())}")

    train(model, X_tr, y_tr, X_va, y_va,
          epochs=1000, batch_size=32, lr=3e-3, weight_decay=5e-3, patience=80)

    val_mae = mae(model, X_va, y[va], scaler_y)
    print(f"  Validation MAE: {val_mae:.3f} t/ha  (noise floor ~0.40)")

    # Quick MC Dropout demo on one sample
    sample_norm = scaler_X.transform(X[va[:1]])
    mean_norm, std_norm = mc_predict(model, sample_norm)
    mean_tha = scaler_y.inverse_transform(mean_norm).flatten()[0]
    conf = confidence_score(float(std_norm.flatten()[0]), scaler_y)
    print(f"  Sample prediction: {mean_tha:.2f} t/ha  confidence={conf}%")

    save_checkpoint(str(OUT), model, scaler_X, scaler_y,
                    meta={"features": FEATURES, "target": TARGET, "val_mae": val_mae})
    print("Done.")


if __name__ == "__main__":
    main()
