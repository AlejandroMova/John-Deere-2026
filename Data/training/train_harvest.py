"""
Train HarvestNet — predicts days until optimal harvest.

Inputs : gdd_acumulados, humedad_grano_pct, temp_promedio_c, cultivo_enc
Output : dias_para_cosechar
Usage  : python3 train_harvest.py
Output : ../models/harvest_net.pt
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from utils import StandardScaler, train, mae, save_checkpoint

FEATURES = ["gdd_acumulados", "humedad_grano_pct", "temp_promedio_c", "cultivo_enc"]
TARGET   = "dias_para_cosechar"
DATA     = Path(__file__).parent.parent / "raw" / "harvest_timing.csv"
OUT      = Path(__file__).parent.parent / "models" / "harvest_net.pt"


class HarvestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.ReLU(),  # output must be non-negative (days ≥ 0)
        )

    def forward(self, x):
        return self.net(x)


def main():
    print("HarvestNet training")
    print("─" * 40)

    df = pd.read_csv(DATA)
    X  = df[FEATURES].values.astype(np.float32)
    y  = df[TARGET].values.astype(np.float32)

    rng   = np.random.default_rng(13)
    idx   = rng.permutation(len(X))
    split = int(0.8 * len(X))
    tr, va = idx[:split], idx[split:]

    scaler_X = StandardScaler().fit(X[tr])
    scaler_y = StandardScaler().fit(y[tr].reshape(-1, 1))

    X_tr = scaler_X.transform(X[tr])
    X_va = scaler_X.transform(X[va])
    y_tr = scaler_y.transform(y[tr].reshape(-1, 1)).flatten()
    y_va = scaler_y.transform(y[va].reshape(-1, 1)).flatten()

    model = HarvestNet()
    print(f"  Parameters: {sum(p.numel() for p in model.parameters())}")

    train(model, X_tr, y_tr, X_va, y_va,
          epochs=1000, batch_size=32, lr=3e-3, weight_decay=1e-3, patience=80)

    val_mae = mae(model, X_va, y[va], scaler_y)
    print(f"  Validation MAE: {val_mae:.2f} días")

    save_checkpoint(str(OUT), model, scaler_X, scaler_y,
                    meta={"features": FEATURES, "target": TARGET, "val_mae": val_mae})
    print("Done.")


if __name__ == "__main__":
    main()
