"""Shared utilities for all AgroIntel training scripts."""

import numpy as np
import torch
from torch.utils.data import Dataset


# ── Normalisation ─────────────────────────────────────────────────────────────

class StandardScaler:
    """numpy-only StandardScaler (no sklearn dependency)."""

    def __init__(self):
        self.mean_ = None
        self.std_  = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        self.mean_ = X.mean(axis=0)
        self.std_  = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0   # avoid division by zero on constant cols
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return X * self.std_ + self.mean_


# ── Dataset ───────────────────────────────────────────────────────────────────

class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Early stopping ────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 80, min_delta: float = 1e-5):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.counter    = 0
        self.best_state = None

    def step(self, val_loss: float, model: torch.nn.Module) -> bool:
        """Returns True when training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.counter    = 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model: torch.nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ── Model persistence ─────────────────────────────────────────────────────────

def save_checkpoint(path: str, model: torch.nn.Module,
                    scaler_X: StandardScaler, scaler_y: StandardScaler,
                    meta: dict):
    torch.save({
        "model_state":   model.state_dict(),
        "scaler_X_mean": scaler_X.mean_,
        "scaler_X_std":  scaler_X.std_,
        "scaler_y_mean": scaler_y.mean_,
        "scaler_y_std":  scaler_y.std_,
        "meta":          meta,
    }, path)
    print(f"  Saved → {path}")


def load_checkpoint(path: str, model: torch.nn.Module):
    ck = torch.load(path, map_location="cpu")
    model.load_state_dict(ck["model_state"])

    sx = StandardScaler()
    sx.mean_, sx.std_ = ck["scaler_X_mean"], ck["scaler_X_std"]

    sy = StandardScaler()
    sy.mean_, sy.std_ = ck["scaler_y_mean"], ck["scaler_y_std"]

    return model, sx, sy, ck.get("meta", {})


# ── Shared training loop ──────────────────────────────────────────────────────

def train(model, X_train, y_train, X_val, y_val,
          epochs=1000, batch_size=32, lr=3e-3, weight_decay=5e-3, patience=80,
          verbose=True):
    from torch.utils.data import DataLoader

    train_loader = DataLoader(TabularDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.MSELoss()
    stopper   = EarlyStopping(patience=patience)

    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()

        if stopper.step(val_loss, model):
            if verbose:
                print(f"  Early stop @ epoch {epoch}  val_loss={val_loss:.5f}")
            break

        if verbose and epoch % 100 == 0:
            print(f"  epoch {epoch:4d}  val_loss={val_loss:.5f}")

    stopper.restore_best(model)
    return model


def mae(model, X, y_true_raw, scaler_y):
    model.eval()
    with torch.no_grad():
        y_pred_norm = model(torch.tensor(X, dtype=torch.float32)).numpy().flatten()
    y_pred = scaler_y.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()
    return float(np.mean(np.abs(y_pred - y_true_raw)))
