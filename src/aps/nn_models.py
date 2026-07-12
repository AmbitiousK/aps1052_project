"""Neural models (PyTorch) — MLP and LSTM with a scikit-style wrapper.

Both wrappers take a feature DataFrame (datetime-indexed) and a label Series, so
they slot into the same evaluation flow as the tree models. Shared design:
  * inputs standardized with a scaler fit on the training data only;
  * class-weighted cross-entropy to counter the 99:1 imbalance;
  * early stopping on a validation split (by macro-F1);
  * fully seeded for reproducibility (CPU is enough at this data size).

The LSTM builds fixed-length lookback sequences and only uses windows whose
hourly timestamps are strictly consecutive, so a sequence never spans a data gap
or a train/val/test boundary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from . import config as C

_HOUR = pd.Timedelta(hours=1)


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _class_weights(y) -> torch.Tensor:
    """Balanced weights w_c = N/(K*N_c) in CLASSES order, as a tensor."""
    y = np.asarray(y)
    n, k = len(y), len(C.CLASSES)
    w = [n / (k * max((y == c).sum(), 1)) for c in C.CLASSES]
    return torch.tensor(w, dtype=torch.float32)


# ===========================================================================
# MLP
# ===========================================================================
class _MLPNet(nn.Module):
    def __init__(self, n_in: int, hidden=(64, 32), p_drop=0.3):
        super().__init__()
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(p_drop)]
            d = h
        layers.append(nn.Linear(d, len(C.CLASSES)))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden=(64, 32), p_drop=0.3, lr=1e-3, max_epochs=200,
                 patience=15, batch_size=512, val_frac=0.15, seed: int = C.SEED):
        self.hidden = hidden
        self.p_drop = p_drop
        self.lr = lr
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.val_frac = val_frac
        self.seed = seed

    def fit(self, X, y):
        _seed_everything(self.seed)
        self.classes_ = np.array(C.CLASSES)
        self.scaler_ = StandardScaler().fit(np.asarray(X))
        Xs = self.scaler_.transform(np.asarray(X)).astype(np.float32)
        y_idx = np.array([C.CLASSES.index(v) for v in np.asarray(y)])

        # inner chronological validation tail for early stopping
        n = len(Xs)
        cut = int(n * (1 - self.val_frac))
        Xtr, ytr = Xs[:cut], y_idx[:cut]
        Xvl, yvl = Xs[cut:], y_idx[cut:]

        net = _MLPNet(Xs.shape[1], self.hidden, self.p_drop)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=1e-5)
        loss_fn = nn.CrossEntropyLoss(weight=_class_weights(y))

        Xtr_t = torch.tensor(Xtr); ytr_t = torch.tensor(ytr)
        Xvl_t = torch.tensor(Xvl)
        best_f1, best_state, wait = -1.0, None, 0
        for _ in range(self.max_epochs):
            net.train()
            perm = torch.randperm(len(Xtr_t))
            for i in range(0, len(Xtr_t), self.batch_size):
                idx = perm[i:i + self.batch_size]
                opt.zero_grad()
                out = net(Xtr_t[idx])
                loss = loss_fn(out, ytr_t[idx])
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                pred = net(Xvl_t).argmax(1).numpy()
            f1 = f1_score(yvl, pred, labels=range(len(C.CLASSES)),
                          average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1, best_state, wait = f1, {k: v.clone() for k, v in net.state_dict().items()}, 0
            else:
                wait += 1
                if wait >= self.patience:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        self.net_ = net
        self.best_val_f1_ = float(best_f1)
        return self

    def predict_proba(self, X):
        Xs = self.scaler_.transform(np.asarray(X)).astype(np.float32)
        self.net_.eval()
        with torch.no_grad():
            logits = self.net_(torch.tensor(Xs))
            proba = torch.softmax(logits, dim=1).numpy()
        return proba

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


# ===========================================================================
# LSTM
# ===========================================================================
def build_sequences(X_df: pd.DataFrame, y: pd.Series, lookback: int):
    """Build (n, lookback, n_feat) sequences from contiguous hourly windows.

    A window ending at row i is kept only if its `lookback` timestamps are strictly
    consecutive hours — so no sequence spans a gap or a split boundary. Returns
    (sequences, targets, end_index) where end_index are the timestamps predicted.
    """
    ts = X_df.index
    Xv = X_df[C.FEATURES].to_numpy(dtype=np.float32)
    yv = np.asarray(y)
    seqs, targets, end_idx = [], [], []
    for i in range(lookback - 1, len(X_df)):
        lo = i - lookback + 1
        window_ts = ts[lo:i + 1]
        # strictly consecutive hourly timestamps
        if (window_ts[-1] - window_ts[0]) == _HOUR * (lookback - 1) and \
           (np.diff(window_ts.values).astype("timedelta64[h]").astype(int) == 1).all():
            seqs.append(Xv[lo:i + 1])
            targets.append(yv[i])
            end_idx.append(ts[i])
    return np.stack(seqs), np.array(targets), pd.DatetimeIndex(end_idx)


class _LSTMNet(nn.Module):
    def __init__(self, n_in: int, hidden=32, n_layers=1, p_drop=0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, num_layers=n_layers, batch_first=True,
                            dropout=p_drop if n_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(p_drop), nn.Linear(hidden, len(C.CLASSES)))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])  # last timestep


class LSTMClassifier(BaseEstimator, ClassifierMixin):
    """Sequence classifier. `fit`/`predict_proba` take a datetime-indexed feature
    DataFrame; predictions are produced only for rows with a full contiguous
    lookback window (see `last_index_`)."""

    def __init__(self, lookback=24, hidden=32, n_layers=1, p_drop=0.3, lr=1e-3,
                 max_epochs=120, patience=12, batch_size=256, val_frac=0.15,
                 seed: int = C.SEED):
        self.lookback = lookback
        self.hidden = hidden
        self.n_layers = n_layers
        self.p_drop = p_drop
        self.lr = lr
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.val_frac = val_frac
        self.seed = seed

    def fit(self, X_df, y):
        _seed_everything(self.seed)
        self.classes_ = np.array(C.CLASSES)
        self.scaler_ = StandardScaler().fit(X_df[C.FEATURES].to_numpy())

        seqs, targets, _ = build_sequences(X_df, y, self.lookback)
        # scale per-timestep with the train scaler
        flat = self.scaler_.transform(seqs.reshape(-1, seqs.shape[-1]))
        seqs = flat.reshape(seqs.shape).astype(np.float32)
        y_idx = np.array([C.CLASSES.index(v) for v in targets])

        n = len(seqs)
        cut = int(n * (1 - self.val_frac))
        Xtr, ytr = seqs[:cut], y_idx[:cut]
        Xvl, yvl = seqs[cut:], y_idx[cut:]

        net = _LSTMNet(seqs.shape[-1], self.hidden, self.n_layers, self.p_drop)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=1e-5)
        loss_fn = nn.CrossEntropyLoss(weight=_class_weights(targets))

        Xtr_t = torch.tensor(Xtr); ytr_t = torch.tensor(ytr)
        Xvl_t = torch.tensor(Xvl)
        best_f1, best_state, wait = -1.0, None, 0
        for _ in range(self.max_epochs):
            net.train()
            perm = torch.randperm(len(Xtr_t))
            for i in range(0, len(Xtr_t), self.batch_size):
                idx = perm[i:i + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(net(Xtr_t[idx]), ytr_t[idx])
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                pred = net(Xvl_t).argmax(1).numpy()
            f1 = f1_score(yvl, pred, labels=range(len(C.CLASSES)),
                          average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1, best_state, wait = f1, {k: v.clone() for k, v in net.state_dict().items()}, 0
            else:
                wait += 1
                if wait >= self.patience:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        self.net_ = net
        self.best_val_f1_ = float(best_f1)
        return self

    def predict_proba_df(self, X_df, y=None) -> tuple[np.ndarray, pd.DatetimeIndex]:
        """Return (proba, end_index) — proba aligned to the predictable rows."""
        y_dummy = y if y is not None else pd.Series(0, index=X_df.index)
        seqs, _, end_idx = build_sequences(X_df, y_dummy, self.lookback)
        flat = self.scaler_.transform(seqs.reshape(-1, seqs.shape[-1]))
        seqs = flat.reshape(seqs.shape).astype(np.float32)
        self.net_.eval()
        with torch.no_grad():
            logits = self.net_(torch.tensor(seqs))
            proba = torch.softmax(logits, dim=1).numpy()
        return proba, end_idx
