import numpy as np
import torch
import torch.nn as nn

from preprocess import preprocess_train, prepare_forecast_components, recompose
from make_windows import make_windows
from models import MLP


def train_one_series(
    train_vals,
    h,
    input_len: int = 12,
    hidden: int = 32,
    lr: float = 1e-3,
    wd: float = 1e-3,
    epochs: int = 180,
    patience: int = 8
):
    comps = preprocess_train(train_vals, season=12)
    resid = np.asarray(comps["resid"], dtype=float)

    mu = float(np.mean(resid))
    sigma = float(np.std(resid) + 1e-8)
    resid_norm = (resid - mu) / sigma
    comps["mu"] = mu
    comps["sigma"] = sigma
    comps["resid_norm"] = resid_norm

    X, Y = make_windows(resid_norm, input_len=input_len, output_len=h)
    if len(X) == 0:
        return None, None, None

    X_t = torch.tensor(X, dtype=torch.float32)
    Y_t = torch.tensor(Y, dtype=torch.float32)

    n = len(X_t)
    split = max(1, int(0.8 * n))
    X_tr, Y_tr = X_t[:split], Y_t[:split]
    X_va, Y_va = X_t[split:], Y_t[split:]

    model = MLP(input_len=input_len, output_len=h, hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.L1Loss()

    best_state, best_val, bad = None, float("inf"), 0
    train_loss_history, val_loss_history = [], []

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        pred_tr = model(X_tr)
        loss_tr = loss_fn(pred_tr, Y_tr)
        loss_tr.backward()
        opt.step()
        train_loss_history.append(float(loss_tr.item()))

        model.eval()
        with torch.no_grad():
            loss_va = loss_fn(model(X_va), Y_va) if len(X_va) else loss_tr
            val_loss_history.append(float(loss_va.item()))

        if float(loss_va) < best_val - 1e-8:
            best_val = float(loss_va)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is None:
        return None, None, None

    model.load_state_dict(best_state)
    return model, comps, {"train": train_loss_history, "val": val_loss_history}


def forecast_series(model, comps, train_vals, h):
    model.eval()
    with torch.no_grad():
        L = model.net[0].in_features
        x_in = torch.tensor(comps["resid_norm"][-L:], dtype=torch.float32).unsqueeze(0)
        resid_pred_std = model(x_in).numpy()[0]

    mu = float(comps["mu"])
    sigma = float(comps["sigma"])
    resid_pred = resid_pred_std * sigma + mu

    trend_f, seasonal_f = prepare_forecast_components(train_vals, season=12, h=h)
    return recompose(resid_pred, trend_f, seasonal_f)


def rolling_splits(n, input_len, val_len, n_folds=3):
    folds = []
    start = input_len
    step = max(1, (n - input_len - val_len) // n_folds)
    for k in range(n_folds):
        val_start = start + k * step
        val_end = val_start + val_len
        if val_end > n:
            break
        folds.append((val_start, val_start, val_end))
    return folds


def cv_select_hyperparams(train_vals, h, candidates=None, season=12):
    if candidates is None:
        candidates = {
            "input_len": [12, 18],
            "hidden": [32],
            "lr": [5e-4, 7e-4, 1e-3],
            "wd": [1e-3, 2e-3, 5e-3],
        }

    comps = preprocess_train(train_vals, season=season)
    resid = np.asarray(comps["resid"], dtype=float)
    mu = float(np.mean(resid))
    sigma = float(np.std(resid) + 1e-8)
    resid_norm = (resid - mu) / sigma

    best_cfg, best_score = None, float("inf")

    for inp in candidates["input_len"]:
        X, Y = make_windows(resid_norm, input_len=inp, output_len=h)
        if len(X) == 0:
            continue
        n = len(X)
        folds = rolling_splits(n, input_len=1, val_len=1, n_folds=3)

        Xt = torch.tensor(X, dtype=torch.float32)
        Yt = torch.tensor(Y, dtype=torch.float32)

        for hid in candidates["hidden"]:
            for lr in candidates["lr"]:
                for wd in candidates["wd"]:
                    scores = []
                    for _, vs, ve in folds:
                        X_tr, Y_tr = Xt[:vs], Yt[:vs]
                        X_va, Y_va = Xt[vs:ve], Yt[vs:ve]
                        if len(X_tr) == 0 or len(X_va) == 0:
                            continue

                        model = MLP(input_len=inp, output_len=h, hidden=hid)
                        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                        loss_fn = nn.L1Loss()

                        for _ in range(120):
                            model.train()
                            opt.zero_grad()
                            pred = model(X_tr)
                            loss = loss_fn(pred, Y_tr)
                            loss.backward()
                            opt.step()

                        model.eval()
                        with torch.no_grad():
                            pv = model(X_va).numpy()[0]
                            yv = Y_va.numpy()[0]
                        denom = (np.abs(yv) + np.abs(pv))
                        denom[denom == 0] = 1e-8
                        smape_val = 100 * np.mean(2 * np.abs(pv - yv) / denom)
                        scores.append(float(smape_val))

                    if scores:
                        m = float(np.mean(scores))
                        if m < best_score:
                            best_score = m
                            best_cfg = {"input_len": inp, "hidden": hid, "lr": lr, "wd": wd}

    return best_cfg or {"input_len": 12, "hidden": 32, "lr": 1e-3, "wd": 1e-3}