import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import csv
import numpy as np
import torch
from pathlib import Path

from load_data import load_monthly_finance_data
from train_model import cv_select_hyperparams, train_one_series, forecast_series
from metrics import smape, mae, mape
from baselines import seasonal_naive_forecast
from make_windows import make_windows
from graphs import plot_loss_curve


if __name__ == "__main__":
    torch.set_num_threads(1)
    Path("figures").mkdir(exist_ok=True)

    data = load_monthly_finance_data()
    h = len(data[0]["test"])

    rows = [(
        "id","cfg_input_len","cfg_hidden","lr","wd",
        "train_smape","test_smape","train_mae","test_mae","train_mape","test_mape",
        "snaive_smape","snaive_mae","snaive_mape"
    )]

    overfit_count = 0
    total_series = len(data)

    all_train_losses = []
    all_val_losses = []

    for i, item in enumerate(data, 1):
        y_tr, y_te = item["train"], item["test"]

        cfg = cv_select_hyperparams(y_tr, h)

        model, comps, loss_history = train_one_series(
            y_tr, h,
            input_len=cfg["input_len"],
            hidden=cfg["hidden"],
            lr=cfg["lr"],
            wd=cfg["wd"],
            epochs=180,
            patience=8
        )

        resid_norm = comps["resid_norm"]
        Xw, Yw = make_windows(resid_norm, input_len=cfg["input_len"], output_len=h)
        if len(Xw) == 0:
            continue

        model.eval()
        with torch.no_grad():
            last_inp = torch.tensor(Xw[-1], dtype=torch.float32).unsqueeze(0)
            resid_pred_train_std = model(last_inp).numpy()[0]

        mu = float(comps["mu"])
        sigma = float(comps["sigma"])
        resid_pred_train = resid_pred_train_std * sigma + mu
        trend_seg    = np.asarray(comps["trend"])[-h:]
        seasonal_seg = np.asarray(comps["seasonal"])[-h:]
        yhat_train   = resid_pred_train + trend_seg + seasonal_seg
        y_train_seg  = np.asarray(y_tr)[-h:]

        mlp_s_train = smape(y_train_seg, yhat_train)
        mlp_a_train = mae(y_train_seg, yhat_train)
        mlp_p_train = mape(y_train_seg, yhat_train)

        yhat_test = forecast_series(model, comps, y_tr, h)

        mlp_s_test = smape(y_te, yhat_test)
        mlp_a_test = mae(y_te, yhat_test)
        mlp_p_test = mape(y_te, yhat_test)

        snaive = seasonal_naive_forecast(y_tr, h, season=12)
        sn_s = smape(y_te, snaive)
        sn_a = mae(y_te, snaive)
        sn_p = mape(y_te, snaive)

        if (mlp_s_test - mlp_s_train) > 5.0:
            overfit_count += 1

        print(f"[{i}/{total_series}] sMAPE  Train={mlp_s_train:.2f} | Test={mlp_s_test:.2f} | S-Naive={sn_s:.2f}")

        rows.append((
            item["id"], cfg["input_len"], cfg["hidden"], cfg["lr"], cfg["wd"],
            mlp_s_train, mlp_s_test, mlp_a_train, mlp_a_test, mlp_p_train, mlp_p_test,
            sn_s, sn_a, sn_p
        ))


    out_csv = Path("figures") / "results_finance_monthly.csv"
    with open(out_csv, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"\nSaved per-series results to {out_csv}")


    arr = np.array(rows[1:], dtype=object)
    print("\n= Averages over all series =")
    print("MLP   Train sMAPE:", round(np.mean(arr[:,5].astype(float)),2),
          "| Test sMAPE:", round(np.mean(arr[:,6].astype(float)),2))
    print("MLP   MAE:", round(np.mean(arr[:,8].astype(float)),2),
          "| MAPE:", round(np.mean(arr[:,10].astype(float)),2))
    print("S-NA  sMAPE:", round(np.mean(arr[:,11].astype(float)),2),
          "MAE:", round(np.mean(arr[:,12].astype(float)),2),
          "MAPE:", round(np.mean(arr[:,13].astype(float)),2))


    max_len = max(len(l) for l in all_train_losses)
    avg_train = np.mean([np.pad(l, (0, max_len-len(l)), 'edge') for l in all_train_losses], axis=0)
    avg_val   = np.mean([np.pad(l, (0, max_len-len(l)), 'edge') for l in all_val_losses], axis=0)

    plot_loss_curve(avg_train, avg_val, save_path=Path("figures")/"avg_loss_curve.png")

    print(f"\nSeries with potential overfitting (Test - Train sMAPE > 5): {overfit_count}/{total_series} "
          f"({overfit_count/total_series*100:.1f}%)")
