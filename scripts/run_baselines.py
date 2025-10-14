import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import numpy as np
from load_data import load_monthly_finance_data
from metrics import smape, mae, mape
from baselines import naive_forecast, seasonal_naive_forecast

if __name__ == "__main__":
    data = load_monthly_finance_data()
    h = len(data[0]["test"])

    sm_naive, sm_snaive = [], []
    mae_naive, mae_snaive = [], []
    mp_naive, mp_snaive = [], []

    for item in data:
        y_tr, y_te = item["train"], item["test"]
        p_naive  = naive_forecast(y_tr, h)
        p_snaive = seasonal_naive_forecast(y_tr, h, season=12)

        sm_naive.append(smape(y_te, p_naive));   mae_naive.append(mae(y_te, p_naive));   mp_naive.append(mape(y_te, p_naive))
        sm_snaive.append(smape(y_te, p_snaive)); mae_snaive.append(mae(y_te, p_snaive)); mp_snaive.append(mape(y_te, p_snaive))

    print("Baseline averages over Finance (Monthly)")
    print("Naive        | sMAPE:", round(float(np.mean(sm_naive)),2),  "MAE:", round(float(np.mean(mae_naive)),2), "MAPE:", round(float(np.mean(mp_naive)),2))
    print("SeasonalNaive| sMAPE:", round(float(np.mean(sm_snaive)),2), "MAE:", round(float(np.mean(mae_snaive)),2), "MAPE:", round(float(np.mean(mp_snaive)),2))