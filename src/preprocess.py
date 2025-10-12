import numpy as np
from statsmodels.tsa.seasonal import STL

def decompose(series: np.ndarray, season: int = 12, robust: bool = True):
    series = np.asarray(series, dtype=float)
    res = STL(series, period=season, robust=robust).fit()
    return res.trend, res.seasonal, series - res.trend - res.seasonal

def _project_trend_linear(trend: np.ndarray, h: int, k: int = 12):
    x = np.arange(len(trend))
    y = np.asarray(trend, dtype=float)
    k = min(k, len(y))
    try:
        coeffs = np.polyfit(x[-k:], y[-k:], 1)  
        x_future = np.arange(len(y), len(y) + h)
        return coeffs[0] * x_future + coeffs[1]
    except Exception:
        return np.full(h, y[-1])

def project_components(trend: np.ndarray, seasonal: np.ndarray, h: int, season: int = 12,
                       trend_mode: str = "linear"):
    if trend_mode == "linear":
        trend_f = _project_trend_linear(trend, h, k=season)
    else:
        trend_f = np.full(h, float(trend[-1]))
    last_cycle = seasonal[-season:]
    reps = int(np.ceil(h / season))
    seasonal_f = np.tile(last_cycle, reps)[:h]
    return trend_f, seasonal_f

def recompose(resid_forecast: np.ndarray, trend_forecast: np.ndarray, seasonal_forecast: np.ndarray):
    return np.asarray(resid_forecast) + np.asarray(trend_forecast) + np.asarray(seasonal_forecast)

def preprocess_train(train: np.ndarray, season: int = 12):
    trend, seasonal, resid = decompose(train, season=season)
    return {"trend": trend, "seasonal": seasonal, "resid": resid}

def prepare_forecast_components(train: np.ndarray, season: int, h: int, trend_mode: str = "linear"):
    trend, seasonal, _ = decompose(train, season=season)
    trend_f, seasonal_f = project_components(trend, seasonal, h=h, season=season, trend_mode=trend_mode)
    return trend_f, seasonal_f

if __name__ == "__main__":
    from load_data import load_monthly_finance_data
    data = load_monthly_finance_data()
    s = data[0]["train"]
    comps = preprocess_train(s, season=12)
    trend_f, seasonal_f = prepare_forecast_components(s, season=12, h=len(data[0]["test"]))
    print("Train len:", len(s), "| Resid len:", len(comps["resid"]), "| Forecast comps:", len(trend_f), len(seasonal_f))