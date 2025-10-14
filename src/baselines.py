import numpy as np

def naive_forecast(train, h):
    return np.full(h, float(train[-1]))

def seasonal_naive_forecast(train, h, season=12):
    return np.array([train[-season + (i % season)] for i in range(h)], dtype=float)