import numpy as np

def smape(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    denom = np.abs(y) + np.abs(yhat)
    return 100 * np.mean(2*np.abs(yhat - y) / np.where(denom==0, 1e-8, denom))

def mape(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return 100 * np.mean(np.abs((y - yhat) / np.where(y==0, 1e-8, y)))

def mae(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.mean(np.abs(y - yhat)))