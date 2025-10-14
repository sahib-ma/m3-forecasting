import numpy as np

def make_windows(series, input_len, output_len):
    X, Y = [], []
    n = len(series)
    for i in range(n - input_len - output_len + 1):
        X.append(series[i:i+input_len])
        Y.append(series[i+input_len:i+input_len+output_len])
    return np.asarray(X, float), np.asarray(Y, float)