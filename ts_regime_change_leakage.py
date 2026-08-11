import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def make_lag_features(series, n_lags):
    n = len(series)
    X = np.full((n, n_lags), np.nan)
    for lag in range(1, n_lags + 1):
        X[lag:, lag - 1] = series[:-lag]
    valid = ~np.isnan(X).any(axis=1)
    return X[valid], series[valid]

def walk_forward_split(n_samples, n_splits=5, min_train=100):
    step = max(1, (n_samples - min_train) // n_splits)
    for i in range(n_splits):
        train_end = min_train + i * step
        test_end = min(train_end + step, n_samples)
        if train_end >= n_samples:
            break
        yield slice(0, train_end), slice(train_end, test_end)

np.random.seed(42)
n = 500
t = np.arange(n)

# Rejim değişimi: ilk yarı YAVAŞ artan trend, ikinci yarı HIZLI artan farklı trend
series = np.where(t < 250, 100 + 0.02*t, 100 + 0.02*250 + 0.3*(t-250)) + np.random.normal(0, 3, n)

X, y = make_lag_features(series, n_lags=10)

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y, test_size=0.2, random_state=42)
model_random = Ridge(alpha=1.0)
model_random.fit(X_train_r, y_train_r)
mae_random = mean_absolute_error(y_test_r, model_random.predict(X_test_r))

wf_maes = []
for train_idx, test_idx in walk_forward_split(len(X), n_splits=5, min_train=100):
    model_wf = Ridge(alpha=1.0)
    model_wf.fit(X[train_idx], y[train_idx])
    mae = mean_absolute_error(y[test_idx], model_wf.predict(X[test_idx]))
    wf_maes.append(mae)

print(f"Rejim değişimi olan seride:")
print(f"Rastgele split MAE: {mae_random:.4f}")
print(f"Walk-forward MAE:   {np.mean(wf_maes):.4f}")
print(f"Fark: {np.mean(wf_maes) - mae_random:+.4f}")
