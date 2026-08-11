import numpy as np
from sklearn.linear_model import Ridge
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
series = 100 + 0.05*t + 10*np.sin(2*np.pi*t/30) + np.random.normal(0, 3, n)

X, y = make_lag_features(series, n_lags=10)

print("=== Baseline'lar vs Gerçek Model (walk-forward validation ile) ===\n")

persistence_maes, seasonal_naive_maes, model_maes = [], [], []

for train_idx, test_idx in walk_forward_split(len(X), n_splits=5, min_train=100):
    y_train, y_test = y[train_idx], y[test_idx]
    X_train, X_test = X[train_idx], X[test_idx]

    # Baseline 1: Persistence (yarın = bugün)
    persistence_pred = X_test[:, 0]  # lag-1 sütunu = "dünkü" değer
    persistence_maes.append(mean_absolute_error(y_test, persistence_pred))

    # Baseline 2: Seasonal naive (30 gün önceki değer -- eğer lag'lerde varsa)
    # Bu örnekte sadece 10 lag var, o yüzden basitçe hareketli ortalama kullanalım
    moving_avg_pred = X_test.mean(axis=1)
    seasonal_naive_maes.append(mean_absolute_error(y_test, moving_avg_pred))

    # Gerçek model
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    model_pred = model.predict(X_test)
    model_maes.append(mean_absolute_error(y_test, model_pred))

print(f"Persistence baseline (dünkü değer) ortalama MAE: {np.mean(persistence_maes):.4f}")
print(f"Hareketli ortalama baseline MAE:                  {np.mean(seasonal_naive_maes):.4f}")
print(f"Ridge modeli MAE:                                 {np.mean(model_maes):.4f}")

print(f"\n=== Sonuç ===")
if np.mean(model_maes) < np.mean(persistence_maes):
    print("Model, persistence baseline'ı GEÇİYOR -- gerçek bir örüntü öğrenmiş.")
else:
    print("UYARI: Model, basit 'dünkü değer' tahmininden bile KÖTÜ -- bir şeyler yanlış!")
