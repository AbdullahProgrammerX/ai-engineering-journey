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

def walk_forward_split(n_samples, n_splits=5, min_train=50):
    step = max(1, (n_samples - min_train) // n_splits)
    for i in range(n_splits):
        train_end = min_train + i * step
        test_end = min(train_end + step, n_samples)
        if train_end >= n_samples:
            break
        yield slice(0, train_end), slice(train_end, test_end)

# Gerçekçi bir zaman serisi: trend + mevsimsellik + gürültü
np.random.seed(42)
n = 500
t = np.arange(n)
trend = 0.05 * t
seasonality = 10 * np.sin(2 * np.pi * t / 30)
noise = np.random.normal(0, 3, n)
series = 100 + trend + seasonality + noise

X, y = make_lag_features(series, n_lags=10)

print("--- YANLIŞ: Rastgele train/test split (verinin zaman sırasını görmezden geliyor) ---")
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y, test_size=0.2, random_state=42)
model_random = Ridge(alpha=1.0)
model_random.fit(X_train_r, y_train_r)
pred_random = model_random.predict(X_test_r)
mae_random = mean_absolute_error(y_test_r, pred_random)
print(f"Rastgele split MAE: {mae_random:.4f}")

print("\n--- DOĞRU: Walk-forward validation (sadece geçmişle geleceği tahmin et) ---")
wf_maes = []
for train_idx, test_idx in walk_forward_split(len(X), n_splits=5, min_train=100):
    model_wf = Ridge(alpha=1.0)
    model_wf.fit(X[train_idx], y[train_idx])
    pred_wf = model_wf.predict(X[test_idx])
    mae = mean_absolute_error(y[test_idx], pred_wf)
    wf_maes.append(mae)
    print(f"  Fold: train_size={train_idx.stop}, test_size={test_idx.stop-test_idx.start}, MAE={mae:.4f}")

print(f"\nWalk-forward ortalama MAE: {np.mean(wf_maes):.4f}")
print(f"\n=== Karşılaştırma ===")
print(f"Rastgele split MAE (yapay olarak iyi görünüyor): {mae_random:.4f}")
print(f"Walk-forward MAE (gerçekçi):                      {np.mean(wf_maes):.4f}")
print(f"Fark: {np.mean(wf_maes) - mae_random:+.4f}  (walk-forward genelde DAHA YÜKSEK/kötü çıkar -- bu doğru, çünkü gerçekçi)")
