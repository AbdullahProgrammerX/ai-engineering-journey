import numpy as np
from sklearn.ensemble import IsolationForest

def zscore_detect(X, threshold=2.5):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    z = np.abs((X - mean) / std)
    return z.max(axis=1) > threshold

np.random.seed(42)
n = 300

# Normal veri: SIKI bir köşegen (küçük std) -- artık anomaliler göreceli olarak daha belirgin
t = np.random.normal(0, 1.0, n)  # std 3 -> 1'e düşürdük
X_normal = t + np.random.normal(0, 0.2, n)
Y_normal = t + np.random.normal(0, 0.2, n)
normal_data = np.column_stack([X_normal, Y_normal])

anomalies = np.array([
    [1.5, -1.5],
    [-1.2, 1.2],
    [1.0, -1.0],
    [-1.5, 1.5],
])

X = np.vstack([normal_data, anomalies])
y_true = np.array([0]*n + [1]*4)

print("--- Z-score ---")
z_flags = zscore_detect(X, threshold=2.5)
print(f"Yakalanan: {(z_flags & (y_true==1)).sum()}/4, Toplam alarm: {z_flags.sum()}")

print("\n--- Isolation Forest ---")
iso = IsolationForest(n_estimators=200, contamination=4/len(X), random_state=42)
iso_flags = iso.fit_predict(X) == -1
print(f"Yakalanan: {(iso_flags & (y_true==1)).sum()}/4, Toplam alarm: {iso_flags.sum()}")
