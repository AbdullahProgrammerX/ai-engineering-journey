import numpy as np

np.random.seed(42)
n = 300
t = np.random.normal(0, 1.0, n)
X_normal = t + np.random.normal(0, 0.2, n)
Y_normal = t + np.random.normal(0, 0.2, n)

print(f"Normal veri X aralığı: [{X_normal.min():.2f}, {X_normal.max():.2f}]")
print(f"Normal veri Y aralığı: [{Y_normal.min():.2f}, {Y_normal.max():.2f}]")
print(f"Normal veride X-Y farkının std'si: {(X_normal - Y_normal).std():.4f}")

anomalies = np.array([[1.5, -1.5], [-1.2, 1.2], [1.0, -1.0], [-1.5, 1.5]])
print(f"\nAnomali noktalarında X-Y farkı: {anomalies[:,0] - anomalies[:,1]}")
print("(Bu fark, normal veride görülen X-Y farkından KAT KAT büyük olmalı)")
