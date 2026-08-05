import numpy as np
from sklearn.neighbors import KNeighborsRegressor

np.random.seed(42)
X = np.sort(np.random.uniform(0, 4*np.pi, 100)).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.normal(0, 0.2, 100)

X_test = np.linspace(0, 4*np.pi, 50).reshape(-1, 1)
y_true = np.sin(X_test).ravel()

print("Ağırlıksız (uniform) vs Ağırlıklı (distance) KNN regresyon karşılaştırması:\n")
for k in [3, 10, 30]:
    knn_uniform = KNeighborsRegressor(n_neighbors=k, weights='uniform')
    knn_uniform.fit(X, y)
    pred_uniform = knn_uniform.predict(X_test)
    mse_uniform = np.mean((pred_uniform - y_true)**2)

    knn_weighted = KNeighborsRegressor(n_neighbors=k, weights='distance')
    knn_weighted.fit(X, y)
    pred_weighted = knn_weighted.predict(X_test)
    mse_weighted = np.mean((pred_weighted - y_true)**2)

    print(f"K={k:>3d}  Uniform MSE={mse_uniform:.4f}   Weighted (distance) MSE={mse_weighted:.4f}")
