import numpy as np

def true_function(x):
    return np.sin(1.5 * x) + 0.5 * x

def generate_data(n_samples=30, noise_std=0.5, x_range=(-3, 3), seed=None):
    rng = np.random.RandomState(seed)
    x = rng.uniform(x_range[0], x_range[1], n_samples)
    y = true_function(x) + rng.normal(0, noise_std, n_samples)
    return x, y

def fit_polynomial_ridge(x_train, y_train, degree, lam=0.0):
    X = np.column_stack([x_train ** d for d in range(degree + 1)])
    if lam > 0:
        penalty = lam * np.eye(X.shape[1])
        penalty[0, 0] = 0
        w = np.linalg.solve(X.T @ X + penalty, X.T @ y_train)
    else:
        w = np.linalg.lstsq(X, y_train, rcond=None)[0]
    return w

def predict_polynomial(x, w):
    X = np.column_stack([x ** d for d in range(len(w))])
    return X @ w

x_test = np.linspace(-3, 3, 50)
y_true = true_function(x_test)

print("Çok daha geniş alpha aralığıyla:\n")
print(f"{'Alpha':>12} {'Bias^2':>10} {'Variance':>12} {'Toplam':>12}")

for alpha in [100, 1000, 10000, 100000, 1000000]:
    predictions = []
    for boot in range(100):
        x_train, y_train = generate_data(n_samples=30, seed=boot)
        w = fit_polynomial_ridge(x_train, y_train, degree=15, lam=alpha)
        pred = predict_polynomial(x_test, w)
        predictions.append(pred)

    predictions = np.array(predictions)
    mean_pred = predictions.mean(axis=0)
    bias_sq = np.mean((mean_pred - y_true) ** 2)
    variance = np.mean(predictions.var(axis=0))
    total = bias_sq + variance

    print(f"{alpha:>12} {bias_sq:>10.4f} {variance:>12.4f} {total:>12.4f}")
