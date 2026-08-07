import numpy as np

def true_function(x):
    return np.sin(1.5 * x) + 0.5 * x

def generate_data(n_samples=30, noise_std=0.5, x_range=(-3, 3), seed=None):
    rng = np.random.RandomState(seed)
    x = rng.uniform(x_range[0], x_range[1], n_samples)
    y = true_function(x) + rng.normal(0, noise_std, n_samples)
    return x, y

def fit_polynomial(x_train, y_train, degree):
    X = np.column_stack([x_train ** d for d in range(degree + 1)])
    return np.linalg.lstsq(X, y_train, rcond=None)[0]

def predict_polynomial(x, w):
    X = np.column_stack([x ** d for d in range(len(w))])
    return X @ w

x_test, y_test = generate_data(n_samples=200, seed=999)

print("--- YÜKSEK VARIANCE model (derece=12): az veride overfitting bekleniyor ---\n")
sizes = [10, 20, 40, 80, 150, 300]
degree = 12
for n in sizes:
    train_errors, test_errors = [], []
    for seed in range(30):
        x_train, y_train = generate_data(n_samples=n, seed=seed * 100)
        w = fit_polynomial(x_train, y_train, degree)
        train_pred = predict_polynomial(x_train, w)
        train_mse = np.mean((train_pred - y_train) ** 2)
        test_pred = predict_polynomial(x_test, w)
        test_mse = np.mean((test_pred - y_test) ** 2)
        train_errors.append(train_mse)
        test_errors.append(test_mse)
    print(f"n={n:>4d}  train_mse={np.mean(train_errors):8.4f}  test_mse={np.mean(test_errors):8.4f}  fark={np.mean(test_errors)-np.mean(train_errors):8.4f}")

print("\n--- YÜKSEK BIAS model (derece=1): daha fazla veri işe yaramamalı ---\n")
degree = 1
for n in sizes:
    train_errors, test_errors = [], []
    for seed in range(30):
        x_train, y_train = generate_data(n_samples=n, seed=seed * 100)
        w = fit_polynomial(x_train, y_train, degree)
        train_pred = predict_polynomial(x_train, w)
        train_mse = np.mean((train_pred - y_train) ** 2)
        test_pred = predict_polynomial(x_test, w)
        test_mse = np.mean((test_pred - y_test) ** 2)
        train_errors.append(train_mse)
        test_errors.append(test_mse)
    print(f"n={n:>4d}  train_mse={np.mean(train_errors):8.4f}  test_mse={np.mean(test_errors):8.4f}  fark={np.mean(test_errors)-np.mean(train_errors):8.4f}")
