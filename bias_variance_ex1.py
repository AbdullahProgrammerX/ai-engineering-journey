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
    w = np.linalg.lstsq(X, y_train, rcond=None)[0]
    return w

def predict_polynomial(x, w):
    X = np.column_stack([x ** d for d in range(len(w))])
    return X @ w

x_test = np.linspace(-3, 3, 50)
y_true = true_function(x_test)

print("Derece arttıkça: Bias azalır ama Variance artar (klasik U-eğrisi)\n")
print(f"{'Derece':>7} {'Bias^2':>10} {'Variance':>10} {'Toplam Hata':>12}")
print("-" * 45)

for degree in [1, 2, 3, 5, 8, 12, 15]:
    predictions = []
    for boot in range(200):
        x_train, y_train = generate_data(n_samples=30, seed=boot)
        w = fit_polynomial(x_train, y_train, degree)
        pred = predict_polynomial(x_test, w)
        predictions.append(pred)

    predictions = np.array(predictions)
    mean_pred = predictions.mean(axis=0)
    bias_sq = np.mean((mean_pred - y_true) ** 2)
    variance = np.mean(predictions.var(axis=0))
    total_error = np.mean(np.mean((predictions - y_true) ** 2, axis=1))

    print(f"{degree:>7} {bias_sq:>10.4f} {variance:>10.4f} {total_error:>12.4f}")
