import numpy as np
import matplotlib
matplotlib.use('Agg')  # ekran olmadan dosyaya kaydetmek için
import matplotlib.pyplot as plt

def true_function(distance):
    return 10 + 2*distance + 0.15*distance**1.5

def generate_data(n_samples=40, noise_std=3.0, seed=None):
    rng = np.random.RandomState(seed)
    distance = rng.uniform(1, 20, n_samples)
    delivery_time = true_function(distance) + rng.normal(0, noise_std, n_samples)
    return distance, delivery_time

def fit_polynomial(x_train, y_train, degree):
    X = np.column_stack([x_train ** d for d in range(degree + 1)])
    return np.linalg.lstsq(X, y_train, rcond=None)[0]

def predict_polynomial(x, w):
    X = np.column_stack([x ** d for d in range(len(w))])
    return X @ w

x_test = np.linspace(1, 20, 50)
y_true = true_function(x_test)

# --- Grafik 1: Model uyumu (underfit / good / overfit) ---
x_sample, y_sample = generate_data(n_samples=25, seed=7)

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(x_sample, y_sample, color='gray', label='Gerçek teslimat verisi', zorder=3)

for degree, label, color, style in [(1, 'Derece 1 (yetersiz uyum)', 'orange', '--'),
                                      (3, 'Derece 3 (iyi uyum)', 'green', '-'),
                                      (9, 'Derece 9 (aşırı öğrenme)', 'red', ':')]:
    w = fit_polynomial(x_sample, y_sample, degree)
    y_fit = predict_polynomial(x_test, w)
    ax.plot(x_test, np.clip(y_fit, -20, 150), color=color, linestyle=style, linewidth=2, label=label)

ax.set_xlabel('Mesafe (km)')
ax.set_ylabel('Teslimat süresi (dakika)')
ax.set_title('Model Uyumu: Yetersiz vs İyi vs Aşırı Öğrenme')
ax.legend()
ax.set_ylim(-10, 90)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/fit_comparison.png', dpi=120)
print("Kaydedildi: fit_comparison.png")

# --- Grafik 2: Bias-Variance decomposition ---
degrees = [1, 2, 3, 5, 7, 9]
bias_list, var_list, total_list = [], [], []

for degree in degrees:
    predictions = []
    for boot in range(150):
        xt, yt = generate_data(n_samples=40, seed=boot)
        w = fit_polynomial(xt, yt, degree)
        predictions.append(predict_polynomial(x_test, w))
    predictions = np.array(predictions)
    mean_pred = predictions.mean(axis=0)
    bias_sq = np.mean((mean_pred - y_true) ** 2)
    variance = np.mean(predictions.var(axis=0))
    bias_list.append(bias_sq)
    var_list.append(variance)
    total_list.append(bias_sq + variance)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(degrees, bias_list, 'o-', color='blue', label='Bias²', linewidth=2)
ax.plot(degrees, var_list, 's-', color='red', label='Variance', linewidth=2)
ax.plot(degrees, total_list, '^-', color='black', label='Toplam Hata', linewidth=2)
ax.set_yscale('log')
ax.set_xlabel('Polinom Derecesi')
ax.set_ylabel('Hata (log ölçek)')
ax.set_title('Bias-Variance Dengesi: Derece Arttıkça Ne Oluyor?')
ax.legend()
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/bias_variance_curve.png', dpi=120)
print("Kaydedildi: bias_variance_curve.png")

# --- Grafik 3: Öğrenme eğrisi ---
x_test_lc, y_test_lc = generate_data(n_samples=200, seed=999)
sizes = [10, 20, 40, 80, 150, 300]
degree_lc = 7
train_mse_list, test_mse_list = [], []

for n in sizes:
    train_errors, test_errors = [], []
    for seed in range(25):
        xt, yt = generate_data(n_samples=n, seed=seed * 100)
        w = fit_polynomial(xt, yt, degree_lc)
        train_pred = predict_polynomial(xt, w)
        train_errors.append(np.mean((train_pred - yt) ** 2))
        test_pred = predict_polynomial(x_test_lc, w)
        test_errors.append(min(np.mean((test_pred - y_test_lc) ** 2), 200))
    train_mse_list.append(np.mean(train_errors))
    test_mse_list.append(np.mean(test_errors))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(sizes, train_mse_list, 'o-', color='blue', label='Eğitim Hatası', linewidth=2)
ax.plot(sizes, test_mse_list, 's-', color='red', label='Test Hatası', linewidth=2)
ax.set_yscale('log')
ax.set_xlabel('Eğitim Seti Büyüklüğü')
ax.set_ylabel('MSE (log ölçek)')
ax.set_title('Öğrenme Eğrisi: Derece 7 Model (Yüksek Variance)')
ax.legend()
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/learning_curve.png', dpi=120)
print("Kaydedildi: learning_curve.png")

print("\nÜç grafik de /mnt/user-data/outputs/ klasörüne kaydedildi.")
