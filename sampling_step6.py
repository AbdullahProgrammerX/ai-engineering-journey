import random
import math

random.seed(42)

def gibbs_sampling_2d(conditional_x_given_y, conditional_y_given_x, x0, y0, n_samples, burn_in):
    x, y = x0, y0
    samples = []
    for i in range(n_samples + burn_in):
        x = conditional_x_given_y(y)
        y = conditional_y_given_x(x)
        if i >= burn_in:
            samples.append((x, y))
    return samples

# İki değişkenli normal dağılım, aralarında korelasyon var (rho=0.8)
rho = 0.8

def conditional_x_given_y(y):
    # X | Y=y ~ Normal(rho*y, sqrt(1-rho^2))
    mean = rho * y
    std = math.sqrt(1 - rho**2)
    return random.gauss(mean, std)

def conditional_y_given_x(x):
    mean = rho * x
    std = math.sqrt(1 - rho**2)
    return random.gauss(mean, std)

samples = gibbs_sampling_2d(conditional_x_given_y, conditional_y_given_x, x0=0.0, y0=0.0, n_samples=5000, burn_in=500)

xs = [s[0] for s in samples]
ys = [s[1] for s in samples]

mean_x = sum(xs) / len(xs)
mean_y = sum(ys) / len(ys)
cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / len(xs)
std_x = math.sqrt(sum((x - mean_x)**2 for x in xs) / len(xs))
std_y = math.sqrt(sum((y - mean_y)**2 for y in ys) / len(ys))
sample_corr = cov_xy / (std_x * std_y)

print(f"X ortalaması: {mean_x:.4f}  (beklenen: 0)")
print(f"Y ortalaması: {mean_y:.4f}  (beklenen: 0)")
print(f"Örneklem korelasyonu: {sample_corr:.4f}  (beklenen: {rho})")
