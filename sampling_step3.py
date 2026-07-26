import math
import random

def importance_sampling_estimate(f, target_pdf, proposal_pdf, proposal_sample, n):
    total = 0
    for _ in range(n):
        x = proposal_sample()
        w = target_pdf(x) / proposal_pdf(x)
        total += f(x) * w
    return total / n

# Hedef: standart normal dağılım N(0,1)
mu, sigma = 0.0, 1.0

def target_pdf(x):
    return math.exp(-(x - mu)**2 / (2 * sigma**2)) / (sigma * math.sqrt(2 * math.pi))

# Öneri: [-10, 10] arasında düzgün dağılım (normal dağılımı "kapsayan" geniş bir aralık)
def proposal_sample():
    return random.uniform(-10, 10)

def proposal_pdf(x):
    return 1 / 20

# Tahmin etmek istediğimiz: E[X^2] (normal dağılım altında)
def f(x):
    return x ** 2

n = 100000
estimate = importance_sampling_estimate(f, target_pdf, proposal_pdf, proposal_sample, n)

# Bilinen gerçek cevap: E[X^2] = mu^2 + sigma^2
true_value = mu**2 + sigma**2

print(f"Importance sampling tahmini: {estimate:.4f}")
print(f"Bilinen gerçek değer (mu^2 + sigma^2): {true_value:.4f}")
print(f"Fark: {abs(estimate - true_value):.4f}")
