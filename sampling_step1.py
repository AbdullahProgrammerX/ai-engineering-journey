import math
import random

def sample_uniform(a, b):
    return a + (b - a) * random.random()

def sample_exponential_inverse_cdf(lam):
    u = random.random()
    return -math.log(u) / lam

# 10.000 üstel (exponential) örnek üret
lam = 2.0
samples = [sample_exponential_inverse_cdf(lam) for _ in range(10000)]

sample_mean = sum(samples) / len(samples)
expected_mean = 1 / lam

print(f"Lambda = {lam}")
print(f"Örneklem ortalaması: {sample_mean:.4f}")
print(f"Beklenen ortalama (1/lambda): {expected_mean:.4f}")
print(f"Fark: {abs(sample_mean - expected_mean):.4f}")
