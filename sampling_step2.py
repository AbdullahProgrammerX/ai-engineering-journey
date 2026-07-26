import math
import random

def rejection_sample(target_pdf, proposal_sample, proposal_pdf, M):
    while True:
        x = proposal_sample()
        u = random.random()
        if u < target_pdf(x) / (M * proposal_pdf(x)):
            return x

# Hedef: kesik (truncated) normal dağılım, [-2, 2] aralığında
def target_pdf(x):
    if x < -2 or x > 2:
        return 0
    return math.exp(-x**2 / 2) / math.sqrt(2 * math.pi)

# Öneri (proposal): [-2, 2] arasında düzgün dağılım
def proposal_sample():
    return random.uniform(-2, 2)

def proposal_pdf(x):
    return 1 / 4  # [-2,2] aralığında düzgün dağılımın yoğunluğu

M = 1.0  # target_pdf'in maksimum değeri / proposal_pdf'in değeri (yeterince büyük olmalı)
# target_pdf'in maksimumu x=0'da: 1/sqrt(2*pi) ≈ 0.3989
# proposal_pdf = 0.25, M = 0.3989/0.25 ≈ 1.6 güvenli olur
M = 1.6

samples = [rejection_sample(target_pdf, proposal_sample, proposal_pdf, M) for _ in range(5000)]

sample_mean = sum(samples) / len(samples)
sample_var = sum((x - sample_mean)**2 for x in samples) / len(samples)

print(f"Üretilen örnek sayısı: {len(samples)}")
print(f"Örneklem ortalaması: {sample_mean:.4f}  (beklenen: ~0, çünkü simetrik)")
print(f"Örneklem varyansı: {sample_var:.4f}")

# Basit histogram (metin tabanlı)
print("\nHistogram (-2'den 2'ye):")
bins = [0] * 20
for x in samples:
    bin_idx = min(19, int((x + 2) / 4 * 20))
    bins[bin_idx] += 1
for i, count in enumerate(bins):
    bin_start = -2 + i * 0.2
    print(f"{bin_start:5.1f}: {'#' * (count // 20)}")
