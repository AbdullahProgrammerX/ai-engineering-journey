import math
import random

random.seed(42)

def metropolis_hastings(target_log_pdf, proposal_sample, proposal_log_pdf, x0, n_samples, burn_in):
    samples = []
    x = x0
    accepted = 0
    for i in range(n_samples + burn_in):
        x_new = proposal_sample(x)
        log_alpha = (target_log_pdf(x_new) + proposal_log_pdf(x, x_new)
                     - target_log_pdf(x) - proposal_log_pdf(x_new, x))
        if math.log(random.random()) < log_alpha:
            x = x_new
            accepted += 1
        if i >= burn_in:
            samples.append(x)
    return samples, accepted / (n_samples + burn_in)

# Hedef: iki Gaussian'ın karışımı (bimodal -- iki tepeli dağılım)
def target_log_pdf(x):
    def gauss_pdf(x, mu, sigma):
        return math.exp(-(x - mu)**2 / (2 * sigma**2)) / (sigma * math.sqrt(2 * math.pi))
    p = 0.5 * gauss_pdf(x, -3, 1) + 0.5 * gauss_pdf(x, 3, 1)
    return math.log(p) if p > 0 else -float('inf')

# Öneri: mevcut noktanın etrafında küçük bir adım (random walk)
def proposal_sample(x):
    return x + random.gauss(0, 1.5)

def proposal_log_pdf(x_new, x_old):
    return 0  # simetrik random walk, oranlar birbirini götürür

samples, accept_rate = metropolis_hastings(
    target_log_pdf, proposal_sample, proposal_log_pdf,
    x0=0.0, n_samples=10000, burn_in=1000
)

print(f"Kabul oranı: {accept_rate:.2%}")
print(f"Örneklerin ortalaması: {sum(samples)/len(samples):.4f}  (beklenen: ~0, çünkü iki tepe simetrik)")

# Histogram -- bimodal (iki tepeli) yapıyı görmeliyiz
print("\nHistogram (-7'den 7'ye):")
bins = [0] * 28
for x in samples:
    bin_idx = int((x + 7) / 14 * 28)
    bin_idx = max(0, min(27, bin_idx))
    bins[bin_idx] += 1
for i, count in enumerate(bins):
    bin_start = -7 + i * 0.5
    print(f"{bin_start:5.1f}: {'#' * (count // 30)}")
