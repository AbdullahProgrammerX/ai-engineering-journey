import numpy as np

def metropolis_hastings(target_log_prob, proposal_std, x0, n_samples, seed=None):
    rng = np.random.RandomState(seed)
    x = np.array(x0, dtype=float)
    samples = [x.copy()]
    accepted = 0
    for _ in range(n_samples - 1):
        x_proposed = x + rng.randn(*x.shape) * proposal_std
        log_ratio = target_log_prob(x_proposed) - target_log_prob(x)
        if np.log(rng.rand()) < log_ratio:
            x = x_proposed
            accepted += 1
        samples.append(x.copy())
    acceptance_rate = accepted / (n_samples - 1)
    return np.array(samples), acceptance_rate

# Hedef: standart normal dağılım (log yoğunluk)
def target_log_prob(x):
    return -0.5 * np.sum(x**2)

print("--- Farklı proposal_std değerleriyle kabul oranı ---")
for std in [0.1, 0.5, 1.0, 2.0, 5.0]:
    samples, rate = metropolis_hastings(target_log_prob, proposal_std=std, x0=[0.0], n_samples=5000, seed=42)
    print(f"proposal_std={std:.1f}  kabul_oranı={rate:.2%}  örneklem_ortalaması={samples.mean():.4f}")

print("\n--- Markov zinciri karışma zamanı analizi (spektral açıklık) ---")
P = np.array([[0.9, 0.1], [0.3, 0.7]])
eigenvalues = np.linalg.eigvals(P)
spectral_gap = 1 - sorted(np.abs(eigenvalues))[-2]
print(f"Özdeğerler: {eigenvalues}")
print(f"Spektral açıklık: {spectral_gap:.4f}")
print(f"Yaklaşık karışma zamanı: {1/spectral_gap:.1f} adım")
