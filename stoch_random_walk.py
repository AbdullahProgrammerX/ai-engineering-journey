import numpy as np

def random_walk_1d(n_steps, seed=None):
    rng = np.random.RandomState(seed)
    steps = rng.choice([-1, 1], size=n_steps)
    positions = np.concatenate([[0], np.cumsum(steps)])
    return positions

rng = np.random.RandomState(42)
walk = np.cumsum(rng.choice([-1, 1], size=10000))
print(f"Son pozisyon: {walk[-1]}")
print(f"Beklenen mesafe (sqrt(10000)): {np.sqrt(10000):.1f}")
print(f"Gerçek mesafe: {abs(walk[-1])}")

print("\n--- 1000 farklı yürüyüşün son pozisyonlarının istatistiği ---")
final_positions = []
for seed in range(1000):
    w = random_walk_1d(10000, seed=seed)
    final_positions.append(w[-1])

final_positions = np.array(final_positions)
print(f"Ortalama: {final_positions.mean():.4f}  (beklenen: 0)")
print(f"Standart sapma: {final_positions.std():.4f}  (beklenen: sqrt(10000)=100)")
