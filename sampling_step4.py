import random

def monte_carlo_pi(n):
    inside = 0
    for _ in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x*x + y*y <= 1:
            inside += 1
    return 4 * inside / n

for n in [100, 1000, 10000, 100000, 1000000]:
    estimate = monte_carlo_pi(n)
    error = abs(estimate - 3.14159265358979)
    print(f"n={n:>8d}  tahmin={estimate:.6f}  hata={error:.6f}")
