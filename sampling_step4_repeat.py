import random

def monte_carlo_pi(n):
    inside = 0
    for _ in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x*x + y*y <= 1:
            inside += 1
    return 4 * inside / n

true_pi = 3.14159265358979

for n in [10000, 100000, 1000000]:
    errors = []
    for _ in range(10):  # her n için 10 kere tekrarla
        estimate = monte_carlo_pi(n)
        errors.append(abs(estimate - true_pi))
    avg_error = sum(errors) / len(errors)
    print(f"n={n:>8d}  10 denemenin ortalama hatası: {avg_error:.6f}")
