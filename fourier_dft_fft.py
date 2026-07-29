import numpy as np
import time

def dft_naive(x):
    N = len(x)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    return X

np.random.seed(42)
x = np.random.randn(64)

X_dft = dft_naive(x)
X_fft = np.fft.fft(x)

print(f"Maksimum fark (DFT vs FFT): {np.max(np.abs(X_dft - X_fft)):.2e}")
print("(1e-10'dan küçük olmalı -- matematiksel olarak birebir aynı sonuç)\n")

print("--- Hız karşılaştırması ---")
for N in [256, 512, 1024, 2048]:
    x = np.random.randn(N)

    start = time.perf_counter()
    dft_naive(x)
    t_dft = time.perf_counter() - start

    start = time.perf_counter()
    np.fft.fft(x)
    t_fft = time.perf_counter() - start

    print(f"N={N:>5d}  DFT: {t_dft*1000:8.2f}ms  FFT: {t_fft*1000:6.4f}ms  Oran: {t_dft/t_fft:.0f}x daha hızlı")
