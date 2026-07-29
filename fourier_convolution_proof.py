import numpy as np

x = np.array([1, 2, 3, 4, 0, 0, 0, 0], dtype=float)
h = np.array([1, 1, 1, 0, 0, 0, 0, 0], dtype=float)

# Dairesel (circular) convolution -- doğrudan döngüyle
N = len(x)
y_direct = np.zeros(N)
for n in range(N):
    for k in range(N):
        y_direct[n] += x[k] * h[(n - k) % N]

# FFT tabanlı convolution
X = np.fft.fft(x)
H = np.fft.fft(h)
Y = X * H
y_fft = np.fft.ifft(Y).real

print(f"Doğrudan (nested loop) dairesel convolution: {y_direct}")
print(f"FFT tabanlı convolution:                     {np.round(y_fft, 6)}")
print(f"Maksimum fark: {np.max(np.abs(y_direct - y_fft)):.2e}")

print("\n--- Doğrusal (linear) convolution için sıfır-doldurma ---")
x2 = np.array([1, 2, 3, 4], dtype=float)
h2 = np.array([1, 1, 1], dtype=float)
y_linear_direct = np.convolve(x2, h2)

pad_len = len(x2) + len(h2) - 1
X2 = np.fft.fft(x2, pad_len)
H2 = np.fft.fft(h2, pad_len)
y_linear_fft = np.fft.ifft(X2 * H2).real

print(f"np.convolve sonucu:    {y_linear_direct}")
print(f"FFT (sıfır-dolgulu):   {np.round(y_linear_fft, 6)}")
