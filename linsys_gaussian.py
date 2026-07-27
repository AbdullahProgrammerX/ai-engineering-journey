import numpy as np

def gaussian_elimination(A, b):
    n = len(b)
    Ab = np.hstack([A.astype(float), b.reshape(-1, 1).astype(float)])

    for k in range(n):
        max_row = k + np.argmax(np.abs(Ab[k:, k]))
        Ab[[k, max_row]] = Ab[[max_row, k]]

        if abs(Ab[k, k]) < 1e-12:
            raise ValueError(f"Matrix is singular or nearly singular at pivot {k}")

        for i in range(k + 1, n):
            m = Ab[i, k] / Ab[k, k]
            Ab[i, k:] -= m * Ab[k, k:]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, -1] - Ab[i, i+1:n] @ x[i+1:n]) / Ab[i, i]

    return x


A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
b = np.array([6, 15, 27])

x_ours = gaussian_elimination(A, b)
x_numpy = np.linalg.solve(A, b)

print(f"Bizim çözümümüz: {x_ours}")
print(f"NumPy çözümü:    {x_numpy}")
print(f"Maksimum fark:   {np.max(np.abs(x_ours - x_numpy)):.2e}")

# Doğrulama: Ax = b tutuyor mu?
print(f"\nDoğrulama -- A @ x = {A @ x_ours}  (b ile aynı olmalı: {b})")
