import numpy as np

def eigenvalues_2x2(matrix):
    a, b = matrix[0]
    c, d = matrix[1]
    trace = a + d
    det = a * d - b * c
    discriminant = trace ** 2 - 4 * det
    sqrt_disc = discriminant ** 0.5
    return ((trace + sqrt_disc) / 2, (trace - sqrt_disc) / 2)

A = [[4, 2], [1, 3]]

print("--- Elle hesaplama (karakteristik denklem) ---")
print("Matris: [[4,2],[1,3]]")
print("Karakteristik denklem: λ² - (a+d)λ + (ad-bc) = 0")
print("λ² - 7λ + (12-2) = 0")
print("λ² - 7λ + 10 = 0")
print("(λ-5)(λ-2) = 0  =>  λ = 5 veya λ = 2")

print("\n--- Kendi fonksiyonumuzla doğrulama ---")
vals = eigenvalues_2x2(A)
print(f"Eigenvalues: {vals[0]}, {vals[1]}")

print("\n--- NumPy ile doğrulama ---")
np_vals, np_vecs = np.linalg.eig(np.array(A, dtype=float))
print(f"NumPy eigenvalues: {np_vals}")
