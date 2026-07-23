import numpy as np

np.random.seed(42)
A = np.random.randn(5, 4)

# NumPy'ın kendi SVD fonksiyonu
U, S, Vt = np.linalg.svd(A, full_matrices=False)

print(f"Orijinal matris şekli: {A.shape}")
print(f"U şekli: {U.shape}")
print(f"S (singular values): {np.round(S, 4)}")
print(f"Vt şekli: {Vt.shape}")

# Geri birleştirme -- A = U @ diag(S) @ Vt
A_reconstructed = U @ np.diag(S) @ Vt
error = np.linalg.norm(A - A_reconstructed)
print(f"\nGeri inşa hatası: {error:.10f}  (0'a çok yakın olmalı)")
