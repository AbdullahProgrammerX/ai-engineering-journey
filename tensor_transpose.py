import numpy as np

print("--- Transpose (2 eksen değişimi) ---")
mat = np.arange(6).reshape(2, 3)
print(f"Orijinal (2,3):\n{mat}")
tr = mat.transpose(1, 0)  # ya da mat.T
print(f"Transpose (3,2):\n{tr}")

print("\n--- Permute (tüm eksenleri yeniden sırala) ---")
t4d = np.arange(24).reshape(1, 2, 3, 4)
print(f"Orijinal şekil (NCHW gibi): {t4d.shape}")
perm = t4d.transpose(0, 2, 3, 1)  # NCHW -> NHWC
print(f"Permute sonrası şekil (NHWC gibi): {perm.shape}")

print("\n--- Contiguous uyarısı ---")
print(f"Transpose sonrası bellek sıralı mı (C_CONTIGUOUS): {tr.flags['C_CONTIGUOUS']}")
print("PyTorch'ta transpose sonrası .view() çalışmaz, .reshape() veya .contiguous() gerekir.")
