import numpy as np

def condition_number(A):
    U, S, Vt = np.linalg.svd(A)
    return S[0] / S[-1]

print("--- İyi durumlu (well-conditioned) matris ---")
A_good = np.array([[2, 0], [0, 3]], dtype=float)
print(f"Matris:\n{A_good}")
print(f"Condition number: {condition_number(A_good):.4f}")

print("\n--- Kötü durumlu (ill-conditioned) matris ---")
A_bad = np.array([[1, 1], [1, 1.0001]], dtype=float)
print(f"Matris:\n{A_bad}")
print(f"Condition number: {condition_number(A_bad):.4f}")

print("\n--- Neden önemli: küçük bir b değişikliği, çözümü nasıl etkiliyor? ---")
b1 = np.array([2.0, 2.0001])
b2 = np.array([2.0, 2.0002])  # b'de çok küçük bir değişiklik

x1_good = np.linalg.solve(A_good[:2,:2] if A_good.shape==(2,2) else A_good, b1[:2])
x1_bad = np.linalg.solve(A_bad, b1)
x2_bad = np.linalg.solve(A_bad, b2)

print(f"Kötü durumlu sistemde b'yi %0.005 değiştirince:")
print(f"  x (b1 ile): {x1_bad}")
print(f"  x (b2 ile): {x2_bad}")
print(f"  Çözümdeki fark: {np.abs(x2_bad - x1_bad)}  <- çok büyük, orantısız!")
