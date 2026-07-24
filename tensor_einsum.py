import numpy as np

print("--- Dot product ---")
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])
dot = np.einsum("i,i->", a, b)
print(f"a . b = {dot}  (beklenen: {np.dot(a,b)})")

print("\n--- Matris çarpımı ---")
A = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
B = np.array([[7, 8, 9], [10, 11, 12]], dtype=float)
matmul = np.einsum("ik,kj->ij", A, B)
print(f"A@B (einsum) =\n{matmul}")
print(f"A@B (normal) =\n{A @ B}")

print("\n--- Batch matris çarpımı (transformer'larda çok sık kullanılır) ---")
batch_A = np.random.randn(4, 3, 5)
batch_B = np.random.randn(4, 5, 2)
batch_mm = np.einsum("bij,bjk->bik", batch_A, batch_B)
print(f"batch_A {batch_A.shape} @ batch_B {batch_B.shape} -> {batch_mm.shape}")
