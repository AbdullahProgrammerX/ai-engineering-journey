import numpy as np

def gram_schmidt(vectors):
    basis = []
    for v in vectors:
        w = v.copy().astype(float)
        for b in basis:
            w -= np.dot(v, b) * b
        norm = np.linalg.norm(w)
        if norm > 1e-10:
            basis.append(w / norm)
    return basis

vecs = [np.array([1, 1, 0]), np.array([1, 0, 1]), np.array([0, 1, 1])]
basis = gram_schmidt(vecs)

print("Ortonormal baz:")
for b in basis:
    print(b)

print("\nDoğrulama:")
for i in range(len(basis)):
    for j in range(i + 1, len(basis)):
        dot = np.dot(basis[i], basis[j])
        print(f"  b{i} . b{j} = {dot:.10f} (0 olmalı)")

for i, b in enumerate(basis):
    print(f"  ||b{i}|| = {np.linalg.norm(b):.10f} (1 olmalı)")
