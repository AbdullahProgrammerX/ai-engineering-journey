import numpy as np

np.random.seed(42)
vectors = [np.random.randn(50) for _ in range(5)]

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

best_pair = None
best_score = -1
for i in range(5):
    for j in range(i + 1, 5):
        sim = cosine_sim(vectors[i], vectors[j])
        print(f"vektör {i} - vektör {j}: benzerlik = {sim:.4f}")
        if sim > best_score:
            best_score = sim
            best_pair = (i, j)

print(f"\nEn benzer çift: {best_pair} (benzerlik: {best_score:.4f})")
