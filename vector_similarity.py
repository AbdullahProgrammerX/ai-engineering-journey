import numpy as np

def cosine_similarity_matrix(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    X_normalized = X / norms
    return X_normalized @ X_normalized.T

np.random.seed(42)
embeddings = np.random.randn(1000, 768)  # 1000 "belge", her biri 768 boyutlu (BERT gibi)

sim_matrix = cosine_similarity_matrix(embeddings)

query_idx = 0
similarities = sim_matrix[query_idx]
top_k = np.argsort(similarities)[::-1][1:6]  # kendisi hariç en benzer 5
print(f"0. öğeye en benzer 5 öğe: {top_k}")
print(f"Benzerlik skorları: {similarities[top_k]}")
