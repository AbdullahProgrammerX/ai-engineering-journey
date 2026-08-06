import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=42)

print("Gerçek küme sayısı: 4 (ama biz bunu 'bilmiyormuş' gibi K'yi arayacağız)\n")

print("--- Elbow Method ---")
inertias = []
for k in range(1, 10):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertias.append(km.inertia_)
    print(f"K={k}  inertia={km.inertia_:.2f}")

print("\n--- Silhouette Score ---")
best_k, best_score = None, -1
for k in range(2, 10):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    score = silhouette_score(X, labels)
    print(f"K={k}  silhouette_score={score:.4f}")
    if score > best_score:
        best_score, best_k = score, k

print(f"\nSilhouette Score'a göre en iyi K: {best_k} (skor: {best_score:.4f})")

# Doğru K ile kümeleme
km_final = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = km_final.fit_predict(X)
print(f"\nBulunan küme merkezleri:\n{np.round(km_final.cluster_centers_, 2)}")
