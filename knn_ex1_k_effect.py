import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

X, y = make_blobs(n_samples=300, centers=3, cluster_std=1.5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("K arttıkça, train doğruluğu düşer ama test doğruluğu genelde önce artar sonra düşer:\n")
for k in [1, 5, 15, 30, 100, len(X_train)]:
    knn = KNeighborsClassifier(n_neighbors=min(k, len(X_train)))
    knn.fit(X_train, y_train)
    train_acc = knn.score(X_train, y_train)
    test_acc = knn.score(X_test, y_test)
    print(f"K={k:>4d}  train_acc={train_acc:.4f}  test_acc={test_acc:.4f}")
