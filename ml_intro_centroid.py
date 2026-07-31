import numpy as np

class NearestCentroid:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.centroids = np.array([
            X[y == c].mean(axis=0) for c in self.classes
        ])

    def predict(self, X):
        distances = np.array([
            np.sqrt(((X - c) ** 2).sum(axis=1))
            for c in self.centroids
        ])
        return self.classes[distances.argmin(axis=0)]


rng = np.random.RandomState(42)
X_class0 = rng.randn(100, 2) + np.array([1.0, 1.0])
X_class1 = rng.randn(100, 2) + np.array([-1.0, -1.0])
X = np.vstack([X_class0, X_class1])
y = np.array([0] * 100 + [1] * 100)

# Basit train/test bölmesi
shuffle_idx = rng.permutation(len(X))
split = int(0.7 * len(X))
train_idx, test_idx = shuffle_idx[:split], shuffle_idx[split:]
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Modelimizi eğit
clf = NearestCentroid()
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
accuracy = np.mean(preds == y_test)

print(f"Öğrenilen merkezler (centroids): \n{clf.centroids}")
print(f"\nNearest Centroid doğruluğu: {accuracy:.2%}")

# Rastgele baseline ile karşılaştır
baseline_preds = rng.choice([0, 1], size=len(y_test))
baseline_acc = np.mean(baseline_preds == y_test)
print(f"Rastgele baseline doğruluğu: {baseline_acc:.2%}")

print(f"\nModelin baseline'a göre iyileşmesi: {(accuracy - baseline_acc)*100:.1f} puan")
