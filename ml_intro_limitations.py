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

print("--- Durum 1: Normal (tek 'blob' sınıflar) -- BAŞARILI ---")
X0 = rng.randn(100, 2) + np.array([2, 2])
X1 = rng.randn(100, 2) + np.array([-2, -2])
X = np.vstack([X0, X1])
y = np.array([0]*100 + [1]*100)
clf = NearestCentroid()
clf.fit(X, y)
acc = np.mean(clf.predict(X) == y)
print(f"Doğruluk: {acc:.2%}")

print("\n--- Durum 2: Bir sınıf, diğerinin ETRAFINI SARIYOR (halka şeklinde) -- BAŞARISIZ ---")
theta = rng.uniform(0, 2*np.pi, 100)
radius_outer = 5 + rng.randn(100) * 0.3
X_ring = np.column_stack([radius_outer * np.cos(theta), radius_outer * np.sin(theta)])
X_center = rng.randn(100, 2) * 0.5  # merkeze yakın küçük bir küme
X2 = np.vstack([X_ring, X_center])
y2 = np.array([0]*100 + [1]*100)
clf2 = NearestCentroid()
clf2.fit(X2, y2)
acc2 = np.mean(clf2.predict(X2) == y2)
print(f"Doğruluk: {acc2:.2%}  <- çok düşük, çünkü halkanın merkezi, iç kümenin merkeziyle AYNI NOKTADA")

print("\n--- Durum 3: Özellikler çok farklı ölçekte -- BAŞARISIZ ---")
X0_scaled = rng.randn(100, 2) + np.array([1, 1])
X0_scaled[:, 0] *= 1000  # ilk özelliği 1000 kat büyüt
X1_scaled = rng.randn(100, 2) + np.array([-1, -1])
X1_scaled[:, 0] *= 1000
X3 = np.vstack([X0_scaled, X1_scaled])
y3 = np.array([0]*100 + [1]*100)
clf3 = NearestCentroid()
clf3.fit(X3, y3)
acc3 = np.mean(clf3.predict(X3) == y3)
print(f"Doğruluk: {acc3:.2%}")
print("(İkinci özellik neredeyse tamamen görmezden geliniyor çünkü mesafe hesabında")
print(" ilk özelliğin büyük ölçeği baskın çıkıyor)")
