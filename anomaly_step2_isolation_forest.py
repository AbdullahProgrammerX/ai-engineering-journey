import numpy as np
from sklearn.ensemble import IsolationForest as SklearnIF

class IsolationTree:
    def __init__(self, max_depth):
        self.max_depth = max_depth

    def fit(self, X, depth=0):
        n, p = X.shape
        if depth >= self.max_depth or n <= 1:
            self.is_leaf = True
            self.size = n
            return self
        self.is_leaf = False
        self.feature = np.random.randint(p)
        x_min, x_max = X[:, self.feature].min(), X[:, self.feature].max()
        if x_min == x_max:
            self.is_leaf = True
            self.size = n
            return self
        self.threshold = np.random.uniform(x_min, x_max)
        left_mask = X[:, self.feature] < self.threshold
        self.left = IsolationTree(self.max_depth).fit(X[left_mask], depth + 1)
        self.right = IsolationTree(self.max_depth).fit(X[~left_mask], depth + 1)
        return self

    def path_length(self, x, depth=0):
        if self.is_leaf:
            return depth + c_factor(self.size)
        if x[self.feature] < self.threshold:
            return self.left.path_length(x, depth + 1)
        return self.right.path_length(x, depth + 1)

def c_factor(n):
    if n <= 1:
        return 0
    return 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n

class IsolationForestScratch:
    def __init__(self, n_estimators=100, max_samples=256, seed=42):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.trees = []
        np.random.seed(seed)

    def fit(self, X):
        sample_size = min(self.max_samples, X.shape[0])
        self.sample_size = sample_size
        max_depth = int(np.ceil(np.log2(sample_size)))
        for _ in range(self.n_estimators):
            idx = np.random.choice(X.shape[0], size=sample_size, replace=False)
            tree = IsolationTree(max_depth=max_depth)
            tree.fit(X[idx])
            self.trees.append(tree)
        return self

    def anomaly_score(self, X):
        scores = np.zeros(len(X))
        for i, x in enumerate(X):
            avg_path = np.mean([tree.path_length(x) for tree in self.trees])
            scores[i] = 2.0 ** (-avg_path / c_factor(self.sample_size))
        return scores


np.random.seed(42)
normal_data = np.random.normal(0, 1, (200, 2))
outliers = np.array([[8, 8], [-7, 7], [9, -6], [-8, -8]])
X = np.vstack([normal_data, outliers])
y_true = np.array([0]*200 + [1]*4)

our_if = IsolationForestScratch(n_estimators=100, max_samples=256, seed=42)
our_if.fit(X)
our_scores = our_if.anomaly_score(X)

sklearn_if = SklearnIF(n_estimators=100, contamination=0.02, random_state=42)
sklearn_if.fit(X)
sklearn_scores = -sklearn_if.score_samples(X)  # sklearn ters işaretli döndürüyor

print("--- Bizim Isolation Forest ---")
print(f"Ortalama anomali skoru (normal): {our_scores[:200].mean():.4f}")
print(f"Ortalama anomali skoru (aykırı): {our_scores[200:].mean():.4f}")
print("(Aykırı değerler HER ZAMAN daha yüksek skor almalı -- daha kolay 'izole' oluyorlar)\n")

print("--- Sklearn Isolation Forest ---")
print(f"Ortalama anomali skoru (normal): {sklearn_scores[:200].mean():.4f}")
print(f"Ortalama anomali skoru (aykırı): {sklearn_scores[200:].mean():.4f}")

correlation = np.corrcoef(our_scores, sklearn_scores)[0, 1]
print(f"\nBizim skorlar ile sklearn skorları arasındaki korelasyon: {correlation:.4f}")
