import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA as SklearnPCA

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        cov_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]
        self.components = eigenvectors[:, :self.n_components].T
        self.eigenvalues = eigenvalues[:self.n_components]
        total_var = np.sum(eigenvalues)
        self.explained_variance_ratio_ = self.eigenvalues / total_var
        return self

    def transform(self, X):
        return (X - self.mean) @ self.components.T

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


print("MNIST veri seti indiriliyor (biraz zaman alabilir)...")
mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
X_mnist = mnist.data[:5000].astype(float)
y_mnist = mnist.target[:5000].astype(int)

pca_mnist = PCA(n_components=50)
X_pca50 = pca_mnist.fit_transform(X_mnist)
print(f"50 bileşen varyansın {sum(pca_mnist.explained_variance_ratio_):.2%}'ini yakalıyor")

pca_2d = PCA(n_components=2)
X_pca2d = pca_2d.fit_transform(X_mnist)
print(f"2 bileşen varyansın {sum(pca_2d.explained_variance_ratio_):.2%}'ini yakalıyor")

print("\n--- sklearn ile karşılaştırma ---")
sklearn_pca = SklearnPCA(n_components=2)
X_sklearn_pca = sklearn_pca.fit_transform(X_mnist)

print(f"Bizim PCA açıklanan varyans:     {pca_2d.explained_variance_ratio_}")
print(f"Sklearn PCA açıklanan varyans:   {sklearn_pca.explained_variance_ratio_}")

diff = np.abs(np.abs(X_pca2d) - np.abs(X_sklearn_pca))
print(f"Maksimum mutlak fark: {diff.max():.10f}")
