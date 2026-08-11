import numpy as np
from sklearn.naive_bayes import MultinomialNB as SklearnMNB

class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        classes = np.unique(y)
        n_classes = len(classes)
        n_features = X.shape[1]

        self.classes_ = classes
        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.class_log_prior_[i] = np.log(X_c.shape[0] / X.shape[0])
            counts = X_c.sum(axis=0) + self.alpha
            self.feature_log_prob_[i] = np.log(counts / counts.sum())

        return self

    def predict_log_proba(self, X):
        return X @ self.feature_log_prob_.T + self.class_log_prior_

    def predict(self, X):
        log_proba = self.predict_log_proba(X)
        return self.classes_[np.argmax(log_proba, axis=1)]


# Sentetik "bag-of-words" verisi: tech vs sports makaleleri
np.random.seed(42)
n_samples = 400
n_words = 200

X = np.zeros((n_samples, n_words))
y = np.zeros(n_samples, dtype=int)

for i in range(n_samples):
    is_tech = i < n_samples // 2
    y[i] = 0 if is_tech else 1
    if is_tech:
        X[i, :40] = np.random.poisson(3, 40)      # tech kelimeleri sık
        X[i, 80:120] = np.random.poisson(0.3, 40)  # spor kelimeleri az
    else:
        X[i, :40] = np.random.poisson(0.3, 40)
        X[i, 80:120] = np.random.poisson(3, 40)
    X[i, 40:80] = np.random.poisson(1, 40)  # nötr kelimeler her ikisinde de orta

shuffle_idx = np.random.permutation(n_samples)
X, y = X[shuffle_idx], y[shuffle_idx]
split = int(0.8 * n_samples)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

our_nb = MultinomialNB(alpha=1.0)
our_nb.fit(X_train, y_train)
our_pred = our_nb.predict(X_test)
our_acc = np.mean(our_pred == y_test)

sklearn_nb = SklearnMNB(alpha=1.0)
sklearn_nb.fit(X_train, y_train)
sklearn_acc = sklearn_nb.score(X_test, y_test)

print(f"Bizim MultinomialNB doğruluğu: {our_acc:.4f}")
print(f"Sklearn MultinomialNB doğruluğu: {sklearn_acc:.4f}")
print(f"\nÖğrenilen log olasılıkları birbirine yakın mı: {np.allclose(our_nb.feature_log_prob_, sklearn_nb.feature_log_prob_, atol=1e-6)}")
