import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

class DecisionStump:
    def __init__(self):
        self.feature_idx = None
        self.threshold = None
        self.polarity = 1
        self.alpha = None

    def fit(self, X, y, weights):
        n_samples, n_features = X.shape
        best_error = float("inf")
        for f in range(n_features):
            thresholds = np.unique(X[:, f])
            for thresh in thresholds:
                for polarity in [1, -1]:
                    pred = np.ones(n_samples)
                    pred[polarity * X[:, f] < polarity * thresh] = -1
                    error = np.sum(weights[pred != y])
                    if error < best_error:
                        best_error = error
                        self.feature_idx = f
                        self.threshold = thresh
                        self.polarity = polarity

    def predict(self, X):
        n = X.shape[0]
        pred = np.ones(n)
        idx = self.polarity * X[:, self.feature_idx] < self.polarity * self.threshold
        pred[idx] = -1
        return pred


class AdaBoostScratch:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.stumps = []
        self.alphas = []

    def fit(self, X, y):
        n = X.shape[0]
        weights = np.full(n, 1 / n)
        train_acc_history = []

        for i in range(self.n_estimators):
            stump = DecisionStump()
            stump.fit(X, y, weights)
            pred = stump.predict(X)

            err = np.sum(weights[pred != y])
            err = np.clip(err, 1e-10, 1 - 1e-10)

            alpha = 0.5 * np.log((1 - err) / err)
            weights *= np.exp(-alpha * y * pred)
            weights /= weights.sum()

            stump.alpha = alpha
            self.stumps.append(stump)
            self.alphas.append(alpha)

            current_pred = self.predict(X)
            acc = np.mean(current_pred == y)
            train_acc_history.append(acc)

        return train_acc_history

    def predict(self, X):
        total = sum(a * s.predict(X) for a, s in zip(self.alphas, self.stumps))
        return np.sign(total)


X, y = make_classification(n_samples=300, n_features=5, n_informative=3, random_state=42)
y = np.where(y == 0, -1, 1)  # AdaBoost -1/+1 etiketleme kullanır

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ada = AdaBoostScratch(n_estimators=50)
train_acc_history = ada.fit(X_train, y_train)

test_pred = ada.predict(X_test)
test_acc = np.mean(test_pred == y_test)

print("Eğitim doğruluğu, tur arttıkça (her tur bir zayıf öğrenici ekleniyor):\n")
for i in [0, 4, 9, 19, 29, 49]:
    print(f"  Tur {i+1:>2d}: train_acc={train_acc_history[i]:.4f}")

print(f"\nSon test doğruluğu: {test_acc:.4f}")
print(f"Kullanılan zayıf öğrenici (stump) sayısı: {len(ada.stumps)}")
print(f"İlk 5 stump'ın alpha (güven) değerleri: {[round(a,3) for a in ada.alphas[:5]]}")
