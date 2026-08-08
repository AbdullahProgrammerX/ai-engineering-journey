import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class GradientBoostingScratch:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_pred = None

    def fit(self, X, y):
        self.initial_pred = np.mean(y)
        current_pred = np.full(len(y), self.initial_pred)
        self.residual_history = []

        for i in range(self.n_estimators):
            residuals = y - current_pred
            self.residual_history.append(np.mean(residuals ** 2))

            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            update = tree.predict(X)
            current_pred += self.lr * update
            self.trees.append(tree)

    def predict(self, X):
        pred = np.full(X.shape[0], self.initial_pred)
        for tree in self.trees:
            pred += self.lr * tree.predict(X)
        return pred


X, y = make_regression(n_samples=300, n_features=5, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gb = GradientBoostingScratch(n_estimators=100, learning_rate=0.1, max_depth=3)
gb.fit(X_train, y_train)

print("Kalan hata (residual MSE), her ağaç eklendikçe nasıl azalıyor:\n")
for i in [0, 9, 24, 49, 74, 99]:
    print(f"  Ağaç {i+1:>3d}: residual_mse={gb.residual_history[i]:.2f}")

train_pred = gb.predict(X_train)
test_pred = gb.predict(X_test)
print(f"\nEğitim MSE: {mean_squared_error(y_train, train_pred):.2f}")
print(f"Test MSE:  {mean_squared_error(y_test, test_pred):.2f}")
