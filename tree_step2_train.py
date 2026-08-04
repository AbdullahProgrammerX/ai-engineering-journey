import math
import random

def gini_impurity(labels):
    n = len(labels)
    if n == 0:
        return 0.0
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return 1.0 - sum((c / n) ** 2 for c in counts.values())

def information_gain(parent_labels, left_labels, right_labels):
    n = len(parent_labels)
    n_left, n_right = len(left_labels), len(right_labels)
    if n_left == 0 or n_right == 0:
        return 0.0
    parent_impurity = gini_impurity(parent_labels)
    child_impurity = (n_left/n)*gini_impurity(left_labels) + (n_right/n)*gini_impurity(right_labels)
    return parent_impurity - child_impurity

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        self.n_features = len(X[0])
        self.tree = self._build(X, y, depth=0)

    def predict(self, X):
        return [self._predict_one(x, self.tree) for x in X]

    def _build(self, X, y, depth):
        if len(set(y)) == 1:
            return {"leaf": True, "value": y[0]}
        if self.max_depth is not None and depth >= self.max_depth:
            return self._make_leaf(y)
        if len(y) < self.min_samples_split:
            return self._make_leaf(y)

        best_feature, best_threshold, best_gain = self._best_split(X, y)
        if best_feature is None or best_gain <= 0:
            return self._make_leaf(y)

        left_X, left_y, right_X, right_y = self._split_data(X, y, best_feature, best_threshold)
        if len(left_y) < self.min_samples_leaf or len(right_y) < self.min_samples_leaf:
            return self._make_leaf(y)

        return {
            "leaf": False, "feature": best_feature, "threshold": best_threshold,
            "left": self._build(left_X, left_y, depth+1),
            "right": self._build(right_X, right_y, depth+1),
        }

    def _make_leaf(self, y):
        counts = {}
        for label in y:
            counts[label] = counts.get(label, 0) + 1
        return {"leaf": True, "value": max(counts, key=counts.get)}

    def _best_split(self, X, y):
        best_feature, best_threshold, best_gain = None, None, -1.0
        for feature_idx in range(self.n_features):
            values = sorted(set(X[i][feature_idx] for i in range(len(X))))
            if len(values) <= 1:
                continue
            for i in range(len(values)-1):
                threshold = (values[i] + values[i+1]) / 2.0
                left_y = [y[j] for j in range(len(X)) if X[j][feature_idx] <= threshold]
                right_y = [y[j] for j in range(len(X)) if X[j][feature_idx] > threshold]
                if len(left_y) < self.min_samples_leaf or len(right_y) < self.min_samples_leaf:
                    continue
                gain = information_gain(y, left_y, right_y)
                if gain > best_gain:
                    best_gain, best_feature, best_threshold = gain, feature_idx, threshold
        return best_feature, best_threshold, best_gain

    def _split_data(self, X, y, feature, threshold):
        left_X, left_y, right_X, right_y = [], [], [], []
        for i in range(len(X)):
            if X[i][feature] <= threshold:
                left_X.append(X[i]); left_y.append(y[i])
            else:
                right_X.append(X[i]); right_y.append(y[i])
        return left_X, left_y, right_X, right_y

    def _predict_one(self, x, node):
        if node["leaf"]:
            return node["value"]
        if x[node["feature"]] <= node["threshold"]:
            return self._predict_one(x, node["left"])
        return self._predict_one(x, node["right"])


from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)
X, y = X.tolist(), y.tolist()

random.seed(42)
combined = list(zip(X, y))
random.shuffle(combined)
X, y = zip(*combined)
X, y = list(X), list(y)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print("Farklı max_depth değerleriyle eğitim vs test doğruluğu:\n")
for depth in [1, 2, 3, 5, 10, None]:
    tree = DecisionTree(max_depth=depth)
    tree.fit(X_train, y_train)
    train_preds = tree.predict(X_train)
    test_preds = tree.predict(X_test)
    train_acc = sum(1 for p, t in zip(train_preds, y_train) if p == t) / len(y_train)
    test_acc = sum(1 for p, t in zip(test_preds, y_test) if p == t) / len(y_test)
    depth_str = str(depth) if depth else "sınırsız"
    print(f"max_depth={depth_str:>8s}  train_acc={train_acc:.4f}  test_acc={test_acc:.4f}")
