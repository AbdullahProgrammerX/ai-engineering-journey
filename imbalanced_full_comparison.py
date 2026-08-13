import numpy as np

def make_imbalanced_data(n_majority=950, n_minority=50, seed=42):
    rng = np.random.RandomState(seed)
    X_maj = rng.randn(n_majority, 2) * 1.0 + np.array([0.0, 0.0])
    X_min = rng.randn(n_minority, 2) * 0.8 + np.array([2.5, 2.5])
    X = np.vstack([X_maj, X_min])
    y = np.concatenate([np.zeros(n_majority), np.ones(n_minority)])
    shuffle_idx = rng.permutation(len(y))
    return X[shuffle_idx], y[shuffle_idx]

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def find_k_neighbors(X, idx, k):
    distances = []
    for i in range(len(X)):
        if i == idx:
            continue
        d = euclidean_distance(X[idx], X[i])
        distances.append((i, d))
    distances.sort(key=lambda x: x[1])
    return [d[0] for d in distances[:k]]

def smote(X_minority, k=5, n_synthetic=100, seed=42):
    rng = np.random.RandomState(seed)
    n_samples = len(X_minority)
    k = min(k, n_samples - 1)
    synthetic = []
    for _ in range(n_synthetic):
        idx = rng.randint(0, n_samples)
        neighbors = find_k_neighbors(X_minority, idx, k)
        neighbor_idx = neighbors[rng.randint(0, len(neighbors))]
        t = rng.random()
        new_point = X_minority[idx] + t * (X_minority[neighbor_idx] - X_minority[idx])
        synthetic.append(new_point)
    return np.array(synthetic)

def random_oversample(X, y, seed=42):
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y, return_counts=True)
    max_count = counts.max()
    X_resampled, y_resampled = list(X), list(y)
    for cls, count in zip(classes, counts):
        if count < max_count:
            cls_indices = np.where(y == cls)[0]
            n_needed = max_count - count
            chosen = rng.choice(cls_indices, size=n_needed, replace=True)
            X_resampled.extend(X[chosen])
            y_resampled.extend(y[chosen])
    X_out, y_out = np.array(X_resampled), np.array(y_resampled)
    shuffle = rng.permutation(len(y_out))
    return X_out[shuffle], y_out[shuffle]

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def logistic_regression_weighted(X, y, weights, lr=0.01, epochs=200):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0
    for _ in range(epochs):
        z = X @ w + b
        pred = sigmoid(z)
        error = pred - y
        weighted_error = error * weights
        gradient_w = (X.T @ weighted_error) / n_samples
        gradient_b = np.mean(weighted_error)
        w -= lr * gradient_w
        b -= lr * gradient_b
    return w, b

def compute_class_weights(y):
    classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(classes)
    weight_map = {}
    for cls, count in zip(classes, counts):
        weight_map[cls] = n_samples / (n_classes * count)
    return np.array([weight_map[yi] for yi in y])

def find_optimal_threshold(y_true, y_probs, metric="f1"):
    best_threshold, best_score = 0.5, -1.0
    for threshold in np.arange(0.05, 0.96, 0.01):
        y_pred = (y_probs >= threshold).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        if score > best_score:
            best_score, best_threshold = score, threshold
    return best_threshold, best_score

def confusion_matrix_values(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return tp, tn, fp, fn

def compute_metrics(y_true, y_pred):
    tp, tn, fp, fn = confusion_matrix_values(y_true, y_pred)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "mcc": mcc}


X, y = make_imbalanced_data(950, 50, seed=42)
split_train = int(0.6 * len(y))
split_val = int(0.8 * len(y))
X_train, X_val, X_test = X[:split_train], X[split_train:split_val], X[split_val:]
y_train, y_val, y_test = y[:split_train], y[split_train:split_val], y[split_val:]

print(f"Eğitim: {len(y_train)} ({y_train.sum():.0f} azınlık), Test: {len(y_test)} ({y_test.sum():.0f} azınlık)\n")

results = {}

# 1. Baseline
w, b = logistic_regression_weighted(X_train, y_train, np.ones(len(y_train)), lr=0.1, epochs=300)
preds = (sigmoid(X_test @ w + b) >= 0.5).astype(int)
results["Baseline (işlemsiz)"] = compute_metrics(y_test, preds)

# 2. Random Oversampling
X_over, y_over = random_oversample(X_train, y_train)
w, b = logistic_regression_weighted(X_over, y_over, np.ones(len(y_over)), lr=0.1, epochs=300)
preds = (sigmoid(X_test @ w + b) >= 0.5).astype(int)
results["Random Oversampling"] = compute_metrics(y_test, preds)

# 3. SMOTE
minority_mask = y_train == 1
X_minority = X_train[minority_mask]
n_needed = len(y_train) - 2 * int(minority_mask.sum())
synthetic = smote(X_minority, k=5, n_synthetic=max(n_needed, 10))
X_smote = np.vstack([X_train, synthetic])
y_smote = np.concatenate([y_train, np.ones(len(synthetic))])
w, b = logistic_regression_weighted(X_smote, y_smote, np.ones(len(y_smote)), lr=0.1, epochs=300)
preds = (sigmoid(X_test @ w + b) >= 0.5).astype(int)
results["SMOTE"] = compute_metrics(y_test, preds)

# 4. Class Weights
sample_weights = compute_class_weights(y_train)
w_cw, b_cw = logistic_regression_weighted(X_train, y_train, sample_weights, lr=0.1, epochs=300)
probs_cw = sigmoid(X_test @ w_cw + b_cw)
preds = (probs_cw >= 0.5).astype(int)
results["Class Weights"] = compute_metrics(y_test, preds)

# 5. Class Weights + Threshold Tuning
probs_val = sigmoid(X_val @ w_cw + b_cw)
best_thresh, best_f1_val = find_optimal_threshold(y_val, probs_val, metric="f1")
preds_thresh = (probs_cw >= best_thresh).astype(int)
results[f"Class Weights + Threshold({best_thresh:.2f})"] = compute_metrics(y_test, preds_thresh)

print(f"{'Yöntem':<35} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>7} {'MCC':>7}")
print("-" * 80)
for name, m in results.items():
    print(f"{name:<35} {m['accuracy']:>9.4f} {m['precision']:>10.4f} {m['recall']:>8.4f} {m['f1']:>7.4f} {m['mcc']:>7.4f}")
