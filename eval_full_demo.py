import random
import math

def train_val_test_split(X, y, train_ratio=0.6, val_ratio=0.2, seed=42):
    random.seed(seed)
    n = len(X)
    indices = list(range(n))
    random.shuffle(indices)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    return ([X[i] for i in train_idx], [y[i] for i in train_idx],
            [X[i] for i in val_idx], [y[i] for i in val_idx],
            [X[i] for i in test_idx], [y[i] for i in test_idx])

def kfold_split(n, k=5, seed=42):
    random.seed(seed)
    indices = list(range(n))
    random.shuffle(indices)
    fold_size = n // k
    folds = []
    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else n
        val_idx = indices[start:end]
        train_idx = indices[:start] + indices[end:]
        folds.append((train_idx, val_idx))
    return folds

def stratified_kfold_split(y, k=5, seed=42):
    random.seed(seed)
    class_indices = {}
    for i, label in enumerate(y):
        class_indices.setdefault(label, []).append(i)
    for label in class_indices:
        random.shuffle(class_indices[label])
    folds = [{"train": [], "val": []} for _ in range(k)]
    for label, indices in class_indices.items():
        fold_size = len(indices) // k
        for i in range(k):
            start = i * fold_size
            end = start + fold_size if i < k - 1 else len(indices)
            val_part = indices[start:end]
            train_part = indices[:start] + indices[end:]
            folds[i]["val"].extend(val_part)
            folds[i]["train"].extend(train_part)
    return [(f["train"], f["val"]) for f in folds]

def cross_validate(X, y, model_fn, k=5, metric_fn=None, stratified=False):
    n = len(X)
    folds = stratified_kfold_split(y, k) if stratified else kfold_split(n, k)
    scores = []
    for train_idx, val_idx in folds:
        X_train = [X[i] for i in train_idx]; y_train = [y[i] for i in train_idx]
        X_val = [X[i] for i in val_idx]; y_val = [y[i] for i in val_idx]
        model = model_fn()
        model.fit(X_train, y_train)
        predictions = [model.predict(x) for x in X_val]
        score = metric_fn(y_val, predictions) if metric_fn else sum(1 for yt,yp in zip(y_val,predictions) if yt==yp)/len(y_val)
        scores.append(score)
    return scores

def confusion_matrix(y_true, y_pred):
    tp = sum(1 for yt,yp in zip(y_true,y_pred) if yt==1 and yp==1)
    tn = sum(1 for yt,yp in zip(y_true,y_pred) if yt==0 and yp==0)
    fp = sum(1 for yt,yp in zip(y_true,y_pred) if yt==0 and yp==1)
    fn = sum(1 for yt,yp in zip(y_true,y_pred) if yt==1 and yp==0)
    return tp, tn, fp, fn

def accuracy(y_true, y_pred):
    tp,tn,fp,fn = confusion_matrix(y_true,y_pred)
    total = tp+tn+fp+fn
    return (tp+tn)/total if total>0 else 0.0

def precision(y_true, y_pred):
    tp,tn,fp,fn = confusion_matrix(y_true,y_pred)
    return tp/(tp+fp) if (tp+fp)>0 else 0.0

def recall(y_true, y_pred):
    tp,tn,fp,fn = confusion_matrix(y_true,y_pred)
    return tp/(tp+fn) if (tp+fn)>0 else 0.0

def f1_score(y_true, y_pred):
    p, r = precision(y_true,y_pred), recall(y_true,y_pred)
    return 2*p*r/(p+r) if (p+r)>0 else 0.0

def roc_curve(y_true, y_scores):
    thresholds = sorted(set(y_scores), reverse=True)
    tpr_list, fpr_list = [], []
    total_positives = sum(y_true)
    total_negatives = len(y_true) - total_positives
    for threshold in thresholds:
        y_pred = [1 if s>=threshold else 0 for s in y_scores]
        tp = sum(1 for yt,yp in zip(y_true,y_pred) if yt==1 and yp==1)
        fp = sum(1 for yt,yp in zip(y_true,y_pred) if yt==0 and yp==1)
        tpr_list.append(tp/total_positives if total_positives>0 else 0.0)
        fpr_list.append(fp/total_negatives if total_negatives>0 else 0.0)
    return fpr_list, tpr_list, thresholds

def auc_roc(y_true, y_scores):
    fpr_list, tpr_list, _ = roc_curve(y_true, y_scores)
    pairs = sorted(zip(fpr_list, tpr_list))
    fpr_sorted = [p[0] for p in pairs]; tpr_sorted = [p[1] for p in pairs]
    area = 0.0
    for i in range(1, len(fpr_sorted)):
        width = fpr_sorted[i] - fpr_sorted[i-1]
        height = (tpr_sorted[i] + tpr_sorted[i-1]) / 2
        area += width * height
    return area

def mse(y_true, y_pred):
    n = len(y_true)
    return sum((yt-yp)**2 for yt,yp in zip(y_true,y_pred))/n

def rmse(y_true, y_pred):
    return math.sqrt(mse(y_true,y_pred))

def mae(y_true, y_pred):
    n = len(y_true)
    return sum(abs(yt-yp) for yt,yp in zip(y_true,y_pred))/n

def r_squared(y_true, y_pred):
    mean_y = sum(y_true)/len(y_true)
    ss_res = sum((yt-yp)**2 for yt,yp in zip(y_true,y_pred))
    ss_tot = sum((yt-mean_y)**2 for yt in y_true)
    return 1.0 - ss_res/ss_tot if ss_tot != 0 else 0.0

def learning_curve(X, y, model_fn, metric_fn, train_sizes=None, val_ratio=0.2, seed=42):
    random.seed(seed)
    n = len(X)
    indices = list(range(n)); random.shuffle(indices)
    val_size = int(n*val_ratio)
    val_idx = indices[:val_size]; pool_idx = indices[val_size:]
    X_val = [X[i] for i in val_idx]; y_val = [y[i] for i in val_idx]
    if train_sizes is None:
        train_sizes = [int(len(pool_idx)*r) for r in [0.1,0.2,0.4,0.6,0.8,1.0]]
    train_scores, val_scores = [], []
    for size in train_sizes:
        subset = pool_idx[:size]
        X_train = [X[i] for i in subset]; y_train = [y[i] for i in subset]
        model = model_fn(); model.fit(X_train, y_train)
        train_pred = [model.predict(x) for x in X_train]
        val_pred = [model.predict(x) for x in X_val]
        train_scores.append(metric_fn(y_train, train_pred))
        val_scores.append(metric_fn(y_val, val_pred))
    return train_sizes, train_scores, val_scores

class SimpleLogistic:
    def __init__(self, lr=0.1, epochs=100):
        self.lr = lr; self.epochs = epochs; self.weights = None; self.bias = 0.0
    def sigmoid(self, z):
        z = max(-500, min(500, z))
        return 1.0/(1.0+math.exp(-z))
    def fit(self, X, y):
        n_features = len(X[0])
        self.weights = [0.0]*n_features; self.bias = 0.0
        for _ in range(self.epochs):
            for xi, yi in zip(X, y):
                z = sum(w*x for w,x in zip(self.weights,xi)) + self.bias
                pred = self.sigmoid(z)
                error = yi - pred
                for j in range(n_features):
                    self.weights[j] += self.lr * error * xi[j]
                self.bias += self.lr * error
    def predict_proba(self, x):
        z = sum(w*xi for w,xi in zip(self.weights,x)) + self.bias
        return self.sigmoid(z)
    def predict(self, x):
        return 1 if self.predict_proba(x) >= 0.5 else 0

def make_classification_data(n=300, seed=42):
    random.seed(seed)
    X, y = [], []
    for _ in range(n):
        x1 = random.gauss(0,1); x2 = random.gauss(0,1)
        label = 1 if (x1+x2+random.gauss(0,0.5)) > 0 else 0
        X.append([x1,x2]); y.append(label)
    return X, y

def make_imbalanced_data(n=300, minority_ratio=0.05, seed=42):
    random.seed(seed)
    X, y = [], []
    for _ in range(n):
        if random.random() < minority_ratio:
            x1 = random.gauss(3,0.5); x2 = random.gauss(3,0.5); label = 1
        else:
            x1 = random.gauss(0,1); x2 = random.gauss(0,1); label = 0
        X.append([x1,x2]); y.append(label)
    return X, y


X_clf, y_clf = make_classification_data(300)

print("=== Train/Validation/Test Split ===")
X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X_clf, y_clf)
print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

model = SimpleLogistic(lr=0.1, epochs=200)
model.fit(X_train, y_train)

print("\n=== Sınıflandırma Metrikleri ===")
y_pred = [model.predict(x) for x in X_test]
tp, tn, fp, fn = confusion_matrix(y_test, y_pred)
print(f"  Confusion matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
print(f"  Accuracy:  {accuracy(y_test, y_pred):.4f}")
print(f"  F1 Score:  {f1_score(y_test, y_pred):.4f}")
y_scores = [model.predict_proba(x) for x in X_test]
print(f"  AUC-ROC:   {auc_roc(y_test, y_scores):.4f}")

print("\n=== K-Fold Cross-Validation (K=5) ===")
cv_scores = cross_validate(X_clf, y_clf, model_fn=lambda: SimpleLogistic(lr=0.1, epochs=200), k=5, metric_fn=accuracy)
mean_cv = sum(cv_scores)/len(cv_scores)
std_cv = math.sqrt(sum((s-mean_cv)**2 for s in cv_scores)/len(cv_scores))
print(f"  Fold skorları: {[round(s,4) for s in cv_scores]}")
print(f"  Ortalama: {mean_cv:.4f} (+/- {std_cv:.4f})")

print("\n=== Dengesiz Veri: Accuracy Neden Yalan Söyler ===")
X_imb, y_imb = make_imbalanced_data(300, minority_ratio=0.05)
positives = sum(y_imb)
print(f"  Sınıf dağılımı: {positives} pozitif, {len(y_imb)-positives} negatif (%{positives/len(y_imb)*100:.1f} pozitif)")
always_negative = [0]*len(y_imb)
print(f"  'Her zaman negatif de' baseline:")
print(f"    Accuracy:  {accuracy(y_imb, always_negative):.4f}")
print(f"    Recall:    {recall(y_imb, always_negative):.4f}")
print(f"    F1 Score:  {f1_score(y_imb, always_negative):.4f}")

print("\n=== Öğrenme Eğrisi (Learning Curve) ===")
sizes, train_sc, val_sc = learning_curve(X_clf, y_clf, model_fn=lambda: SimpleLogistic(lr=0.1, epochs=200), metric_fn=accuracy)
print(f"  {'Boyut':>6} {'Train':>8} {'Val':>8}")
for s, tr, va in zip(sizes, train_sc, val_sc):
    print(f"  {s:>6} {tr:>8.4f} {va:>8.4f}")
