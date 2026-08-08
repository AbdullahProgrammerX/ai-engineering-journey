import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score

X, y = make_classification(n_samples=2000, n_features=20, n_informative=15,
                             n_redundant=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("=== Gerçek Dünya Karşılaştırması: Hangi Ensemble Yöntemi Ne Zaman Kazanır ===\n")

methods = {
    "Random Forest (Bagging, variance azaltır)": RandomForestClassifier(n_estimators=100, random_state=42),
    "AdaBoost (bias azaltır)": AdaBoostClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting (bias azaltır)": GradientBoostingClassifier(n_estimators=100, random_state=42),
}

results = {}
for name, model in methods.items():
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    test_acc = model.score(X_test, y_test)
    cv_scores = cross_val_score(model, X, y, cv=5)

    results[name] = (test_acc, cv_scores.mean(), cv_scores.std(), train_time)
    print(f"{name}")
    print(f"  Test doğruluğu: {test_acc:.4f}")
    print(f"  CV ortalaması:  {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"  Eğitim süresi:  {train_time:.3f} saniye\n")

try:
    import xgboost as xgb
    print("=== Bonus: XGBoost karşılaştırması ===\n")
    xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
    start = time.time()
    xgb_model.fit(X_train, y_train)
    xgb_time = time.time() - start
    xgb_acc = xgb_model.score(X_test, y_test)
    print(f"XGBoost test doğruluğu: {xgb_acc:.4f}")
    print(f"XGBoost eğitim süresi: {xgb_time:.3f} saniye")
except ImportError:
    print("XGBoost kurulu değil. Kurmak için: uv pip install xgboost")
