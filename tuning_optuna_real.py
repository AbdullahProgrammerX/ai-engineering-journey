import optuna
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split

optuna.logging.set_verbosity(optuna.logging.WARNING)  # log gürültüsünü azalt

data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("=== Optuna ile RandomForest Hiperparametre Optimizasyonu ===\n")
print(f"Veri seti: Breast Cancer Wisconsin ({X.shape[0]} örnek, {X.shape[1]} özellik)\n")

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 2, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
    }
    model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")
    return scores.mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, show_progress_bar=False)

print(f"En iyi CV AUC-ROC skoru: {study.best_value:.4f}")
print(f"En iyi parametreler: {study.best_params}\n")

# Varsayılan parametrelerle karşılaştırma
default_model = RandomForestClassifier(random_state=42, n_jobs=-1)
default_scores = cross_val_score(default_model, X_train, y_train, cv=5, scoring="roc_auc")
print(f"Varsayılan parametrelerle CV AUC-ROC: {default_scores.mean():.4f}")
print(f"Optuna ile iyileşme: {study.best_value - default_scores.mean():+.4f}\n")

# Test setinde final değerlendirme
best_model = RandomForestClassifier(**study.best_params, random_state=42, n_jobs=-1)
best_model.fit(X_train, y_train)
from sklearn.metrics import roc_auc_score
test_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
print(f"Test setinde final AUC-ROC: {test_auc:.4f}")

# Hiperparametre önem sırası
print("\n=== Hangi Hiperparametre En Önemli? ===")
importances = optuna.importance.get_param_importances(study)
for param, imp in importances.items():
    bar = "#" * int(imp * 50)
    print(f"  {param:20s} {imp:.4f}  {bar}")
