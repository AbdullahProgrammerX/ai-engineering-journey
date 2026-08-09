import optuna
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Optuna çalışıyor (30 deneme, birkaç saniye sürmeli)...\n")

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 2, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 15),
    }
    model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring="roc_auc")
    return scores.mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

print(f"En iyi CV AUC-ROC: {study.best_value:.4f}")
print(f"En iyi parametreler: {study.best_params}")

default_model = RandomForestClassifier(random_state=42, n_jobs=-1)
default_scores = cross_val_score(default_model, X_train, y_train, cv=3, scoring="roc_auc")
print(f"\nVarsayılan CV AUC-ROC: {default_scores.mean():.4f}")
print(f"İyileşme: {study.best_value - default_scores.mean():+.4f}")
