import numpy as np
import pandas as pd
import time
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

print("Adult Census Income veri seti indiriliyor (gerçek ABD nüfus sayımı verisi, ~49K kayıt)...\n")
data = fetch_openml(name="adult", version=2, as_frame=True, parser="auto")
df = data.frame

print(f"Toplam kayıt: {len(df)}")
print(f"Özellik sayısı: {df.shape[1] - 1}")
print(f"\nİlk birkaç kayıt:\n{df.head(3)}\n")

# Hedef değişken: gelir >50K mi?
target_col = 'class'
y = (df[target_col] == '>50K').astype(int)
X_raw = df.drop(columns=[target_col])

print(f"Gelir >$50K oranı: %{y.mean()*100:.2f}\n")

# Kategorik sütunları encode et
X = X_raw.copy()
categorical_cols = X.select_dtypes(include=['category', 'object']).columns
for col in categorical_cols:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# Eksik değerleri median ile doldur
X = X.fillna(X.median(numeric_only=True))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print(f"Eğitim: {len(X_train)}, Test: {len(X_test)}\n")

results = {}

# --- XGBoost ---
import xgboost as xgb
print("--- XGBoost ---")
start = time.time()
xgb_model = xgb.XGBClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=6,
    eval_metric='logloss', random_state=42
)
xgb_model.fit(X_train, y_train)
xgb_time = time.time() - start
xgb_pred = xgb_model.predict(X_test)
xgb_proba = xgb_model.predict_proba(X_test)[:, 1]

print(f"Eğitim süresi: {xgb_time:.3f}s")
print(f"Accuracy:  {accuracy_score(y_test, xgb_pred):.4f}")
print(f"Precision: {precision_score(y_test, xgb_pred):.4f}")
print(f"Recall:    {recall_score(y_test, xgb_pred):.4f}")
print(f"F1:        {f1_score(y_test, xgb_pred):.4f}")
print(f"AUC-ROC:   {roc_auc_score(y_test, xgb_proba):.4f}")

# --- LightGBM ---
import lightgbm as lgb
print("\n--- LightGBM ---")
start = time.time()
lgb_model = lgb.LGBMClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=6,
    random_state=42, verbose=-1
)
lgb_model.fit(X_train, y_train)
lgb_time = time.time() - start
lgb_pred = lgb_model.predict(X_test)
lgb_proba = lgb_model.predict_proba(X_test)[:, 1]

print(f"Eğitim süresi: {lgb_time:.3f}s")
print(f"Accuracy:  {accuracy_score(y_test, lgb_pred):.4f}")
print(f"Precision: {precision_score(y_test, lgb_pred):.4f}")
print(f"Recall:    {recall_score(y_test, lgb_pred):.4f}")
print(f"F1:        {f1_score(y_test, lgb_pred):.4f}")
print(f"AUC-ROC:   {roc_auc_score(y_test, lgb_proba):.4f}")

print(f"\n=== Hız Karşılaştırması ===")
print(f"XGBoost:  {xgb_time:.3f}s")
print(f"LightGBM: {lgb_time:.3f}s")
print(f"LightGBM, XGBoost'tan {xgb_time/lgb_time:.2f}x daha hızlı")

# --- Feature importance ---
print(f"\n=== XGBoost -- Geliri en çok belirleyen faktörler ===")
importances = sorted(zip(X.columns, xgb_model.feature_importances_), key=lambda x: -x[1])
for name, imp in importances[:8]:
    bar = "#" * int(imp * 60)
    print(f"  {name:20s} {imp:.4f}  {bar}")

# --- Cross-validation ---
print(f"\n=== 5-Fold Cross-Validation (AUC-ROC) ===")
xgb_cv = cross_val_score(xgb_model, X, y, cv=5, scoring='roc_auc', n_jobs=-1)
lgb_cv = cross_val_score(lgb_model, X, y, cv=5, scoring='roc_auc', n_jobs=-1)
print(f"XGBoost:  {xgb_cv.mean():.4f} (+/- {xgb_cv.std():.4f})")
print(f"LightGBM: {lgb_cv.mean():.4f} (+/- {lgb_cv.std():.4f})")
