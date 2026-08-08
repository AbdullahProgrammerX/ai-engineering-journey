import numpy as np
import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

np.random.seed(42)

# Gerçekçi kredi başvurusu verisi simülasyonu
n_applicants = 5000

age = np.random.normal(40, 12, n_applicants).clip(18, 75)
annual_income = np.random.lognormal(10.5, 0.5, n_applicants)
loan_amount = np.random.lognormal(9.5, 0.6, n_applicants)
credit_score = np.random.normal(650, 80, n_applicants).clip(300, 850)
debt_to_income = np.random.beta(2, 5, n_applicants) * 60
years_employed = np.random.exponential(5, n_applicants).clip(0, 40)
num_credit_lines = np.random.poisson(4, n_applicants)
prior_defaults = np.random.poisson(0.15, n_applicants)

# Temerrüt (default) olasılığı -- gerçekçi risk faktörleri
default_logit = (
    -3.0
    - 0.01 * (credit_score - 650)
    + 0.03 * debt_to_income
    + 0.8 * prior_defaults
    - 0.05 * years_employed
    + 0.3 * (loan_amount / annual_income)
    + np.random.normal(0, 0.5, n_applicants)
)
default_prob = 1 / (1 + np.exp(-default_logit))
is_default = (np.random.random(n_applicants) < default_prob).astype(int)

X = np.column_stack([age, annual_income, loan_amount, credit_score,
                      debt_to_income, years_employed, num_credit_lines, prior_defaults])
feature_names = ["Yaş", "Yıllık gelir", "Kredi tutarı", "Kredi skoru",
                  "Borç/gelir oranı", "Çalışma yılı", "Kredi hattı sayısı", "Önceki temerrüt"]
y = is_default

print("=== Kredi Temerrüt Tahmini -- XGBoost vs LightGBM ===\n")
print(f"Toplam başvuru: {n_applicants}")
print(f"Temerrüt oranı: %{y.mean()*100:.2f} ({y.sum()} vaka)\n")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

results = {}

# --- XGBoost ---
import xgboost as xgb
print("--- XGBoost ---")
start = time.time()
xgb_model = xgb.XGBClassifier(
    n_estimators=200, learning_rate=0.05, max_depth=5,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),  # dengesiz veri için
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
results['XGBoost'] = (xgb_time, roc_auc_score(y_test, xgb_proba))

# --- LightGBM ---
import lightgbm as lgb
print("\n--- LightGBM ---")
start = time.time()
lgb_model = lgb.LGBMClassifier(
    n_estimators=200, learning_rate=0.05, max_depth=5,
    class_weight='balanced', random_state=42, verbose=-1
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
results['LightGBM'] = (lgb_time, roc_auc_score(y_test, lgb_proba))

# --- Karşılaştırma özeti ---
print("\n=== Özet Karşılaştırma ===")
print(f"{'Model':<12} {'Süre (s)':>10} {'AUC-ROC':>10}")
for name, (t, auc) in results.items():
    print(f"{name:<12} {t:>10.3f} {auc:>10.4f}")

# --- Özellik önem sırası (feature importance) ---
print("\n=== XGBoost -- Hangi özellik temerrüt riskini en çok belirliyor? ===")
importances = sorted(zip(feature_names, xgb_model.feature_importances_), key=lambda x: -x[1])
for name, imp in importances:
    bar = "#" * int(imp * 60)
    print(f"  {name:20s} {imp:.4f}  {bar}")

# --- Cross-validation ile güvenilirlik testi ---
print("\n=== 5-Fold Cross-Validation (AUC-ROC) ===")
xgb_cv = cross_val_score(xgb_model, X, y, cv=5, scoring='roc_auc')
lgb_cv = cross_val_score(lgb_model, X, y, cv=5, scoring='roc_auc')
print(f"XGBoost:  {xgb_cv.mean():.4f} (+/- {xgb_cv.std():.4f})")
print(f"LightGBM: {lgb_cv.mean():.4f} (+/- {lgb_cv.std():.4f})")

# --- İş kararı: hangi başvuruları manuel incelemeye gönder ---
print("\n=== İş Kararı: Risk Skoruna Göre Öncelik ===")
high_risk_threshold = 0.5
high_risk_count = (xgb_proba >= high_risk_threshold).sum()
print(f"Yüksek riskli ({high_risk_threshold}+) başvuru sayısı: {high_risk_count}/{len(X_test)}")
print("Banka bu başvuruları otomatik reddetmek yerine, insan kredi uzmanına")
print("yönlendirip manuel inceleme yaptırabilir -- model, önceliklendirme aracı olarak kullanılıyor.")
