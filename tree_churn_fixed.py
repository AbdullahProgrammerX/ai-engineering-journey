import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

np.random.seed(42)
n_customers = 1000
tenure_months = np.random.randint(1, 72, n_customers)
monthly_charge = np.random.uniform(20, 120, n_customers)
support_tickets = np.random.poisson(2, n_customers)
usage_hours_per_week = np.random.uniform(0, 40, n_customers)
has_premium = np.random.choice([0, 1], n_customers, p=[0.7, 0.3])

churn_probability = (
    0.4 * (tenure_months < 6).astype(float) +
    0.3 * (usage_hours_per_week < 5).astype(float) +
    0.2 * (support_tickets > 4).astype(float) +
    0.15 * (monthly_charge > 90).astype(float) -
    0.2 * has_premium
)
churn_probability = np.clip(churn_probability, 0.02, 0.95)
churn = (np.random.random(n_customers) < churn_probability).astype(int)

X = np.column_stack([tenure_months, monthly_charge, support_tickets, usage_hours_per_week, has_premium])
X_train, X_test, y_train, y_test = train_test_split(X, churn, test_size=0.2, random_state=42)

print("=== ÇÖZÜM: class_weight='balanced' kullanarak azınlık sınıfına daha çok önem ver ===\n")
rf_balanced = RandomForestClassifier(n_estimators=200, max_depth=6, class_weight='balanced', random_state=42)
rf_balanced.fit(X_train, y_train)
y_pred_balanced = rf_balanced.predict(X_test)

print(classification_report(y_test, y_pred_balanced, target_names=["Kalıcı", "Kayıp (Churn)"]))

print("--- Threshold düşürerek recall'ı zorla artırma (0.5 yerine 0.3) ---")
probs = rf_balanced.predict_proba(X_test)[:, 1]
y_pred_lower_threshold = (probs >= 0.3).astype(int)
print(classification_report(y_test, y_pred_lower_threshold, target_names=["Kalıcı", "Kayıp (Churn)"]))
