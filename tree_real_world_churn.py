import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

np.random.seed(42)

# Gerçekçi bir müşteri kaybı (churn) veri seti simülasyonu
n_customers = 1000

# Özellikler
tenure_months = np.random.randint(1, 72, n_customers)  # kaç aydır müşteri
monthly_charge = np.random.uniform(20, 120, n_customers)  # aylık ücret
support_tickets = np.random.poisson(2, n_customers)  # destek talebi sayısı
usage_hours_per_week = np.random.uniform(0, 40, n_customers)  # haftalık kullanım
has_premium = np.random.choice([0, 1], n_customers, p=[0.7, 0.3])

# Churn (kayıp) mantığı: düşük kullanım + kısa süre + çok destek talebi -> kayıp riski yüksek
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
feature_names = ["Üyelik süresi (ay)", "Aylık ücret", "Destek talebi sayısı", "Haftalık kullanım (saat)", "Premium üye mi"]

X_train, X_test, y_train, y_test = train_test_split(X, churn, test_size=0.2, random_state=42)

print("=== Müşteri Kaybı (Churn) Tahmini -- Gerçek Dünya Senaryosu ===\n")
print(f"Toplam müşteri: {n_customers}, Kayıp oranı: {churn.mean():.1%}\n")

# Tekil ağaç
dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(X_train, y_train)
print(f"Decision Tree (max_depth=4) doğruluğu: {dt.score(X_test, y_test):.4f}")

# Random Forest
rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
rf.fit(X_train, y_train)
print(f"Random Forest doğruluğu: {rf.score(X_test, y_test):.4f}\n")

print("--- Hangi özellik, müşteri kaybını en çok belirliyor? ---")
importances = sorted(zip(feature_names, rf.feature_importances_), key=lambda x: -x[1])
for name, imp in importances:
    bar = "#" * int(imp * 50)
    print(f"  {name:28s} {imp:.4f}  {bar}")

print("\n--- Detaylı sınıflandırma raporu ---")
y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=["Kalıcı", "Kayıp (Churn)"]))

print("--- Örnek müşteri profilleri üzerinde tahmin ---")
sample_customers = [
    ([2, 95, 6, 2, 0], "Yeni üye, pahalı paket, çok şikayet, az kullanım, premium değil"),
    ([48, 60, 1, 25, 1], "Uzun süreli sadık müşteri, normal ücret, az şikayet, çok kullanım, premium"),
    ([12, 70, 3, 10, 0], "Orta seviye, belirsiz profil"),
]

for features, description in sample_customers:
    prob = rf.predict_proba([features])[0][1]
    print(f"\n  {description}")
    print(f"    Kayıp (churn) olasılığı: {prob:.1%}")

print("\n=== İş Kararı ===")
print("Şirket, kayıp riski %50'nin üzerindeki müşterilere otomatik olarak")
print("indirim/özel teklif e-postası gönderebilir -- bu model, hangi müşterilere")
print("öncelik verileceğini belirlemek için doğrudan kullanılabilir.")
