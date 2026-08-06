import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
n_customers = 500
segments_true = np.random.choice(4, n_customers, p=[0.35, 0.25, 0.25, 0.15])

annual_spending = np.zeros(n_customers)
purchase_frequency = np.zeros(n_customers)
avg_basket_size = np.zeros(n_customers)
days_since_last_purchase = np.zeros(n_customers)

for i in range(n_customers):
    s = segments_true[i]
    if s == 0:
        annual_spending[i] = np.random.normal(150, 40)
        purchase_frequency[i] = np.random.normal(2, 1)
        avg_basket_size[i] = np.random.normal(75, 15)
        days_since_last_purchase[i] = np.random.normal(180, 60)
    elif s == 1:
        annual_spending[i] = np.random.normal(800, 100)
        purchase_frequency[i] = np.random.normal(12, 2)
        avg_basket_size[i] = np.random.normal(65, 10)
        days_since_last_purchase[i] = np.random.normal(25, 10)
    elif s == 2:
        annual_spending[i] = np.random.normal(3000, 400)
        purchase_frequency[i] = np.random.normal(20, 3)
        avg_basket_size[i] = np.random.normal(150, 25)
        days_since_last_purchase[i] = np.random.normal(10, 5)
    else:
        annual_spending[i] = np.random.normal(600, 150)
        purchase_frequency[i] = np.random.normal(5, 2)
        avg_basket_size[i] = np.random.normal(80, 20)
        days_since_last_purchase[i] = np.random.normal(200, 40)

X = np.column_stack([annual_spending, purchase_frequency, avg_basket_size, days_since_last_purchase])
X = np.clip(X, 0, None)
feature_names = ["Yıllık harcama", "Alışveriş sıklığı", "Ortalama sepet", "Son alışverişten geçen gün"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

km = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = km.fit_predict(X_scaled)

print("=== K=4 ile ZORLA segmentasyon (iş ihtiyacına göre) ===\n")
for cluster_id in range(4):
    mask = labels == cluster_id
    count = mask.sum()
    print(f"--- Segment {cluster_id} ({count} müşteri) ---")
    for j, name in enumerate(feature_names):
        print(f"  {name:30s}: {X[mask, j].mean():7.1f}")
    print()
