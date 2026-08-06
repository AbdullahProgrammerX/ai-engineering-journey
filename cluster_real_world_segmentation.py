import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

np.random.seed(42)

# Gerçekçi müşteri verisi simülasyonu
n_customers = 500

# 4 doğal müşteri tipi kurguluyoruz (ama modelin bunu "bilmediğini" varsayıyoruz)
segments_true = np.random.choice(4, n_customers, p=[0.35, 0.25, 0.25, 0.15])

annual_spending = np.zeros(n_customers)
purchase_frequency = np.zeros(n_customers)
avg_basket_size = np.zeros(n_customers)
days_since_last_purchase = np.zeros(n_customers)

for i in range(n_customers):
    s = segments_true[i]
    if s == 0:  # "Fırsatçı / nadir alışverişçi"
        annual_spending[i] = np.random.normal(150, 40)
        purchase_frequency[i] = np.random.normal(2, 1)
        avg_basket_size[i] = np.random.normal(75, 15)
        days_since_last_purchase[i] = np.random.normal(180, 60)
    elif s == 1:  # "Düzenli, orta harcamalı"
        annual_spending[i] = np.random.normal(800, 100)
        purchase_frequency[i] = np.random.normal(12, 2)
        avg_basket_size[i] = np.random.normal(65, 10)
        days_since_last_purchase[i] = np.random.normal(25, 10)
    elif s == 2:  # "VIP / yüksek değerli"
        annual_spending[i] = np.random.normal(3000, 400)
        purchase_frequency[i] = np.random.normal(20, 3)
        avg_basket_size[i] = np.random.normal(150, 25)
        days_since_last_purchase[i] = np.random.normal(10, 5)
    else:  # "Kayıp riski / eskiden aktif"
        annual_spending[i] = np.random.normal(600, 150)
        purchase_frequency[i] = np.random.normal(5, 2)
        avg_basket_size[i] = np.random.normal(80, 20)
        days_since_last_purchase[i] = np.random.normal(200, 40)

X = np.column_stack([annual_spending, purchase_frequency, avg_basket_size, days_since_last_purchase])
X = np.clip(X, 0, None)  # negatif değer olmasın
feature_names = ["Yıllık harcama", "Alışveriş sıklığı", "Ortalama sepet", "Son alışverişten geçen gün"]

print("=== Müşteri Segmentasyonu -- Gerçek Dünya Senaryosu ===\n")
print(f"Toplam müşteri: {n_customers}\n")

# Ölçekleme -- ZORUNLU, çünkü özellikler çok farklı birimlerde
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Doğru K'yi bul
print("--- Doğru segment sayısını bulma ---")
best_k, best_score = None, -1
for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    print(f"K={k}  silhouette_score={score:.4f}")
    if score > best_score:
        best_score, best_k = score, k

print(f"\nEn iyi segment sayısı: {best_k}\n")

# Final kümeleme
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels = km_final.fit_predict(X_scaled)

print("=== Segment Profilleri (orijinal ölçekte) ===\n")
for cluster_id in range(best_k):
    mask = labels == cluster_id
    count = mask.sum()
    print(f"--- Segment {cluster_id} ({count} müşteri, %{count/n_customers*100:.1f}) ---")
    for j, name in enumerate(feature_names):
        print(f"  {name:30s}: ortalama={X[mask, j].mean():7.1f}")
    print()

print("=== Pazarlama Stratejisi Önerisi ===")
for cluster_id in range(best_k):
    mask = labels == cluster_id
    avg_spending = X[mask, 0].mean()
    avg_recency = X[mask, 3].mean()
    count = mask.sum()

    if avg_spending > 2000:
        strategy = "VIP programı, özel erken erişim, kişisel danışman ata"
    elif avg_recency > 150:
        strategy = "'Seni özledik' kampanyası, geri kazanım indirimi gönder"
    elif avg_spending > 500:
        strategy = "Sadakat puanı, çapraz satış (cross-sell) önerileri"
    else:
        strategy = "Düşük maliyetli otomatik e-posta kampanyaları, minimum yatırım"

    print(f"Segment {cluster_id} ({count} müşteri): {strategy}")
