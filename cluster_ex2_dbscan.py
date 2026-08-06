import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import make_moons

X, y_true = make_moons(n_samples=300, noise=0.08, random_state=42)

print("--- K-Means (K=2) -- HİLAL ŞEKLİNDE VERİDE BAŞARISIZ OLMALI ---")
km = KMeans(n_clusters=2, random_state=42, n_init=10)
km_labels = km.fit_predict(X)

# Doğruluk için basit bir eşleştirme (etiketler ters de olabilir)
match1 = np.mean(km_labels == y_true)
match2 = np.mean(km_labels == (1 - y_true))
km_accuracy = max(match1, match2)
print(f"K-Means 'doğruluğu' (kabaca): {km_accuracy:.2%}")

print("\n--- DBSCAN -- HİLAL ŞEKLİNİ DOĞRU YAKALAMALI ---")
db = DBSCAN(eps=0.2, min_samples=5)
db_labels = db.fit_predict(X)

n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_noise = list(db_labels).count(-1)
print(f"Bulunan küme sayısı: {n_clusters}")
print(f"Gürültü (outlier) olarak işaretlenen nokta sayısı: {n_noise}")

match1_db = np.mean(db_labels == y_true)
match2_db = np.mean(db_labels == (1 - y_true))
db_accuracy = max(match1_db, match2_db)
print(f"DBSCAN 'doğruluğu' (kabaca): {db_accuracy:.2%}")
