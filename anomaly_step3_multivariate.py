import numpy as np
from sklearn.ensemble import IsolationForest

def zscore_detect(X, threshold=3.0):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    z = np.abs((X - mean) / std)
    return z.max(axis=1) > threshold

np.random.seed(42)
n = 300

# Normal veri: X ve Y GÜÇLÜ KORELASYONLU (bir çapraz çizgi boyunca kümeleniyor)
t = np.random.normal(0, 3, n)
X_normal = t + np.random.normal(0, 0.3, n)
Y_normal = t + np.random.normal(0, 0.3, n)
normal_data = np.column_stack([X_normal, Y_normal])

# Anomali: HER İKİ özellik de tek başına normal aralıkta, ama KORELASYONU BOZUYOR
# (çapraz çizginin tamamen dışında, ama X ve Y'nin marjinal dağılımı normal görünüyor)
anomalies = np.array([
    [3.0, -3.0],   # X yüksek, Y düşük -- korelasyonu bozuyor
    [-2.5, 2.5],
    [2.0, -2.0],
    [-3.0, 3.0],
])

X = np.vstack([normal_data, anomalies])
y_true = np.array([0]*n + [1]*4)

print("--- Z-score (her özelliği AYRI AYRI kontrol ediyor) ---")
z_flags = zscore_detect(X, threshold=2.5)
caught_z = (z_flags & (y_true == 1)).sum()
print(f"Yakalanan gerçek anomali: {caught_z}/4")
print(f"Toplam alarm: {z_flags.sum()}")
print("(Anomaliler her tek özellikte 'normal aralıkta' göründüğü için Z-score bunları KAÇIRABİLİR)\n")

print("--- Isolation Forest (özellikler arası İLİŞKİYİ de görebiliyor) ---")
iso = IsolationForest(n_estimators=100, contamination=4/len(X), random_state=42)
iso_preds = iso.fit_predict(X)
iso_flags = iso_preds == -1
caught_iso = (iso_flags & (y_true == 1)).sum()
print(f"Yakalanan gerçek anomali: {caught_iso}/4")
print(f"Toplam alarm: {iso_flags.sum()}")
print("(Isolation Forest, noktaların KOMBINASYONUNA bakabildiği için bu anomalileri yakalayabiliyor)")
