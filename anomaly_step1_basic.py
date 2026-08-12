import numpy as np

def zscore_detect(X, threshold=3.0):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    z = np.abs((X - mean) / std)
    return z.max(axis=1) > threshold

def iqr_detect(X, factor=1.5):
    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)
    iqr = q3 - q1
    iqr = np.where(iqr == 0, 1.0, iqr)
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    outside = (X < lower) | (X > upper)
    return outside.any(axis=1)


np.random.seed(42)
# Normal küme + bilerek eklenen aykırı değerler
normal_data = np.random.normal(0, 1, (200, 2))
outliers = np.array([[8, 8], [-7, 7], [9, -6], [-8, -8]])
X = np.vstack([normal_data, outliers])
y_true = np.array([0]*200 + [1]*4)  # gerçek etiketler (sadece değerlendirme için)

z_flags = zscore_detect(X, threshold=3.0)
iqr_flags = iqr_detect(X, factor=1.5)

print(f"Toplam nokta: {len(X)}, gerçek aykırı değer sayısı: {y_true.sum()}\n")
print(f"Z-score ile tespit edilen: {z_flags.sum()}")
print(f"  Gerçek aykırı değerlerin kaçı yakalandı: {(z_flags & (y_true==1)).sum()}/{y_true.sum()}")
print(f"  Yanlış alarm sayısı: {(z_flags & (y_true==0)).sum()}")

print(f"\nIQR ile tespit edilen: {iqr_flags.sum()}")
print(f"  Gerçek aykırı değerlerin kaçı yakalandı: {(iqr_flags & (y_true==1)).sum()}/{y_true.sum()}")
print(f"  Yanlış alarm sayısı: {(iqr_flags & (y_true==0)).sum()}")
