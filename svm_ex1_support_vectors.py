import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=100, centers=2, cluster_std=1.0, random_state=42)
y = np.where(y == 0, -1, 1)  # SVM -1/+1 etiketleme kullanır

clf = SVC(kernel='linear', C=1000)  # yüksek C, hard margin'e yakın
clf.fit(X, y)

print(f"Toplam nokta sayısı: {len(X)}")
print(f"Support vector sayısı: {len(clf.support_vectors_)}")
print(f"Support vector oranı: {len(clf.support_vectors_)/len(X):.1%}")

# Her noktanın karar sınırına mesafesini hesapla
decision_values = clf.decision_function(X)
distances = np.abs(decision_values) / np.linalg.norm(clf.coef_[0])

# Support vector olan ve olmayan noktaların ortalama mesafesi
sv_indices = clf.support_
non_sv_indices = [i for i in range(len(X)) if i not in sv_indices]

print(f"\nSupport vector'ların ortalama mesafesi: {distances[sv_indices].mean():.4f}")
print(f"Diğer noktaların ortalama mesafesi:      {distances[non_sv_indices].mean():.4f}")
print("\n(Support vector'lar sınıra ÇOK DAHA YAKIN olmalı)")
