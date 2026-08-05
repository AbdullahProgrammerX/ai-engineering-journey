import numpy as np
from scipy.spatial.distance import pdist

np.random.seed(42)

print("Boyut arttıkça, en uzak/en yakın mesafe oranı 1'e yaklaşıyor")
print("(yani tüm noktalar birbirine 'eşit uzaklıkta' görünmeye başlıyor):\n")

for dim in [2, 5, 10, 50, 100, 500]:
    points = np.random.uniform(0, 1, size=(1000, dim))
    distances = pdist(points)
    ratio = distances.max() / distances.min()
    print(f"Boyut={dim:>4d}  max/min mesafe oranı = {ratio:.4f}")
