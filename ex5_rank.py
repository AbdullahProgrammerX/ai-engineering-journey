import numpy as np

# 3x3 ama rank 2: üçüncü satır ilk ikisinin toplamı olsun
A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [5, 7, 9]  # 1. satır + 2. satır
])

rank = np.linalg.matrix_rank(A)
print(f"Matris:\n{A}")
print(f"Rank: {rank}")
print("\nGeometrik anlamı: Sütunlar 3 boyutlu uzayda değil, "
      "2 boyutlu bir DÜZLEM (subspace) içinde yaşıyor. "
      "Yani 3 sütun vektörü aslında birbirine bağımlı (lineer bağımlı), "
      "üçüncüsü ilk ikisinin bir kombinasyonu.")
