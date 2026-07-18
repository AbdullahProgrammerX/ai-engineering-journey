import numpy as np

# x'i 2 katına, y'yi 3 katına çıkaran ölçekleme matrisi
scale_matrix = np.array([
    [2, 0],
    [0, 3]
])

v = np.array([1, 1])
result = scale_matrix @ v
print(f"Orijinal vektör: {v}")
print(f"Ölçeklenmiş vektör: {result}")  # [2, 3] beklenir
