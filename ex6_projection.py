import numpy as np

v = np.array([1, 2, 3])
u = np.array([1, 1, 1])

# v'yi u üzerine izdüşür
proj = (np.dot(v, u) / np.dot(u, u)) * u
print(f"v = {v}")
print(f"u = {u}")
print(f"v'nin u üzerine izdüşümü: {proj}")
print("\nGeometrik anlamı: Bu, v vektörünün u doğrultusundaki "
      "'gölgesi' -- v'yi u yönünde ne kadar ilerlediğini gösteren bileşen. "
      "v - proj ise u'ya dik kalan kısımdır.")
