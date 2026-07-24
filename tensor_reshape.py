import numpy as np

print("--- Reshape ---")
t = np.arange(12).reshape(2, 6)
print(f"Orijinal (2,6):\n{t}")
r = t.reshape(3, 4)
print(f"\n(3,4)'e reshape:\n{r}")
r2 = t.reshape(-1, 3)  # -1: "bu boyutu otomatik hesapla"
print(f"\n(-1,3)'e reshape (otomatik boyut): {r2.shape}")

print("\n--- Squeeze / Unsqueeze ---")
t2 = np.zeros((1, 3, 1, 2))
print(f"Squeeze öncesi: {t2.shape}")
print(f"Squeeze sonrası: {t2.squeeze().shape}")  # boyutu 1 olan eksenleri kaldırır

v = np.array([1, 2, 3])
print(f"\nUnsqueeze öncesi: {v.shape}")
u = np.expand_dims(v, axis=0)  # PyTorch'ta .unsqueeze(0)
print(f"Unsqueeze(0) sonrası: {u.shape}")
