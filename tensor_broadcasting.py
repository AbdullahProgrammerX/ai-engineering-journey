import numpy as np

print("--- Örnek 1: Bias ekleme ---")
activations = np.random.randn(4, 3)
bias = np.array([0.1, 0.2, 0.3])
result = activations + bias  # (4,3) + (3,) -> otomatik (4,3)
print(f"activations {activations.shape} + bias {bias.shape} = {result.shape}")

print("\n--- Örnek 2: Kanal bazlı ölçekleme (görüntü işleme) ---")
images = np.random.randn(2, 3, 4, 4)  # (batch, channel, height, width)
scale = np.array([0.5, 1.0, 1.5]).reshape(1, 3, 1, 1)
result = images * scale
print(f"images {images.shape} * scale {scale.shape} = {result.shape}")

print("\n--- Örnek 3: Dış çarpım (outer product) ---")
a = np.array([1, 2, 3]).reshape(-1, 1)   # (3,1)
b = np.array([10, 20, 30, 40]).reshape(1, -1)  # (1,4)
outer = a * b
print(f"a {a.shape} * b {b.shape} = {outer.shape}")
print(outer)

print("\n--- Örnek 4: İkili mesafe hesaplama (pairwise distance) ---")
M, N = 3, 4
points_a = np.random.randn(M, 2)
points_b = np.random.randn(N, 2)
diff = points_a.reshape(M, 1, 2) - points_b.reshape(1, N, 2)
distances = np.sqrt((diff ** 2).sum(axis=-1))
print(f"points_a {points_a.shape}, points_b {points_b.shape} -> distances {distances.shape}")
