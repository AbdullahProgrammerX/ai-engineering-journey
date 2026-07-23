import numpy as np

def compress_image_svd(image_matrix, k):
    U, S, Vt = np.linalg.svd(image_matrix, full_matrices=False)
    # Sadece en önemli k tane bileşeni kullanarak yeniden oluştur
    compressed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    return compressed

np.random.seed(42)
rows, cols = 200, 300
image = np.random.randn(rows, cols)  # rastgele "görüntü" (gerçek görüntü olsaydı daha net görülürdü)

print("k arttıkça hata azalır ama depolama maliyeti artar:\n")
for k in [1, 5, 10, 20, 50]:
    compressed = compress_image_svd(image, k)
    error = np.linalg.norm(image - compressed) / np.linalg.norm(image)
    original_size = rows * cols
    compressed_size = k * (rows + cols + 1)
    ratio = compressed_size / original_size
    print(f"k={k:>3d}  hata={error:.4f}  depolama_oranı={ratio:.1%}")
