import numpy as np

np.random.seed(42)

# Temiz, düzenli bir sinyal oluştur (düşük "rank" -- basit bir örüntü)
clean = np.outer(np.sin(np.linspace(0, 4*np.pi, 100)),
                 np.cos(np.linspace(0, 2*np.pi, 80)))

# Üzerine gürültü ekle
noise = 0.3 * np.random.randn(100, 80)
noisy = clean + noise

# SVD uygula, sadece en önemli 5 bileşeni tut
U, S, Vt = np.linalg.svd(noisy, full_matrices=False)
denoised = U[:, :5] @ np.diag(S[:5]) @ Vt[:5, :]

print(f"Gürültülü veri, temiz veriden ne kadar uzak: {np.linalg.norm(noisy - clean):.4f}")
print(f"Temizlenmiş veri, temiz veriden ne kadar uzak: {np.linalg.norm(denoised - clean):.4f}")
improvement = (1 - np.linalg.norm(denoised - clean) / np.linalg.norm(noisy - clean))
print(f"İyileşme oranı: {improvement:.1%}")
