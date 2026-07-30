import numpy as np

np.random.seed(42)

# Orijinal sinyal: bir sinüs dalgası
t = np.linspace(0, 4*np.pi, 100)
x0 = np.sin(t)

n_steps = 100
betas = np.linspace(0.0001, 0.02, n_steps)  # doğrusal gürültü programı (noise schedule)

# İleri süreç: kademeli gürültü ekleme
def forward_diffusion(x0, betas):
    x = x0.copy()
    trajectory = [x.copy()]
    for beta in betas:
        noise = np.random.randn(*x.shape)
        x = np.sqrt(1 - beta) * x + np.sqrt(beta) * noise
        trajectory.append(x.copy())
    return np.array(trajectory)

trajectory = forward_diffusion(x0, betas)

print("Sinyalin bozulma süreci (varyans olarak):")
for step in [0, 10, 25, 50, 75, 99]:
    signal_std = trajectory[step].std()
    print(f"  Adım {step:3d}: sinyal std={signal_std:.4f}")

print(f"\nBaşlangıç sinyali (ilk 5 değer): {np.round(x0[:5], 4)}")
print(f"Son adım (neredeyse saf gürültü, ilk 5): {np.round(trajectory[-1][:5], 4)}")

# Basit bir "denoiser": ortalama gürültü tahminini çıkarma (naif yaklaşım)
print("\n--- Basit (naif) geri süreç denemesi ---")
def naive_denoise_step(x, beta):
    # Gerçek modellerde bir sinir ağı gürültüyü tahmin eder,
    # burada basitçe "beklenen gürültü büyüklüğü kadar" geri ölçekliyoruz
    return x / np.sqrt(1 - beta)

x_denoised = trajectory[-1].copy()
for beta in reversed(betas):
    x_denoised = naive_denoise_step(x_denoised, beta)

reconstruction_error = np.mean((x_denoised - x0)**2)
print(f"Naif geri-ölçekleme sonrası yeniden yapılandırma hatası (MSE): {reconstruction_error:.4f}")
print("(Gerçek diffusion modelleri, gürültüyü DOĞRU TAHMİN ETMEK için bir sinir ağı")
print(" eğitir -- bu naif yöntem sadece süreci göstermek için, gerçek bir denoiser değil.)")
