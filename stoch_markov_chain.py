import numpy as np

P = np.array([[0.7, 0.1, 0.2],
              [0.3, 0.4, 0.3],
              [0.4, 0.2, 0.4]])

print("--- Yöntem 1: Power method (dağılımı tekrar tekrar P ile çarpmak) ---")
distribution = np.array([1.0, 0.0, 0.0])  # "Sunny"den başla
for i in range(100):
    distribution = distribution @ P
print(f"100 geçiş sonrası dağılım: {np.round(distribution, 4)}")

# Farklı bir başlangıç noktasından başlarsak aynı sonuca ulaşıyor muyuz?
distribution2 = np.array([0.0, 1.0, 0.0])  # "Rainy"den başla
for i in range(100):
    distribution2 = distribution2 @ P
print(f"Farklı başlangıçtan (Rainy) 100 geçiş sonrası: {np.round(distribution2, 4)}")

print("\n--- Yöntem 2: Özdeğer (eigenvalue) yöntemi ---")
eigenvalues, eigenvectors = np.linalg.eig(P.T)
idx = np.argmin(np.abs(eigenvalues - 1.0))
stationary = np.real(eigenvectors[:, idx])
stationary = stationary / stationary.sum()
print(f"Özvektör yöntemiyle durağan dağılım: {np.round(np.abs(stationary), 4)}")

print("\nSonuç: Hangi durumdan başlarsan başla, yeterince geçiş sonrası AYNI durağan")
print("dağılıma ulaşıyorsun -- bu, Markov zincirlerinin temel bir özelliği.")
