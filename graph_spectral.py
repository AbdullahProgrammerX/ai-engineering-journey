import numpy as np

# İki "kümeyi" tek bir zayıf kenarla bağlayan bir graf kuruyoruz
# Küme A: 0,1,2 (birbirine sıkı bağlı) -- Küme B: 3,4,5 (birbirine sıkı bağlı)
# Aralarında sadece TEK bir köprü kenarı: 2-3
A = np.array([
    [0, 1, 1, 0, 0, 0],
    [1, 0, 1, 0, 0, 0],
    [1, 1, 0, 1, 0, 0],  # <- 2, köprü düğümü (3'e bağlı)
    [0, 0, 1, 0, 1, 1],  # <- 3, köprü düğümü (2'ye bağlı)
    [0, 0, 0, 1, 0, 1],
    [0, 0, 0, 1, 1, 0]
])

D = np.diag(A.sum(axis=1))
L = D - A

print("Komşuluk matrisi A:")
print(A)
print("\nLaplacian L = D - A:")
print(L)

eigenvalues, eigenvectors = np.linalg.eigh(L)
print(f"\nÖzdeğerler (küçükten büyüğe): {np.round(eigenvalues, 4)}")

zero_count = np.sum(np.abs(eigenvalues) < 1e-9)
print(f"Sıfıra yakın özdeğer sayısı: {zero_count}  (= bağlı bileşen sayısı)")

fiedler_value = eigenvalues[1]
fiedler_vector = eigenvectors[:, 1]
print(f"\nFiedler değeri (en küçük sıfır-olmayan özdeğer): {fiedler_value:.4f}")
print(f"Fiedler vektörü: {np.round(fiedler_vector, 4)}")

group_a = np.where(fiedler_vector >= 0)[0]
group_b = np.where(fiedler_vector < 0)[0]
print(f"\nSpektral kümeleme sonucu:")
print(f"  Küme A (pozitif): düğümler {group_a}")
print(f"  Küme B (negatif): düğümler {group_b}")
print("\n(Beklenen: {0,1,2} bir grupta, {3,4,5} diğer grupta -- HİÇ graf gezmeden,")
print("sadece özdeğer/özvektör hesaplayarak grafın doğal yapısını bulduk!)")
