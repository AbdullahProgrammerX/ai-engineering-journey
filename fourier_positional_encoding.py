import numpy as np

def positional_encoding(pos, d_model):
    pe = np.zeros(d_model)
    for i in range(0, d_model, 2):
        pe[i] = np.sin(pos / (10000 ** (i / d_model)))
        if i + 1 < d_model:
            pe[i + 1] = np.cos(pos / (10000 ** (i / d_model)))
    return pe

d_model = 128
max_pos = 512

encodings = np.array([positional_encoding(p, d_model) for p in range(max_pos)])

print("--- Dot product, pozisyon farkına (|p1-p2|) göre nasıl değişiyor? ---")
p1 = 100
for distance in [0, 1, 5, 10, 50, 100, 200]:
    p2 = p1 + distance
    if p2 >= max_pos:
        continue
    dot = np.dot(encodings[p1], encodings[p2])
    print(f"  |p1-p2|={distance:>4d}  dot_product={dot:8.4f}")

print("\n--- Aynı mesafe (distance=10), farklı MUTLAK pozisyonlarda -- dot product AYNI mı? ---")
distance = 10
for p1_test in [0, 50, 100, 200, 300]:
    p2_test = p1_test + distance
    if p2_test >= max_pos:
        continue
    dot = np.dot(encodings[p1_test], encodings[p2_test])
    print(f"  p1={p1_test:>4d}, p2={p2_test:>4d}  dot_product={dot:8.4f}")

print("\nSonuç: Dot product SADECE mesafeye (|p1-p2|) bağlı, mutlak pozisyona değil.")
print("Bu, modelin 'göreceli pozisyon' kavramını öğrenebilmesini sağlıyor --")
print("örneğin 'bir önceki kelime' ilişkisi, cümlenin neresinde olursa olsun aynı şekilde kodlanıyor.")
