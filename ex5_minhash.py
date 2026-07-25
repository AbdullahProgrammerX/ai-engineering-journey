import numpy as np
import random

random.seed(42)
np.random.seed(42)

def exact_jaccard(set_a, set_b):
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def make_hash_functions(n_hashes, max_val=2**32 - 1):
    # Her hash fonksiyonu: h(x) = (a*x + b) mod p şeklinde basit bir "universal hash"
    p = 4294967311  # 2^32'den büyük bir asal sayı
    hash_funcs = []
    for _ in range(n_hashes):
        a = random.randint(1, p - 1)
        b = random.randint(0, p - 1)
        hash_funcs.append((a, b, p))
    return hash_funcs


def minhash_signature(s, hash_funcs):
    signature = []
    for a, b, p in hash_funcs:
        min_hash = min(((a * hash(elem) + b) % p) for elem in s)
        signature.append(min_hash)
    return signature


def minhash_similarity(sig_a, sig_b):
    matches = sum(1 for x, y in zip(sig_a, sig_b) if x == y)
    return matches / len(sig_a)


# 100 rastgele küme oluştur (evren: 1-200 arası sayılar, her küme 10-30 eleman)
n_sets = 100
universe = list(range(1, 201))
sets = [set(random.sample(universe, random.randint(10, 30))) for _ in range(n_sets)]

# Tüm ikili çiftler için tam (exact) Jaccard hesapla
print("Tam Jaccard hesaplanıyor (tüm çiftler)...")
exact_scores = {}
for i in range(n_sets):
    for j in range(i + 1, n_sets):
        exact_scores[(i, j)] = exact_jaccard(sets[i], sets[j])

print(f"Toplam çift sayısı: {len(exact_scores)}")

# Farklı hash fonksiyonu sayılarıyla MinHash yaklaşıklığını test et
for n_hashes in [50, 100, 200]:
    hash_funcs = make_hash_functions(n_hashes)
    signatures = [minhash_signature(s, hash_funcs) for s in sets]

    errors = []
    for (i, j), exact in exact_scores.items():
        approx = minhash_similarity(signatures[i], signatures[j])
        errors.append(abs(exact - approx))

    mean_error = np.mean(errors)
    max_error = np.max(errors)
    print(f"\nHash sayısı={n_hashes:>3d}  ortalama_hata={mean_error:.4f}  maksimum_hata={max_error:.4f}")
