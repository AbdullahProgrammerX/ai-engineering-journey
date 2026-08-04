import math

def gini_impurity(labels):
    n = len(labels)
    if n == 0:
        return 0.0
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return 1.0 - sum((c / n) ** 2 for c in counts.values())

def entropy(labels):
    n = len(labels)
    if n == 0:
        return 0.0
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return -sum((c / n) * math.log2(c / n) for c in counts.values() if c > 0)


print("--- Saf (pure) küme -- tüm etiketler aynı ---")
pure = [1, 1, 1, 1, 1]
print(f"Gini: {gini_impurity(pure):.4f}  (0 olmalı -- hiç 'kirlilik' yok)")
print(f"Entropy: {entropy(pure):.4f}  (0 olmalı)")

print("\n--- Tamamen karışık küme -- 50/50 ---")
mixed = [0, 1, 0, 1, 0, 1]
print(f"Gini: {gini_impurity(mixed):.4f}  (0.5 olmalı -- maksimum kirlilik)")
print(f"Entropy: {entropy(mixed):.4f}  (1.0 olmalı -- maksimum belirsizlik)")

print("\n--- Kısmen karışık küme ---")
partial = [0, 0, 0, 1]
print(f"Gini: {gini_impurity(partial):.4f}")
print(f"Entropy: {entropy(partial):.4f}")


def information_gain(parent_labels, left_labels, right_labels, criterion="gini"):
    measure = gini_impurity if criterion == "gini" else entropy
    n = len(parent_labels)
    n_left, n_right = len(left_labels), len(right_labels)
    if n_left == 0 or n_right == 0:
        return 0.0
    parent_impurity = measure(parent_labels)
    child_impurity = (n_left/n)*measure(left_labels) + (n_right/n)*measure(right_labels)
    return parent_impurity - child_impurity


print("\n--- Bilgi kazancı: hangi bölünme daha iyi? ---")
parent = [0, 0, 0, 1, 1, 1]

# İyi bölünme: sınıfları tamamen ayırıyor
good_left, good_right = [0, 0, 0], [1, 1, 1]
gain_good = information_gain(parent, good_left, good_right)
print(f"İyi bölünme (tam ayrım): kazanç = {gain_good:.4f}")

# Kötü bölünme: hiçbir şey ayırmıyor
bad_left, bad_right = [0, 1, 0], [0, 1, 1]
gain_bad = information_gain(parent, bad_left, bad_right)
print(f"Kötü bölünme (karışık kalıyor): kazanç = {gain_bad:.4f}")
