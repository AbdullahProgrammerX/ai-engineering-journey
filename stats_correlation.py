import math

def mean(data):
    return sum(data) / len(data)

def pearson_correlation(x, y):
    mx, my = mean(x), mean(y)
    numerator = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    denom_x = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    denom_y = math.sqrt(sum((yi - my) ** 2 for yi in y))
    return numerator / (denom_x * denom_y)

def rank_data(data):
    sorted_idx = sorted(range(len(data)), key=lambda i: data[i])
    ranks = [0] * len(data)
    for rank, idx in enumerate(sorted_idx):
        ranks[idx] = rank + 1
    return ranks

def spearman_correlation(x, y):
    rx = rank_data(x)
    ry = rank_data(y)
    return pearson_correlation(rx, ry)


x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y_linear = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]  # tam doğrusal ilişki
y_nonlinear = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]  # x^2, doğrusal değil ama monoton

print("--- Doğrusal ilişki (y = 2x) ---")
print(f"Pearson:  {pearson_correlation(x, y_linear):.4f}")
print(f"Spearman: {spearman_correlation(x, y_linear):.4f}")

print("\n--- Doğrusal olmayan ama monoton ilişki (y = x^2) ---")
print(f"Pearson:  {pearson_correlation(x, y_nonlinear):.4f}")
print(f"Spearman: {spearman_correlation(x, y_nonlinear):.4f}")
