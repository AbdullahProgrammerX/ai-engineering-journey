import random

random.seed(42)

def bootstrap_ci(data, statistic_fn, n_bootstrap=2000, ci=0.95):
    n = len(data)
    boot_stats = []
    for _ in range(n_bootstrap):
        sample = [random.choice(data) for _ in range(n)]
        boot_stats.append(statistic_fn(sample))
    boot_stats.sort()
    lower_idx = int((1 - ci) / 2 * n_bootstrap)
    upper_idx = int((1 - (1 - ci) / 2) * n_bootstrap)
    return boot_stats[lower_idx], boot_stats[upper_idx]

def mean(data):
    return sum(data) / len(data)

def median(data):
    s = sorted(data)
    n = len(s)
    mid = n // 2
    return (s[mid-1] + s[mid]) / 2 if n % 2 == 0 else s[mid]


data = [23, 45, 12, 67, 34, 45, 89, 23, 56, 78, 45, 12, 90, 34, 67, 21, 55, 43, 38, 29]

print(f"Veri: {data}")
print(f"Gerçek ortalama: {mean(data):.4f}")

lower, upper = bootstrap_ci(data, mean, n_bootstrap=2000, ci=0.95)
print(f"Ortalama için %95 bootstrap güven aralığı: [{lower:.4f}, {upper:.4f}]")

lower_med, upper_med = bootstrap_ci(data, median, n_bootstrap=2000, ci=0.95)
print(f"Medyan için %95 bootstrap güven aralığı: [{lower_med:.4f}, {upper_med:.4f}]")

print("\nSezgi: Elimizdeki veriden TEKRARLI ÖRNEKLEME yaparak (yerine koyarak),")
print("'gerçek popülasyon ortalaması muhtemelen bu aralıkta' diyebiliyoruz --")
print("hiçbir dağılım varsayımı (normal dağılım vs.) yapmadan.")
