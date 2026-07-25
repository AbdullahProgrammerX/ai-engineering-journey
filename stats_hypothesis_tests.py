import math

def mean(data):
    return sum(data) / len(data)

def std_dev(data):
    m = mean(data)
    return math.sqrt(sum((x - m) ** 2 for x in data) / (len(data) - 1))

def one_sample_t_test(data, popmean):
    m = mean(data)
    s = std_dev(data)
    n = len(data)
    t_stat = (m - popmean) / (s / math.sqrt(n))
    df = n - 1
    return t_stat, df

def two_sample_t_test(a, b):
    ma, mb = mean(a), mean(b)
    sa, sb = std_dev(a), std_dev(b)
    na, nb = len(a), len(b)
    pooled_se = math.sqrt(sa**2/na + sb**2/nb)
    t_stat = (ma - mb) / pooled_se
    df = na + nb - 2
    return t_stat, df

def chi_squared_test(observed, expected):
    chi2 = sum((o - e) ** 2 / e for o, e in zip(observed, expected))
    return chi2


print("--- Tek örneklem t-testi ---")
sample = [102, 98, 105, 110, 95, 103, 99, 108]
t_stat, df = one_sample_t_test(sample, popmean=100)
print(f"Örneklem: {sample}")
print(f"H0: popülasyon ortalaması = 100")
print(f"t-istatistiği: {t_stat:.4f}, serbestlik derecesi: {df}")

print("\n--- İki örneklem t-testi ---")
group_a = [23, 25, 21, 27, 24, 26, 22]
group_b = [30, 32, 28, 31, 29, 33, 27]
t_stat2, df2 = two_sample_t_test(group_a, group_b)
print(f"Grup A: {group_a}")
print(f"Grup B: {group_b}")
print(f"t-istatistiği: {t_stat2:.4f}, serbestlik derecesi: {df2}")
print("(Büyük |t| değeri, iki grup ortalamasının birbirinden anlamlı şekilde farklı olduğuna işaret eder)")

print("\n--- Chi-squared testi ---")
observed = [18, 22, 20, 40]  # gözlenen frekanslar
expected = [25, 25, 25, 25]  # beklenen (eşit dağılım varsayımı)
chi2 = chi_squared_test(observed, expected)
print(f"Gözlenen: {observed}")
print(f"Beklenen: {expected}")
print(f"Chi-squared istatistiği: {chi2:.4f}")
print("(Büyük değer, gözlenen dağılımın beklenenden anlamlı şekilde saptığına işaret eder)")
