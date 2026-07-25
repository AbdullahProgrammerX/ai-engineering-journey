import random
import math

random.seed(42)

def mean(data):
    return sum(data) / len(data)

def std_dev(data):
    m = mean(data)
    return math.sqrt(sum((x - m) ** 2 for x in data) / (len(data) - 1))

def two_sample_t_test(a, b):
    ma, mb = mean(a), mean(b)
    sa, sb = std_dev(a), std_dev(b)
    na, nb = len(a), len(b)
    pooled_se = math.sqrt(sa**2/na + sb**2/nb)
    return (ma - mb) / pooled_se


print("Aynı KÜÇÜK etkiyi (ortalama farkı=0.5), farklı örneklem büyüklükleriyle test ediyoruz:\n")

for n in [10, 100, 1000, 10000, 100000]:
    group_a = [random.gauss(50.0, 10) for _ in range(n)]
    group_b = [random.gauss(50.5, 10) for _ in range(n)]  # sadece 0.5 birimlik fark, pratikte önemsiz

    t_stat = two_sample_t_test(group_a, group_b)
    practical_diff = mean(group_b) - mean(group_a)

    print(f"n={n:>6d}  t-istatistiği={t_stat:>8.4f}  gerçek_fark={practical_diff:.4f}  "
          f"{'ANLAMLI (|t|>1.96)' if abs(t_stat) > 1.96 else 'anlamlı değil'}")

print("\nSonuç: Örneklem büyüklüğü (n) arttıkça, PRATİKTE ÖNEMSİZ olan aynı küçük fark")
print("(0.5 birim) bile istatistiksel olarak 'anlamlı' hale geliyor. Bu yüzden sadece")
print("p-value'ya bakmak yetmez -- etki büyüklüğüne (effect size) de bakmak gerekir.")
