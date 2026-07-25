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
    t_stat = (ma - mb) / pooled_se
    return t_stat

def simulate_ab_test(mean_a, mean_b, std, n, alpha_threshold=1.96):
    group_a = [random.gauss(mean_a, std) for _ in range(n)]
    group_b = [random.gauss(mean_b, std) for _ in range(n)]
    t_stat = two_sample_t_test(group_a, group_b)
    significant = abs(t_stat) > alpha_threshold
    return significant


print("--- Tip I hata (yanlış pozitif): gerçekte fark YOK ama test 'anlamlı' diyor mu? ---")
n_trials = 1000
false_positives = 0
for _ in range(n_trials):
    # İki grup da AYNI ortalamaya sahip (gerçek fark yok)
    if simulate_ab_test(mean_a=50, mean_b=50, std=10, n=30):
        false_positives += 1
print(f"{n_trials} denemede {false_positives} kere yanlışlıkla 'anlamlı' bulundu")
print(f"Tip I hata oranı: {false_positives/n_trials:.4f}  (beklenen: ~0.05)")

print("\n--- Tip II hata (yanlış negatif): gerçekte KÜÇÜK bir fark var ama test kaçırıyor mu? ---")
false_negatives = 0
for _ in range(n_trials):
    # Gruplar arasında küçük bir fark var (mean_b biraz daha yüksek)
    if not simulate_ab_test(mean_a=50, mean_b=52, std=10, n=30):
        false_negatives += 1
print(f"{n_trials} denemede {false_negatives} kere gerçek farkı KAÇIRDI")
print(f"Tip II hata oranı: {false_negatives/n_trials:.4f}")
