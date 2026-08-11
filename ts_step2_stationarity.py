import numpy as np

def check_stationarity(series, window=50):
    rolling_mean = np.array([series[max(0, i-window):i].mean() for i in range(1, len(series)+1)])
    rolling_std = np.array([series[max(0, i-window):i].std() for i in range(1, len(series)+1)])
    return rolling_mean, rolling_std

def is_stationary(series):
    n = len(series)
    first_half, second_half = series[:n//2], series[n//2:]
    mean_diff = abs(first_half.mean() - second_half.mean())
    std_pooled = series.std()
    var_ratio = max(first_half.var(), second_half.var()) / (min(first_half.var(), second_half.var()) + 1e-9)
    stationary = mean_diff < 0.5 * std_pooled and var_ratio < 2
    return stationary, mean_diff, std_pooled, var_ratio

np.random.seed(42)
n = 500
t = np.arange(n)
trend_series = 100 + 0.1 * t + np.random.normal(0, 3, n)

print("--- Orijinal seri (trend içeriyor) ---")
stat, mean_diff, std_pooled, var_ratio = is_stationary(trend_series)
print(f"Durağan mı: {stat}")
print(f"İlk yarı - ikinci yarı ortalama farkı: {mean_diff:.4f} (std'nin yarısı: {0.5*std_pooled:.4f})")
print(f"Varyans oranı: {var_ratio:.4f}")

print("\n--- Bir kere farklama sonrası (y_t - y_{t-1}) ---")
diff_series = np.diff(trend_series)
stat2, mean_diff2, std_pooled2, var_ratio2 = is_stationary(diff_series)
print(f"Durağan mı: {stat2}")
print(f"İlk yarı - ikinci yarı ortalama farkı: {mean_diff2:.4f} (std'nin yarısı: {0.5*std_pooled2:.4f})")
print(f"Varyans oranı: {var_ratio2:.4f}")

print("\n--- Kuadratik trend ile kaç farklama gerekiyor? ---")
quad_series = 100 + 0.01 * t**2 + np.random.normal(0, 3, n)
current = quad_series.copy()
for round_num in range(1, 4):
    stat, md, sp, vr = is_stationary(current)
    print(f"  {round_num}. farklama sonrası durağan mı: {stat}  (mean_diff={md:.4f}, threshold={0.5*sp:.4f})")
    if stat:
        break
    current = np.diff(current)
