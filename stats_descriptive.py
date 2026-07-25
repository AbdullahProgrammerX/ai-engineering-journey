import math

def mean(data):
    return sum(data) / len(data)

def median(data):
    s = sorted(data)
    n = len(s)
    mid = n // 2
    if n % 2 == 0:
        return (s[mid-1] + s[mid]) / 2
    return s[mid]

def mode(data):
    counts = {}
    for x in data:
        counts[x] = counts.get(x, 0) + 1
    max_count = max(counts.values())
    return [k for k, v in counts.items() if v == max_count]

def std_dev(data, sample=True):
    m = mean(data)
    variance = sum((x - m) ** 2 for x in data) / (len(data) - (1 if sample else 0))
    return math.sqrt(variance)

def percentile(data, p):
    s = sorted(data)
    idx = (p / 100) * (len(s) - 1)
    lower = int(math.floor(idx))
    upper = int(math.ceil(idx))
    if lower == upper:
        return s[lower]
    frac = idx - lower
    return s[lower] + frac * (s[upper] - s[lower])

def iqr(data):
    return percentile(data, 75) - percentile(data, 25)


data = [23, 45, 12, 67, 34, 45, 89, 23, 56, 78, 45, 12, 90, 34, 67]

print(f"Veri: {data}")
print(f"Ortalama (mean):     {mean(data):.4f}")
print(f"Medyan (median):     {median(data):.4f}")
print(f"Mod (mode):          {mode(data)}")
print(f"Std sapma (sample):  {std_dev(data):.4f}")
print(f"25. yüzdelik:        {percentile(data, 25):.4f}")
print(f"75. yüzdelik:        {percentile(data, 75):.4f}")
print(f"IQR:                 {iqr(data):.4f}")
