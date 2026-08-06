import math
import random

def min_max_scale(values):
    min_val, max_val = min(values), max(values)
    if max_val == min_val:
        return [0.0] * len(values)
    return [(v - min_val) / (max_val - min_val) for v in values]

def standardize(values):
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(variance) if variance > 0 else 1.0
    return [(v - mean) / std for v in values]

def impute_median(values):
    present = sorted(v for v in values if v is not None)
    n = len(present)
    median = (present[n//2-1] + present[n//2]) / 2 if n % 2 == 0 else present[n//2]
    return [v if v is not None else median for v in values], median

def impute_mean(values):
    present = [v for v in values if v is not None]
    mean = sum(present) / len(present)
    return [v if v is not None else mean for v in values], mean

def add_missing_indicator(values):
    return [0 if v is not None else 1 for v in values]

def one_hot_encode(values):
    categories = sorted(set(values))
    cat_to_idx = {cat: i for i, cat in enumerate(categories)}
    encoded = []
    for v in values:
        row = [0] * len(categories)
        row[cat_to_idx[v]] = 1
        encoded.append(row)
    return encoded, categories


def make_housing_data(n=200, seed=42):
    random.seed(seed)
    data = []
    for _ in range(n):
        sqft = random.uniform(500, 5000)
        bedrooms = random.choice([1, 2, 3, 4, 5])
        age = random.uniform(0, 50)
        neighborhood = random.choice(["downtown", "suburbs", "rural"])
        has_pool = random.choice([True, False])
        sqft_with_missing = sqft if random.random() > 0.05 else None
        age_with_missing = age if random.random() > 0.08 else None
        price = (50*sqft + 20000*bedrooms - 1000*age +
                 (50000 if neighborhood=="downtown" else 10000 if neighborhood=="suburbs" else 0) +
                 (15000 if has_pool else 0) + random.gauss(0, 20000))
        data.append({"sqft": sqft_with_missing, "bedrooms": bedrooms, "age": age_with_missing,
                     "neighborhood": neighborhood, "has_pool": has_pool, "price": price})
    return data


data = make_housing_data(200)
print("=== Ham Veri Örneği ===")
for row in data[:2]:
    print(f"  {row}")

sqft_raw = [d["sqft"] for d in data]
age_raw = [d["age"] for d in data]

print(f"\n=== Eksik Veri Yönetimi ===")
print(f"sqft eksik: {sum(1 for v in sqft_raw if v is None)}/{len(sqft_raw)}")
print(f"age eksik: {sum(1 for v in age_raw if v is None)}/{len(age_raw)}")

sqft_indicator = add_missing_indicator(sqft_raw)
sqft_imputed, sqft_fill = impute_median(sqft_raw)
age_imputed, age_fill = impute_mean(age_raw)
print(f"sqft medyan ile dolduruldu: {sqft_fill:.0f}")
print(f"age ortalama ile dolduruldu: {age_fill:.1f}")

print(f"\n=== Sayısal Dönüşümler ===")
sqft_scaled = standardize(sqft_imputed)
age_scaled = min_max_scale(age_imputed)
print(f"sqft standartlaştırıldı: ortalama={sum(sqft_scaled)/len(sqft_scaled):.4f} (0'a yakın olmalı)")
print(f"age min-max: [{min(age_scaled):.2f}, {max(age_scaled):.2f}] (0-1 arası olmalı)")

print(f"\n=== Kategorik Encoding ===")
neighborhoods = [d["neighborhood"] for d in data]
ohe, ohe_cats = one_hot_encode(neighborhoods)
print(f"One-hot kategoriler: {ohe_cats}")
print(f"Örnek: '{neighborhoods[0]}' -> {ohe[0]}")
