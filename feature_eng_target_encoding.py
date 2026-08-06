def target_encode(feature_values, target_values, smoothing=10):
    global_mean = sum(target_values) / len(target_values)
    category_stats = {}
    for feat, target in zip(feature_values, target_values):
        if feat not in category_stats:
            category_stats[feat] = {"sum": 0.0, "count": 0}
        category_stats[feat]["sum"] += target
        category_stats[feat]["count"] += 1

    encoding = {}
    for cat, stats in category_stats.items():
        cat_mean = stats["sum"] / stats["count"]
        weight = stats["count"] / (stats["count"] + smoothing)
        encoding[cat] = weight * cat_mean + (1 - weight) * global_mean
    return [encoding[v] for v in feature_values], encoding


def label_encode(values):
    categories = sorted(set(values))
    cat_to_int = {cat: i for i, cat in enumerate(categories)}
    return [cat_to_int[v] for v in values], cat_to_int


neighborhoods = ["downtown"]*50 + ["suburbs"]*30 + ["rural"]*5  # rural ÇOK AZ örnek
prices = [500000 + i*100 for i in range(50)] + [300000 + i*100 for i in range(30)] + [200000, 210000, 195000, 205000, 198000]

print("--- Label Encoding (naif, anlamsız sıralama) ---")
le, le_map = label_encode(neighborhoods)
print(f"Kategoriler: {le_map}")
print("Sorun: 'downtown'=0, 'rural'=1, 'suburbs'=2 -- bu sayılar arasında hiçbir")
print("gerçek matematiksel ilişki yok, ama model bunu 'rural, downtown'dan büyük' sanabilir!\n")

print("--- Target Encoding (akıllı, ama dikkatli kullanılmalı) ---")
te, te_map = target_encode(neighborhoods, prices, smoothing=10)
print(f"Encoding haritası: {({k: round(v) for k, v in te_map.items()})}")
print(f"\n'rural' sadece 5 örneğe sahip -- smoothing sayesinde,")
print(f"encoding'i global ortalamaya doğru 'çekiliyor', aşırı güvenilmiyor.")

print("\n--- Smoothing'in etkisini gözlemleme ---")
for smooth in [0, 5, 10, 50, 200]:
    te_s, te_map_s = target_encode(neighborhoods, prices, smoothing=smooth)
    print(f"smoothing={smooth:>3d}  rural encoding={te_map_s['rural']:.0f}  "
          f"(global ortalama={sum(prices)/len(prices):.0f})")
