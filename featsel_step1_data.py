import numpy as np

def make_feature_selection_data(n_samples=500, seed=42):
    rng = np.random.RandomState(seed)
    x1 = rng.randn(n_samples)
    x2 = rng.randn(n_samples)
    x3 = rng.randn(n_samples)
    x4 = x1 + 0.1 * rng.randn(n_samples)
    x5 = x2 + 0.1 * rng.randn(n_samples)
    informative = np.column_stack([x1, x2, x3, x4, x5])
    correlated = np.column_stack([
        x1 * 0.9 + 0.1 * rng.randn(n_samples),
        x2 * 0.8 + 0.2 * rng.randn(n_samples),
        x3 * 0.7 + 0.3 * rng.randn(n_samples),
        x1 * 0.5 + x2 * 0.5 + 0.1 * rng.randn(n_samples),
        x2 * 0.6 + x3 * 0.4 + 0.1 * rng.randn(n_samples),
    ])
    noise = rng.randn(n_samples, 10) * 0.5
    X = np.hstack([informative, correlated, noise])
    y = (2 * x1 - 1.5 * x2 + x3 + 0.5 * rng.randn(n_samples) > 0).astype(int)
    feature_names = [f"info_{i}" for i in range(5)] + [f"corr_{i}" for i in range(5)] + [f"noise_{i}" for i in range(10)]
    return X, y, feature_names

X, y, feature_names = make_feature_selection_data()
print(f"Veri seti: {X.shape[0]} örnek, {X.shape[1]} özellik")
print(f"\nGerçek yapı (biz biliyoruz, algoritma bilmiyor):")
print(f"  info_0..4: gerçekten bilgilendirici (info_3,4 aslında info_0,1'in neredeyse kopyası)")
print(f"  corr_0..4: bilgilendirici özelliklerle korelasyonlu ama dolaylı")
print(f"  noise_0..9: TAMAMEN rastgele, hedefle hiçbir ilişkisi yok")
print(f"\nHedef formülü: y = (2*x1 - 1.5*x2 + x3 + gürültü > 0)")
print("(yani gerçekte SADECE x1, x2, x3 hedefi belirliyor)")
