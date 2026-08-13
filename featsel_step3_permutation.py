import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

def make_feature_selection_data(n_samples=500, seed=42):
    rng = np.random.RandomState(seed)
    x1 = rng.randn(n_samples); x2 = rng.randn(n_samples); x3 = rng.randn(n_samples)
    x4 = x1 + 0.1 * rng.randn(n_samples); x5 = x2 + 0.1 * rng.randn(n_samples)
    informative = np.column_stack([x1, x2, x3, x4, x5])
    correlated = np.column_stack([
        x1*0.9 + 0.1*rng.randn(n_samples), x2*0.8 + 0.2*rng.randn(n_samples),
        x3*0.7 + 0.3*rng.randn(n_samples), x1*0.5+x2*0.5 + 0.1*rng.randn(n_samples),
        x2*0.6+x3*0.4 + 0.1*rng.randn(n_samples),
    ])
    # Bilerek YÜKSEK KARDİNALİTELİ ama tamamen rastgele bir özellik ekle
    high_card_noise = rng.randint(0, n_samples, size=(n_samples, 1)).astype(float)
    noise = rng.randn(n_samples, 9) * 0.5
    X = np.hstack([informative, correlated, high_card_noise, noise])
    y = (2*x1 - 1.5*x2 + x3 + 0.5*rng.randn(n_samples) > 0).astype(int)
    names = ([f"info_{i}" for i in range(5)] + [f"corr_{i}" for i in range(5)]
              + ["noise_HIGH_CARDINALITY"] + [f"noise_{i}" for i in range(9)])
    return X, y, names

X, y, feature_names = make_feature_selection_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

mdi_importance = rf.feature_importances_

perm_result = permutation_importance(rf, X_test, y_test, n_repeats=20, random_state=42)
perm_importance = perm_result.importances_mean

print(f"{'Özellik':<25} {'MDI (Tree)':>12} {'Permutation':>14}")
print("-" * 55)
for i, name in enumerate(feature_names):
    tag = " <-- YÜKSEK KARDİNALİTELİ GÜRÜLTÜ" if "HIGH_CARDINALITY" in name else ""
    print(f"{name:<25} {mdi_importance[i]:>12.4f} {perm_importance[i]:>14.4f}{tag}")

print("\nBeklenen: MDI, yüksek kardinaliteli gürültüyü YANLIŞLIKLA önemli gösterebilir.")
print("Permutation importance ise bu gürültüyü doğru şekilde DÜŞÜK/SIFIRA yakın gösterir.")
