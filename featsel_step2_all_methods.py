import numpy as np
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, RFE
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.ensemble import RandomForestClassifier

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
    noise = rng.randn(n_samples, 10) * 0.5
    X = np.hstack([informative, correlated, noise])
    y = (2*x1 - 1.5*x2 + x3 + 0.5*rng.randn(n_samples) > 0).astype(int)
    names = [f"info_{i}" for i in range(5)] + [f"corr_{i}" for i in range(5)] + [f"noise_{i}" for i in range(10)]
    return X, y, names

X, y, feature_names = make_feature_selection_data()

# Mutual Information
mi_scores = mutual_info_classif(X, y, random_state=42)

# RFE (top 5 seçsin)
rfe = RFE(LogisticRegression(max_iter=500), n_features_to_select=5)
rfe.fit(X, y)

# L1/Lasso
lasso = Lasso(alpha=0.02)
lasso.fit(X, y)

# Tree importance (MDI)
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X, y)

print(f"{'Özellik':<10} {'MI Skoru':>10} {'RFE Seçti':>10} {'Lasso Katsayı':>14} {'Tree Importance':>16}")
print("-" * 65)
for i, name in enumerate(feature_names):
    tag = "GERÇEK" if i < 5 else ("korelasyon" if i < 10 else "GÜRÜLTÜ")
    print(f"{name:<10} {mi_scores[i]:>10.4f} {'✓' if rfe.support_[i] else '':>10} "
          f"{lasso.coef_[i]:>14.4f} {rf.feature_importances_[i]:>16.4f}   [{tag}]")
