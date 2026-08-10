import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold
from sklearn.datasets import make_classification

np.random.seed(42)
X, y = make_classification(n_samples=200, n_features=50, n_informative=10,
                             n_redundant=0, random_state=42)

print("=== YANLIŞ YAKLAŞIM: Scaler'ı TÜM veriye fit et, sonra CV yap ===\n")
scaler_leaky = StandardScaler()
X_scaled_leaky = scaler_leaky.fit_transform(X)  # TÜM veriye bakıyor -- LEAKAGE!

model_leaky = LogisticRegression()
cv = KFold(n_splits=5, shuffle=True, random_state=42)
leaky_scores = cross_val_score(model_leaky, X_scaled_leaky, y, cv=cv)

print(f"Leaky CV skorları: {np.round(leaky_scores, 4)}")
print(f"Leaky ortalama: {leaky_scores.mean():.4f}\n")

print("=== DOĞRU YAKLAŞIM: Pipeline içinde, her fold'da ayrı ayrı fit et ===\n")
pipe_clean = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression()),
])
clean_scores = cross_val_score(pipe_clean, X, y, cv=cv)

print(f"Temiz CV skorları: {np.round(clean_scores, 4)}")
print(f"Temiz ortalama: {clean_scores.mean():.4f}\n")

print(f"=== Fark ===")
print(f"Leaky - Temiz = {leaky_scores.mean() - clean_scores.mean():+.4f}")
print("(Leakage genelde skorları YAPAY OLARAK YÜKSELTİR -- gerçekte olmayan bir avantaj sağlıyor)")
