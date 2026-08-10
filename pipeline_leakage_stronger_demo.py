import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, f_classif

np.random.seed(42)
# ÇOK küçük veri seti + çok fazla özellik (leakage'ın en görünür olduğu senaryo)
X, y = make_classification(n_samples=60, n_features=100, n_informative=5,
                             n_redundant=0, random_state=42)

cv = KFold(n_splits=5, shuffle=True, random_state=42)

print("=== YANLIŞ: Feature selection'ı TÜM veriye uygula, sonra CV yap ===\n")
selector_leaky = SelectKBest(f_classif, k=10)
X_selected_leaky = selector_leaky.fit_transform(X, y)  # TÜM veriye (X VE y) bakıyor!

model_leaky = KNeighborsClassifier(n_neighbors=5)
leaky_scores = cross_val_score(model_leaky, X_selected_leaky, y, cv=cv)
print(f"Leaky ortalama: {leaky_scores.mean():.4f}\n")

print("=== DOĞRU: Feature selection pipeline İÇİNDE, her fold'da ayrı yapılır ===\n")
pipe_clean = Pipeline([
    ("selector", SelectKBest(f_classif, k=10)),
    ("model", KNeighborsClassifier(n_neighbors=5)),
])
clean_scores = cross_val_score(pipe_clean, X, y, cv=cv)
print(f"Temiz ortalama: {clean_scores.mean():.4f}\n")

print(f"=== Fark ===")
print(f"Leaky - Temiz = {leaky_scores.mean() - clean_scores.mean():+.4f}")
print("\nBu sefer fark GÖRÜNÜR olmalı -- çünkü feature selection, hangi 10 özelliğin")
print("'en iyi' olduğuna TÜM veriye (test dahil) bakarak karar veriyor. Bu, gerçek")
print("dünyada asla sahip olamayacağın bir bilgi avantajı.")
