import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

np.random.seed(42)
n = 300

df = pd.DataFrame({
    "age": np.random.normal(40, 12, n),
    "income": np.random.normal(50000, 15000, n),
    "credit_score": np.random.normal(650, 80, n),
    "city": np.random.choice(["Istanbul", "Ankara", "Izmir"], n),
    "employment": np.random.choice(["Full-time", "Part-time", "Self-employed"], n),
})

# Bilerek eksik değer ekle
df.loc[np.random.choice(n, 20, replace=False), "age"] = np.nan
df.loc[np.random.choice(n, 15, replace=False), "city"] = np.nan

y = ((df["income"] > 50000) & (df["credit_score"] > 640)).astype(int)
y = y.fillna(0) if hasattr(y, 'fillna') else y

numeric_features = ["age", "income", "credit_score"]
categorical_features = ["city", "employment"]

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_features),
    ("cat", categorical_pipe, categorical_features),
])

full_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
])

print("=== Tam Production Pipeline ===\n")
print(f"Veri seti: {n} satır, {len(numeric_features)} sayısal + {len(categorical_features)} kategorik özellik")
print(f"Eksik değerler: age'de {df['age'].isna().sum()}, city'de {df['city'].isna().sum()}\n")

scores = cross_val_score(full_pipeline, df, y, cv=5, scoring="accuracy")
print(f"5-fold CV doğruluk skorları: {np.round(scores, 4)}")
print(f"Ortalama: {scores.mean():.4f} (+/- {scores.std():.4f})")

full_pipeline.fit(df, y)
print(f"\nPipeline eğitildi. Tek satırda tüm ön işleme + model dahil.")
print(f"Yeni veri geldiğinde: full_pipeline.predict(yeni_df) -- hepsi otomatik.")
