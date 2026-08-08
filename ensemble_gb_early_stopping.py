import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X, y = make_regression(n_samples=300, n_features=5, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train2, X_val, y_train2, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

print("--- Early stopping OLMADAN (n_estimators=100, sabit) ---")
gb_no_stop = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_no_stop.fit(X_train, y_train)
print(f"Test MSE: {mean_squared_error(y_test, gb_no_stop.predict(X_test)):.2f}")

print("\n--- Early stopping İLE (validation loss 10 tur iyileşmezse dur) ---")
gb_early = GradientBoostingRegressor(
    n_estimators=500, learning_rate=0.1, max_depth=3,
    validation_fraction=0.2, n_iter_no_change=10, random_state=42
)
gb_early.fit(X_train, y_train)
print(f"Gerçekte kullanılan ağaç sayısı: {gb_early.n_estimators_}")
print(f"Test MSE: {mean_squared_error(y_test, gb_early.predict(X_test)):.2f}")

print("\n--- max_depth küçültme ve learning_rate düşürme ile ---")
gb_regularized = GradientBoostingRegressor(n_estimators=100, learning_rate=0.03, max_depth=2, random_state=42)
gb_regularized.fit(X_train, y_train)
print(f"Test MSE: {mean_squared_error(y_test, gb_regularized.predict(X_test)):.2f}")
