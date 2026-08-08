from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X, y = make_regression(n_samples=300, n_features=5, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("learning_rate düştükçe, n_estimators'ı DA artırarak dengeli tutmak:\n")
for lr, n_est in [(0.1, 100), (0.05, 200), (0.03, 400), (0.01, 1000)]:
    gb = GradientBoostingRegressor(n_estimators=n_est, learning_rate=lr, max_depth=3, random_state=42)
    gb.fit(X_train, y_train)
    test_mse = mean_squared_error(y_test, gb.predict(X_test))
    print(f"learning_rate={lr:<5}  n_estimators={n_est:<5}  Test MSE={test_mse:.2f}")
