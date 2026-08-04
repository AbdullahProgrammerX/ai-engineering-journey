from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("--- Tekil Decision Tree (sınırsız derinlik) ---")
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
print(f"Train accuracy: {dt.score(X_train, y_train):.4f}")
print(f"Test accuracy:  {dt.score(X_test, y_test):.4f}")

print("\n--- Random Forest (100 ağaç) ---")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print(f"Train accuracy: {rf.score(X_train, y_train):.4f}")
print(f"Test accuracy:  {rf.score(X_test, y_test):.4f}")

print(f"\nFeature importances (Random Forest): {rf.feature_importances_}")
feature_names = ["sepal length", "sepal width", "petal length", "petal width"]
for name, importance in zip(feature_names, rf.feature_importances_):
    print(f"  {name:15s}: {importance:.4f}")

print("\n--- Farklı ağaç sayılarıyla test doğruluğu ---")
for n_trees in [1, 5, 10, 50, 200]:
    rf_n = RandomForestClassifier(n_estimators=n_trees, random_state=42)
    rf_n.fit(X_train, y_train)
    print(f"n_trees={n_trees:>4d}  test_accuracy={rf_n.score(X_test, y_test):.4f}")
