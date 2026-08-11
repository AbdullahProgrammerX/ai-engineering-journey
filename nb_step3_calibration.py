import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve

categories = ['sci.space', 'rec.sport.baseball']
data = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))
X_text, y = data.data, data.target

vectorizer = CountVectorizer(max_features=5000, stop_words='english')
X_all = vectorizer.fit_transform(X_text)
X_train, X_test, y_train, y_test = train_test_split(X_all, y, test_size=0.3, random_state=42)

nb = MultinomialNB(alpha=1.0)
nb.fit(X_train, y_train)
nb_proba = nb.predict_proba(X_test)[:, 1]

calibrated = CalibratedClassifierCV(MultinomialNB(alpha=1.0), cv=5, method='sigmoid')
calibrated.fit(X_train, y_train)
calibrated_proba = calibrated.predict_proba(X_test)[:, 1]

print("--- Ham NB olasılıkları ne kadar 'aşırı emin' (0 veya 1'e çok yakın) ---")
print(f"NB olasılıklarının %90+ veya %10- olan oranı: {np.mean((nb_proba > 0.9) | (nb_proba < 0.1)):.2%}")
print(f"Kalibre edilmiş olasılıkların %90+ veya %10- olan oranı: {np.mean((calibrated_proba > 0.9) | (calibrated_proba < 0.1)):.2%}")

print("\n--- Kalibrasyon eğrisi: tahmin edilen olasılık ile gerçek oran ne kadar uyuşuyor ---")
prob_true_nb, prob_pred_nb = calibration_curve(y_test, nb_proba, n_bins=10)
prob_true_cal, prob_pred_cal = calibration_curve(y_test, calibrated_proba, n_bins=10)

print(f"\n{'Tahmin bin':>12} {'NB gerçek oran':>15} {'Kalibre gerçek oran':>20}")
for i in range(len(prob_pred_nb)):
    print(f"{prob_pred_nb[i]:>12.3f} {prob_true_nb[i]:>15.3f}", end="")
    if i < len(prob_pred_cal):
        print(f" {prob_true_cal[i]:>20.3f}")
    else:
        print()
