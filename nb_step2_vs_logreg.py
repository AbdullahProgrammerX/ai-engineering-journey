import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups

print("20 Newsgroups veri seti indiriliyor (gerçek metin verisi)...\n")
categories = ['sci.space', 'rec.sport.baseball']
data = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))

X_text, y = data.data, data.target
vectorizer = CountVectorizer(max_features=5000, stop_words='english')
X_all = vectorizer.fit_transform(X_text)

print(f"Toplam örnek: {X_all.shape[0]}, özellik sayısı: {X_all.shape[1]}\n")

from sklearn.model_selection import train_test_split
X_train_full, X_test, y_train_full, y_test = train_test_split(X_all, y, test_size=0.3, random_state=42)

print(f"{'Eğitim boyutu':>15} {'NB doğruluk':>12} {'LogReg doğruluk':>16}")
print("-" * 45)

for n_train in [20, 50, 100, 300, 600, len(y_train_full)]:
    n_train = min(n_train, len(y_train_full))
    X_sub = X_train_full[:n_train]
    y_sub = y_train_full[:n_train]

    nb = MultinomialNB(alpha=1.0)
    nb.fit(X_sub, y_sub)
    nb_acc = nb.score(X_test, y_test)

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_sub, y_sub)
    lr_acc = lr.score(X_test, y_test)

    print(f"{n_train:>15} {nb_acc:>12.4f} {lr_acc:>16.4f}")
