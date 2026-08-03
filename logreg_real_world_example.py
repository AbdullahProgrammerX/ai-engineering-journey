import random
import math

def sigmoid(z):
    z = max(-500, min(500, z))
    return 1.0 / (1.0 + math.exp(-z))

random.seed(7)

# Basit bir e-posta veri seti simülasyonu:
# Özellik 1: "şüpheli kelime sayısı" (ücretsiz, kazandınız, tıklayın vb.)
# Özellik 2: "bilinmeyen gönderici mi" (0=hayır, 1=evet)
# Etiket: 0=normal mail, 1=spam

N = 300
X, y = [], []

# Normal mailler: az şüpheli kelime, çoğunlukla tanıdık gönderici
for _ in range(N // 2):
    suspicious_words = max(0, random.gauss(1, 1))
    unknown_sender = 1 if random.random() < 0.15 else 0
    X.append([suspicious_words, unknown_sender])
    y.append(0)

# Spam mailler: çok şüpheli kelime, çoğunlukla bilinmeyen gönderici
for _ in range(N // 2):
    suspicious_words = max(0, random.gauss(6, 2))
    unknown_sender = 1 if random.random() < 0.85 else 0
    X.append([suspicious_words, unknown_sender])
    y.append(1)

combined = list(zip(X, y))
random.shuffle(combined)
X, y = zip(*combined)
X, y = list(X), list(y)


class LogisticRegression:
    def __init__(self, n_features, learning_rate=0.05):
        self.weights = [0.0] * n_features
        self.bias = 0.0
        self.lr = learning_rate

    def predict_proba(self, x):
        z = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
        return sigmoid(z)

    def predict(self, x, threshold=0.5):
        return 1 if self.predict_proba(x) >= threshold else 0

    def fit(self, X, y, epochs=500):
        n = len(y)
        n_features = len(X[0])
        for epoch in range(epochs):
            dw = [0.0] * n_features
            db = 0.0
            for i in range(n):
                p = self.predict_proba(X[i])
                error = p - y[i]
                for j in range(n_features):
                    dw[j] += error * X[i][j]
                db += error
            for j in range(n_features):
                self.weights[j] -= self.lr * (dw[j] / n)
            self.bias -= self.lr * (db / n)
        return self


class Metrics:
    def __init__(self, y_true, y_pred):
        self.tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        self.tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
        self.fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        self.fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    def precision(self):
        d = self.tp + self.fp
        return self.tp / d if d > 0 else 0

    def recall(self):
        d = self.tp + self.fn
        return self.tp / d if d > 0 else 0


split = int(0.8 * N)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = LogisticRegression(n_features=2)
model.fit(X_train, y_train, epochs=500)

print("=== E-posta Spam Filtresi -- Gerçek Dünya Senaryosu ===\n")
print("Özellikler: [şüpheli kelime sayısı, bilinmeyen gönderici mi]\n")

# Gerçek bir mail örneği test edelim
test_emails = [
    ([0.5, 0], "Patronundan gelen normal bir toplantı maili"),
    ([7.0, 1], "'ÜCRETSİZ KAZANDINIZ! Hemen tıklayın!' -- bilinmeyen gönderici"),
    ([3.0, 1], "Belirsiz -- biraz şüpheli ama tanıdık olmayan biri"),
]

for features, description in test_emails:
    prob = model.predict_proba(features)
    print(f"  {description}")
    print(f"    Spam olasılığı: {prob:.2%}\n")

print("--- Neden threshold önemli: spam filtresinde iş kararı ---")
print("Precision düşükse: normal mailler yanlışlıkla SPAM'e düşer (kullanıcı sinirlenir, önemli maili kaçırabilir)")
print("Recall düşükse: gerçek spam mailler gelen kutusuna sızar (can sıkıcı ama geri dönüşü var)\n")

for t in [0.3, 0.5, 0.7, 0.9]:
    y_pred_t = [1 if model.predict_proba(x) >= t else 0 for x in X_test]
    m = Metrics(y_test, y_pred_t)
    verdict = "Çok agresif -- normal mailleri de spam'e atabilir" if t <= 0.3 else \
              "Dengeli" if t <= 0.7 else "Çok temkinli -- bazı spam'ler gelen kutusuna sızabilir"
    print(f"Threshold={t}  Precision={m.precision():.3f}  Recall={m.recall():.3f}  -> {verdict}")

print("\n=== ÖZET ===")
print("1. Model, iki basit özellikten (şüpheli kelime + bilinmeyen gönderici) yola çıkarak")
print("   spam olasılığını başarıyla tahmin etti.")
print("2. Threshold, iş kararına göre ayarlanır: spam filtresinde genelde YÜKSEK threshold")
print("   tercih edilir çünkü normal bir maili kaybetmek, birkaç spam'i kaçırmaktan daha kötüdür.")
print("3. Precision ve Recall birbirine ZIT yönde hareket eder -- ikisini birden maksimize")
print("   edemezsin, işin doğasına göre hangisine öncelik vereceğine karar vermen gerekir.")
