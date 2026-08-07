import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

np.random.seed(42)

# Gerçekçi kredi kartı işlem verisi simülasyonu
n_transactions = 5000
fraud_ratio = 0.017  # gerçek dünyada dolandırıcılık genelde %1-2 civarı

is_fraud = (np.random.random(n_transactions) < fraud_ratio).astype(int)

# Özellikler: normal işlemler vs dolandırıcılık işlemleri farklı dağılımlardan geliyor
transaction_amount = np.where(
    is_fraud == 1,
    np.random.exponential(800, n_transactions),   # dolandırıcılık: genelde büyük tutarlar
    np.random.exponential(60, n_transactions)      # normal: küçük tutarlar
)

hour_of_day = np.where(
    is_fraud == 1,
    np.random.normal(3, 2, n_transactions) % 24,   # dolandırıcılık: gece saatleri
    np.random.normal(14, 5, n_transactions) % 24    # normal: gündüz
)

distance_from_home = np.where(
    is_fraud == 1,
    np.random.exponential(500, n_transactions),    # dolandırıcılık: uzak mesafe
    np.random.exponential(10, n_transactions)       # normal: yakın
)

num_transactions_today = np.where(
    is_fraud == 1,
    np.random.poisson(1, n_transactions),
    np.random.poisson(3, n_transactions)
)

X = np.column_stack([transaction_amount, hour_of_day, distance_from_home, num_transactions_today])
y = is_fraud

print("=== Kredi Kartı Dolandırıcılık Tespiti -- Gerçek Dünya Senaryosu ===\n")
print(f"Toplam işlem: {n_transactions}")
print(f"Dolandırıcılık oranı: %{y.mean()*100:.2f} ({y.sum()} işlem)\n")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# --- YANLIŞ YAKLAŞIM: sadece accuracy'ye bakmak ---
print("--- YANLIŞ YAKLAŞIM: 'Hiçbir işlem dolandırıcılık değil' de ---")
naive_pred = np.zeros(len(y_test))
print(f"Accuracy: {accuracy_score(y_test, naive_pred):.4f}  <- ETKİLEYİCİ görünüyor ama...")
print(f"Recall (yakalanan dolandırıcılık oranı): {recall_score(y_test, naive_pred, zero_division=0):.4f}  <- SIFIR!")
print("Bu model, TÜM dolandırıcılık işlemlerini kaçırıyor ama %98+ accuracy alıyor.\n")

# --- DOĞRU YAKLAŞIM: dengeli model + doğru metrikler ---
print("--- DOĞRU YAKLAŞIM: class_weight='balanced' + doğru metrikler ---")
rf = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
y_scores = rf.predict_proba(X_test)[:, 1]

print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
print(f"AUC-ROC:   {roc_auc_score(y_test, y_scores):.4f}")

print(f"\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"                  Tahmin: Normal  Tahmin: Fraud")
print(f"Gerçek: Normal        {cm[0,0]:>6d}         {cm[0,1]:>6d}")
print(f"Gerçek: Fraud         {cm[1,0]:>6d}         {cm[1,1]:>6d}")

# --- Cross-validation ile modelin GÜVENİLİRLİĞİNİ test et ---
print("\n--- Stratified Cross-Validation (5-fold) -- tek sonuca güvenme ---")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf, X, y, cv=skf, scoring='recall')
print(f"Recall skorları (her fold): {np.round(cv_scores, 4)}")
print(f"Ortalama: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# --- Threshold ayarı: iş kararı ---
print("\n--- İş Kararı: Threshold Ayarı ---")
print("Banka için maliyet dengesi:")
print("  - Kaçırılan dolandırıcılık (False Negative): müşteri parası kaybolur, itibar zararı BÜYÜK")
print("  - Yanlış alarm (False Positive): müşteri rahatsız olur ama işlem onaylanabilir, maliyet KÜÇÜK")
print("  -> Bu yüzden RECALL öncelenmeli, threshold düşürülmeli\n")

for threshold in [0.5, 0.3, 0.15]:
    y_pred_t = (y_scores >= threshold).astype(int)
    prec = precision_score(y_test, y_pred_t, zero_division=0)
    rec = recall_score(y_test, y_pred_t, zero_division=0)
    caught = int(rec * y_test.sum())
    total_fraud = y_test.sum()
    print(f"Threshold={threshold:.2f}  Precision={prec:.3f}  Recall={rec:.3f}  "
          f"-> {caught}/{total_fraud} dolandırıcılık işlemi yakalandı")
