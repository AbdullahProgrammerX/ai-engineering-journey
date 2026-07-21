# Bir paranın hileli olup olmadığını tahmin etmek
# Beta(a, b) dağılımı: a = "yazı" sayısı + 1, b = "tura" sayısı + 1

def beta_mean(a, b):
    return a / (a + b)

# Gün 1: Hiç veri yok -- düz/tarafsız önsel (prior)
a, b = 1, 1
print(f"Gün 1 (prior):      Beta({a},{b})  ortalama={beta_mean(a,b):.3f}")

# Gün 2: 7 yazı, 3 tura gözlemlendi
heads, tails = 7, 3
a, b = a + heads, b + tails
print(f"Gün 2 (posterior):  Beta({a},{b})  ortalama={beta_mean(a,b):.3f}")

# Gün 3: dünün posterior'ı bugünün prior'ı oluyor, yeni veri: 5 yazı, 5 tura
heads, tails = 5, 5
a, b = a + heads, b + tails
print(f"Gün 3 (posterior):  Beta({a},{b})  ortalama={beta_mean(a,b):.3f}")

print("\nSezgi: Her gün yeni veri geldikçe, dünün 'inancımız' bugünün başlangıç noktası oluyor.")
print("Tüm veriyi tek seferde işlemekle, gün gün işlemek MATEMATİKSEL OLARAK AYNI sonucu verir.")
print("Bu yüzden online öğrenme, A/B testing ve streaming sistemler bu mantığı kullanır.")
