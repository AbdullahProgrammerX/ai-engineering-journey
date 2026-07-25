import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 0, 6])

diff = np.abs(a - b)
print(f"Fark vektörü: {diff}")

l1 = np.sum(diff)
l2 = np.sqrt(np.sum(diff**2))
linf = np.max(diff)

print(f"\nL1 (Manhattan):    {l1}")
print(f"L2 (Euclidean):     {l2:.4f}")
print(f"L-infinity (max):   {linf}")

print(f"\nSıralama kontrolü: L-inf <= L2 <= L1 ?")
print(f"{linf} <= {l2:.4f} <= {l1}  ->  {linf <= l2 <= l1}")

print("""
Neden bu sıralama her zaman doğru?
- L-inf, sadece EN BÜYÜK farkı alır -- diğer tüm boyutları görmezden gelir.
- L2, tüm farkların KARELERİNİN toplamının kökü -- büyük fark baskın olsa da
  diğer boyutlar biraz katkı yapar, bu yüzden L-inf'den büyük ya da eşittir.
- L1, tüm farkların DÜZ TOPLAMI -- hiçbir "sönümleme" yok, her boyut tam ağırlıkta
  katkı yapar, bu yüzden her zaman en büyük (veya eşit) çıkar.
Matematiksel olarak: n boyutlu uzayda L-inf <= L2 <= L1 <= sqrt(n)*L-inf
her zaman geçerlidir.
""")
