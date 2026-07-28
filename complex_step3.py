import math

class Complex:
    def __init__(self, real, imag=0.0):
        self.real = real
        self.imag = imag
    def __mul__(self, other):
        r = self.real * other.real - self.imag * other.imag
        i = self.real * other.imag + self.imag * other.real
        return Complex(r, i)
    def magnitude(self):
        return math.sqrt(self.real ** 2 + self.imag ** 2)
    def phase(self):
        return math.atan2(self.imag, self.real)
    def __repr__(self):
        sign = "+" if self.imag >= 0 else "-"
        return f"{self.real:.4f} {sign} {abs(self.imag):.4f}i"

def euler(theta):
    return Complex(math.cos(theta), math.sin(theta))

point = Complex(3, 4)
print(f"Orijinal nokta: {point}")
print(f"Orijinal magnitude: {point.magnitude():.4f}, faz: {math.degrees(point.phase()):.2f} derece")

rotated = point * euler(math.pi / 4)  # 45 derece döndür
print(f"\n45 derece döndürülmüş: {rotated}")
print(f"Yeni magnitude: {rotated.magnitude():.4f}  (AYNI kalmalı!)")
print(f"Yeni faz: {math.degrees(rotated.phase()):.2f} derece  (45 derece artmış olmalı)")
