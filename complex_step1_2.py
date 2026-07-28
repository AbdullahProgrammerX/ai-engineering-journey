import math

class Complex:
    def __init__(self, real, imag=0.0):
        self.real = real
        self.imag = imag

    def __add__(self, other):
        return Complex(self.real + other.real, self.imag + other.imag)

    def __mul__(self, other):
        r = self.real * other.real - self.imag * other.imag
        i = self.real * other.imag + self.imag * other.real
        return Complex(r, i)

    def __truediv__(self, other):
        denom = other.real ** 2 + other.imag ** 2
        r = (self.real * other.real + self.imag * other.imag) / denom
        i = (self.imag * other.real - self.real * other.imag) / denom
        return Complex(r, i)

    def magnitude(self):
        return math.sqrt(self.real ** 2 + self.imag ** 2)

    def phase(self):
        return math.atan2(self.imag, self.real)

    def conjugate(self):
        return Complex(self.real, -self.imag)

    def __repr__(self):
        sign = "+" if self.imag >= 0 else "-"
        return f"{self.real:.4f} {sign} {abs(self.imag):.4f}i"


def to_polar(z):
    return z.magnitude(), z.phase()

def from_polar(r, theta):
    return Complex(r * math.cos(theta), r * math.sin(theta))

def euler(theta):
    return Complex(math.cos(theta), math.sin(theta))


print("--- Temel aritmetik ---")
z1 = Complex(3, 2)
z2 = Complex(1, 4)
print(f"z1={z1}, z2={z2}")
print(f"z1+z2 = {z1+z2}")
print(f"z1*z2 = {z1*z2}")
print(f"z1/z2 = {z1/z2}")
print(f"z1'in eşleniği = {z1.conjugate()}")

print("\n--- Euler formülü doğrulaması ---")
print(f"euler(0) = {euler(0)}  (beklenen: 1+0i)")
print(f"euler(pi/2) = {euler(math.pi/2)}  (beklenen: 0+1i)")
print(f"euler(pi) = {euler(math.pi)}  (beklenen: -1+0i)")
print(f"euler(theta) her zaman magnitude=1 mi? {euler(1.234).magnitude():.6f}")
