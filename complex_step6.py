import math

class Complex:
    def __init__(self, real, imag=0.0):
        self.real = real
        self.imag = imag
    def __add__(self, other):
        return Complex(self.real + other.real, self.imag + other.imag)
    def magnitude(self):
        return math.sqrt(self.real ** 2 + self.imag ** 2)
    def __repr__(self):
        sign = "+" if self.imag >= 0 else "-"
        return f"{self.real:.4f} {sign} {abs(self.imag):.4f}i"

def euler(theta):
    return Complex(math.cos(theta), math.sin(theta))

def roots_of_unity(N):
    return [euler(2 * math.pi * k / N) for k in range(N)]

N = 8
roots = roots_of_unity(N)

print(f"{N}. dereceden birlik kökleri:")
for k, r in enumerate(roots):
    print(f"  k={k}: {r}  (magnitude={r.magnitude():.6f})")

total_real = sum(r.real for r in roots)
total_imag = sum(r.imag for r in roots)
print(f"\nTüm köklerin toplamı: {total_real:.10f} + {total_imag:.10f}i  (sıfır olmalı, N>1 için)")

print(f"\nHer kökün büyüklüğü tam olarak 1 mi? {all(abs(r.magnitude()-1) < 1e-9 for r in roots)}")
