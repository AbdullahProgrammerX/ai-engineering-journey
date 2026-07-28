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
    def magnitude(self):
        return math.sqrt(self.real ** 2 + self.imag ** 2)

def euler(theta):
    return Complex(math.cos(theta), math.sin(theta))

def dft(signal):
    N = len(signal)
    result = []
    for k in range(N):
        total = Complex(0, 0)
        for n in range(N):
            angle = -2 * math.pi * k * n / N
            total = total + Complex(signal[n], 0) * euler(angle)
        result.append(total)
    return result

def idft(spectrum):
    N = len(spectrum)
    result = []
    for n in range(N):
        total = Complex(0, 0)
        for k in range(N):
            angle = 2 * math.pi * k * n / N
            total = total + spectrum[k] * euler(angle)
        result.append(Complex(total.real / N, total.imag / N))
    return result


signal = [1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0]
print(f"Orijinal sinyal: {signal}")

spectrum = dft(signal)
print("\nDFT (frekans spektrumu):")
for k, x in enumerate(spectrum):
    print(f"  k={k}: magnitude={x.magnitude():.4f}")

reconstructed = idft(spectrum)
reconstructed_reals = [round(x.real, 6) for x in reconstructed]
print(f"\nTers DFT ile geri inşa edilen sinyal: {reconstructed_reals}")
print(f"Orijinalle aynı mı? {all(abs(a-b) < 1e-9 for a, b in zip(signal, reconstructed_reals))}")
