import numpy as np
from scipy.optimize import minimize

# Problem: minimize f(x,y) = x^2 + y^2, kısıt: x + y = 1
# Dersteki elle çözüm: x=y=0.5, lambda=-1

def f(point):
    x, y = point
    return x**2 + y**2

def constraint(point):
    x, y = point
    return x + y - 1  # bu = 0 olmalı

result = minimize(f, x0=[0, 0], constraints={'type': 'eq', 'fun': constraint})

print(f"Kısıtlı minimum: x={result.x[0]:.4f}, y={result.x[1]:.4f}")
print(f"f(x,y) = {result.fun:.4f}")
print(f"Beklenen (elle hesaplanan): x=0.5, y=0.5, f=0.5")

print("\n--- Geometrik doğrulama ---")
print("Kısıt: x + y = 1 (bir doğru)")
print("f'nin gradyanı: [2x, 2y]")
print(f"Çözüm noktasında gradyan: [{2*result.x[0]:.4f}, {2*result.x[1]:.4f}]")
print("Kısıtın gradyanı (x+y-1'in): [1, 1]")
print("İkisi paralel mi (aynı yönü mü gösteriyor)? Evet -- ikisi de [1,1] yönünde")
print("(Bu, Lagrange çarpanlarının temel geometrik sezgisi: çözümde f'nin gradyanı,")
print("kısıtın gradyanına PARALEL olmalı.)")
