def numerical_gradient(f, point, h=1e-7):
    gradient = []
    for i in range(len(point)):
        point_plus = list(point)
        point_minus = list(point)
        point_plus[i] += h
        point_minus[i] -= h
        partial = (f(point_plus) - f(point_minus)) / (2 * h)
        gradient.append(partial)
    return gradient

def f(point):
    x, y = point
    return (x - 3)**2 + (y + 1)**2

point = [0.0, 0.0]
lr = 0.1

for step in range(50):
    grad = numerical_gradient(f, point)
    point = [p - lr * g for p, g in zip(point, grad)]
    if step % 10 == 0 or step == 49:
        print(f"step {step:2d}  point=({point[0]:.4f}, {point[1]:.4f})  f={f(point):.6f}")

print(f"\nBeklenen minimum: (3, -1)")
print(f"Bulunan minimum: ({point[0]:.4f}, {point[1]:.4f})")
