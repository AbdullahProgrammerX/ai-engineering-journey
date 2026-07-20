def numerical_derivative(f, x, h=1e-7):
    return (f(x + h) - f(x - h)) / (2 * h)

def f(x):
    return x**4 - 3*x**2

print("--- Momentum'suz gradient descent ---")
x = 2.0
lr = 0.01
for step in range(50):
    grad = numerical_derivative(f, x)
    x = x - lr * grad
    if step % 10 == 0 or step == 49:
        print(f"step {step:2d}  x={x:.4f}  f(x)={f(x):.6f}")

print("\n--- Momentum'lu gradient descent ---")
x = 2.0
lr = 0.01
velocity = 0.0
beta = 0.9  # momentum katsayısı

for step in range(50):
    grad = numerical_derivative(f, x)
    velocity = beta * velocity - lr * grad
    x = x + velocity
    if step % 10 == 0 or step == 49:
        print(f"step {step:2d}  x={x:.4f}  f(x)={f(x):.6f}")
