import numpy as np

def f(x):
    return x[0]**2 + 10*x[1]**2  # bilerek "dar vadili" bir fonksiyon

def gradient(x):
    return np.array([2*x[0], 20*x[1]])

def hessian(x):
    return np.array([[2, 0], [0, 20]])

print("--- Gradient Descent ---")
x_gd = np.array([5.0, 5.0])
lr = 0.09
for step in range(50):
    x_gd = x_gd - lr * gradient(x_gd)
    if step % 10 == 0 or step == 49:
        print(f"step {step:2d}  x={x_gd}  f(x)={f(x_gd):.8f}")

print("\n--- Newton's Method ---")
x_newton = np.array([5.0, 5.0])
for step in range(5):
    H = hessian(x_newton)
    grad = gradient(x_newton)
    x_newton = x_newton - np.linalg.inv(H) @ grad
    print(f"step {step:2d}  x={x_newton}  f(x)={f(x_newton):.8f}")
