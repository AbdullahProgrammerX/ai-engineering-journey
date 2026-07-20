def numerical_derivative(f, x, h=1e-7):
    return (f(x + h) - f(x - h)) / (2 * h)

def numerical_second_derivative(f, x, h=1e-4):
    # Türev fonksiyonunu tekrar türev alarak ikinci türevi buluyoruz
    df = lambda x: numerical_derivative(f, x, h=1e-5)
    return numerical_derivative(df, x, h)

def f(x):
    return x ** 3

x = 2.0
second_deriv = numerical_second_derivative(f, x)
print(f"f(x) = x^3, x={x}")
print(f"Sayısal ikinci türev: {second_deriv:.6f}")
print(f"Analitik ikinci türev (6x): {6*x:.1f}")
