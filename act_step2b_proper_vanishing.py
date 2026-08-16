import math
import random

def sigmoid(x):
    x = max(-500, min(500, x))
    return 1.0 / (1.0 + math.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return max(0.0, x)

def relu_derivative(x):
    return 1.0 if x > 0 else 0.0

def gelu(x):
    return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))

def gelu_derivative(x):
    phi = 0.5 * (1 + math.erf(x / math.sqrt(2)))
    pdf = math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
    return phi + x * pdf

def proper_gradient_chain(deriv_fn, name, n_layers=15):
    random.seed(42)
    # Her katmanda BAĞIMSIZ rastgele bir aktivasyon girdisi simüle ediyoruz
    # ve GRADYANIN kendisinin çarpımsal olarak nasıl küçüldüğünü izliyoruz
    gradient = 1.0
    print(f"\n{name}: gradyanın geriye doğru (backprop) birikimli çarpımı")
    for layer in range(n_layers):
        x = random.gauss(0, 1)  # her katmanda BAĞIMSIZ rastgele girdi
        local_grad = deriv_fn(x)
        gradient *= local_grad
        bar = "#" * min(int(abs(gradient) * 50), 50)
        print(f"  Katman {layer+1:2d}: birikimli gradyan = {gradient:.10f}  {bar}")

proper_gradient_chain(sigmoid_derivative, "Sigmoid")
proper_gradient_chain(relu_derivative, "ReLU")
proper_gradient_chain(gelu_derivative, "GELU")
