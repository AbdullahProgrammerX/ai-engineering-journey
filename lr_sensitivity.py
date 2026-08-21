import math
import random

def constant_schedule(step, lr=0.01, **kwargs):
    return lr

def sigmoid(x):
    x = max(-500, min(500, x))
    return 1.0 / (1.0 + math.exp(-x))

def relu(x):
    return max(0.0, x)

def relu_deriv(x):
    return 1.0 if x > 0 else 0.0

def make_circle_data(n=200, seed=42):
    random.seed(seed)
    data = []
    for _ in range(n):
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
        label = 1.0 if x*x + y*y < 1.5 else 0.0
        data.append(([x, y], label))
    return data

def train_with_schedule(schedule_fn, data, epochs=300, base_lr=0.05, **kwargs):
    random.seed(0)
    hidden_size = 8
    total_steps = epochs * len(data)
    std = math.sqrt(2.0 / 2)
    w1 = [[random.gauss(0, std) for _ in range(2)] for _ in range(hidden_size)]
    b1 = [0.0] * hidden_size
    w2 = [random.gauss(0, std) for _ in range(hidden_size)]
    b2 = 0.0
    step = 0
    epoch_losses = []

    for epoch in range(epochs):
        total_loss = 0
        for x, target in data:
            lr = schedule_fn(step, lr=base_lr, total_steps=total_steps, **kwargs)
            z1, h = [], []
            for i in range(hidden_size):
                z = w1[i][0]*x[0] + w1[i][1]*x[1] + b1[i]
                z1.append(z); h.append(relu(z))
            z2 = sum(w2[i]*h[i] for i in range(hidden_size)) + b2
            out = sigmoid(z2)
            error = out - target
            d_out = error * out * (1 - out)
            for i in range(hidden_size):
                d_h = d_out * w2[i] * relu_deriv(z1[i])
                w2[i] -= lr * d_out * h[i]
                for j in range(2):
                    w1[i][j] -= lr * d_h * x[j]
                b1[i] -= lr * d_h
            b2 -= lr * d_out
            total_loss += (out - target) ** 2
            step += 1
        epoch_losses.append(total_loss / len(data))
    return epoch_losses


def lr_sensitivity(data):
    learning_rates = [1.0, 0.1, 0.01, 0.001, 0.0001]
    print("\nLR Hassasiyeti (sabit schedule, 100 epoch):")
    print(f"  {'LR':>10} {'Başlangıç Loss':>15} {'Bitiş Loss':>12} {'Durum':>15}")
    print("  " + "-" * 57)

    for lr in learning_rates:
        losses = train_with_schedule(constant_schedule, data, epochs=100, base_lr=lr)
        start = losses[0]
        end = losses[-1]

        if math.isnan(end) or end > start or end > 1.0:
            status = "IRAKSADI"
        elif end > start * 0.9:
            status = "ZAR ZOR HAREKETTİ"
        elif end < 0.15:
            status = "YAKINSADI"
        else:
            status = "ÖĞRENİYOR"

        end_str = f"{end:.6f}" if not math.isnan(end) else "NaN"
        print(f"  {lr:>10.4f} {start:>15.6f} {end_str:>12} {status:>15}")

data = make_circle_data()
lr_sensitivity(data)
