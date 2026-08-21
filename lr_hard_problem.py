import numpy as np

np.random.seed(42)

def make_spiral_data(n_per_class=300, n_classes=2, noise=0.2):
    X = np.zeros((n_per_class * n_classes, 2))
    y = np.zeros(n_per_class * n_classes)
    for class_idx in range(n_classes):
        ix = range(n_per_class * class_idx, n_per_class * (class_idx + 1))
        r = np.linspace(0.1, 1, n_per_class)
        t = np.linspace(class_idx * 4, (class_idx + 1) * 4, n_per_class) + np.random.randn(n_per_class) * noise
        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[ix] = class_idx
    return X, y

X, y = make_spiral_data(n_per_class=300, noise=0.25)
print(f"Veri seti: {len(X)} örnek (iç içe geçmiş spiral -- zor, doğrusal olmayan bir problem)\n")


def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


class DeeperNetwork:
    def __init__(self, input_size=2, hidden1=32, hidden2=16, seed=0):
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(input_size, hidden1) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden1)
        self.W2 = rng.randn(hidden1, hidden2) * np.sqrt(2.0 / hidden1)
        self.b2 = np.zeros(hidden2)
        self.W3 = rng.randn(hidden2, 1) * np.sqrt(2.0 / hidden2)
        self.b3 = np.zeros(1)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = relu(self.z2)
        self.z3 = self.a2 @ self.W3 + self.b3
        self.out = sigmoid(self.z3)
        return self.out

    def backward(self, X, y_true, lr):
        n = X.shape[0]
        y_true = y_true.reshape(-1, 1)

        d_out = (self.out - y_true) / n
        dW3 = self.a2.T @ d_out
        db3 = d_out.sum(axis=0)

        d_a2 = d_out @ self.W3.T
        d_z2 = d_a2 * relu_deriv(self.z2)
        dW2 = self.a1.T @ d_z2
        db2 = d_z2.sum(axis=0)

        d_a1 = d_z2 @ self.W2.T
        d_z1 = d_a1 * relu_deriv(self.z1)
        dW1 = X.T @ d_z1
        db1 = d_z1.sum(axis=0)

        self.W3 -= lr * dW3; self.b3 -= lr * db3
        self.W2 -= lr * dW2; self.b2 -= lr * db2
        self.W1 -= lr * dW1; self.b1 -= lr * db1

    def loss(self, y_true):
        y_true = y_true.reshape(-1, 1)
        eps = 1e-15
        p = np.clip(self.out, eps, 1 - eps)
        return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

    def accuracy(self, y_true):
        preds = (self.out.flatten() >= 0.5).astype(float)
        return np.mean(preds == y_true) * 100


def constant_schedule(step, lr=0.01, **kwargs):
    return lr

def cosine_schedule(step, lr=0.01, total_steps=1000, lr_min=1e-5, **kwargs):
    if step >= total_steps:
        return lr_min
    return lr_min + 0.5 * (lr - lr_min) * (1 + np.cos(np.pi * step / total_steps))

def warmup_cosine_schedule(step, lr=0.01, total_steps=1000, warmup_steps=100, lr_min=1e-5, **kwargs):
    if step < warmup_steps:
        return lr * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return lr_min + 0.5 * (lr - lr_min) * (1 + np.cos(np.pi * progress))

def one_cycle_schedule(step, lr=0.01, total_steps=1000, **kwargs):
    mid = max(total_steps // 2, 1)
    if step < mid:
        return (lr / 25) + (lr - lr / 25) * step / mid
    else:
        progress = (step - mid) / max(total_steps - mid, 1)
        return lr * (1 - progress) + (lr / 10000) * progress


def train_full_batch(schedule_fn, X, y, epochs=2000, base_lr=0.5, **kwargs):
    net = DeeperNetwork(seed=0)
    total_steps = epochs
    for epoch in range(epochs):
        lr = schedule_fn(epoch, lr=base_lr, total_steps=total_steps, **kwargs)
        net.forward(X)
        net.backward(X, y, lr)
    net.forward(X)
    return net.loss(y), net.accuracy(y)


print("=== Zor Problem (spiral, 2 katmanlı ağ, 2000 epoch, base_lr=0.5) ===\n")
configs = [
    ("Constant", constant_schedule, {}),
    ("Cosine", cosine_schedule, {"lr_min": 1e-5}),
    ("Warmup+Cosine", warmup_cosine_schedule, {"warmup_steps": 100, "lr_min": 1e-5}),
    ("1cycle", one_cycle_schedule, {}),
]

print(f"{'Schedule':<20} {'Final Loss':>12} {'Accuracy':>10}")
print("-" * 45)
for name, fn, kwargs in configs:
    loss, acc = train_full_batch(fn, X, y, epochs=2000, base_lr=0.5, **kwargs)
    print(f"{name:<20} {loss:>12.6f} {acc:>9.2f}%")


print("\n=== LR Hassasiyeti (zor problem, sabit schedule) ===\n")
print(f"{'LR':>10} {'Final Loss':>12} {'Accuracy':>10} {'Durum':>15}")
print("-" * 55)
for lr in [5.0, 1.0, 0.5, 0.1, 0.01, 0.001]:
    loss, acc = train_full_batch(constant_schedule, X, y, epochs=2000, base_lr=lr)
    if np.isnan(loss) or loss > 5:
        status = "IRAKSADI"
    elif acc > 90:
        status = "YAKINSADI"
    elif acc > 60:
        status = "ÖĞRENİYOR"
    else:
        status = "ÖĞRENEMEDİ"
    print(f"{lr:>10.4f} {loss:>12.6f} {acc:>9.2f}% {status:>15}")
