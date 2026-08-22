import math
import random
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

random.seed(42)

# ============ FRAMEWORK ============

class Module:
    def __init__(self):
        self.training = True
    def forward(self, x):
        raise NotImplementedError
    def backward(self, grad):
        raise NotImplementedError
    def parameters(self):
        return []
    def train(self):
        self.training = True
    def eval(self):
        self.training = False


class Linear(Module):
    def __init__(self, fan_in, fan_out):
        super().__init__()
        std = math.sqrt(2.0 / fan_in)
        self.weights = [[random.gauss(0, std) for _ in range(fan_in)] for _ in range(fan_out)]
        self.biases = [0.0] * fan_out
        self.weight_grads = [[0.0] * fan_in for _ in range(fan_out)]
        self.bias_grads = [0.0] * fan_out
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.input = None

    def forward(self, x):
        self.input = x
        output = []
        for i in range(self.fan_out):
            val = self.biases[i]
            for j in range(self.fan_in):
                val += self.weights[i][j] * x[j]
            output.append(val)
        return output

    def backward(self, grad):
        input_grad = [0.0] * self.fan_in
        for i in range(self.fan_out):
            self.bias_grads[i] += grad[i]
            for j in range(self.fan_in):
                self.weight_grads[i][j] += grad[i] * self.input[j]
                input_grad[j] += grad[i] * self.weights[i][j]
        return input_grad

    def parameters(self):
        params = []
        for i in range(self.fan_out):
            for j in range(self.fan_in):
                params.append((self.weights, i, j, self.weight_grads))
            params.append((self.biases, i, None, self.bias_grads))
        return params


class ReLU(Module):
    def __init__(self):
        super().__init__()
        self.mask = None
    def forward(self, x):
        self.mask = [1.0 if v > 0 else 0.0 for v in x]
        return [max(0.0, v) for v in x]
    def backward(self, grad):
        return [g * m for g, m in zip(grad, self.mask)]


class Sigmoid(Module):
    def __init__(self):
        super().__init__()
        self.output = None
    def forward(self, x):
        self.output = []
        for v in x:
            v = max(-500, min(500, v))
            self.output.append(1.0 / (1.0 + math.exp(-v)))
        return self.output
    def backward(self, grad):
        return [g * o * (1 - o) for g, o in zip(grad, self.output)]


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.mask = None
    def forward(self, x):
        if not self.training:
            return x
        self.mask = [0.0 if random.random() < self.p else 1.0 / (1 - self.p) for _ in x]
        return [v * m for v, m in zip(x, self.mask)]
    def backward(self, grad):
        if self.mask is None:
            return grad
        return [g * m for g, m in zip(grad, self.mask)]


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = list(modules)
    def forward(self, x):
        for module in self.modules:
            x = module.forward(x)
        return x
    def backward(self, grad):
        for module in reversed(self.modules):
            grad = module.backward(grad)
        return grad
    def parameters(self):
        params = []
        for module in self.modules:
            params.extend(module.parameters())
        return params
    def train(self):
        self.training = True
        for module in self.modules:
            module.train()
    def eval(self):
        self.training = False
        for module in self.modules:
            module.eval()


class BCELoss:
    def __call__(self, predicted, target):
        self.predicted = predicted
        self.target = target
        eps = 1e-7
        n = len(predicted)
        self.loss = 0
        for p, t in zip(predicted, target):
            p = max(eps, min(1 - eps, p))
            self.loss += -(t * math.log(p) + (1 - t) * math.log(1 - p))
        self.loss /= n
        return self.loss
    def backward(self):
        eps = 1e-7
        n = len(self.predicted)
        grads = []
        for p, t in zip(self.predicted, self.target):
            p = max(eps, min(1 - eps, p))
            grads.append((-t / p + (1 - t) / (1 - p)) / n)
        return grads


class Adam:
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = [0.0] * len(parameters)
        self.v = [0.0] * len(parameters)

    def step(self):
        self.t += 1
        for idx, (container, i, j, grad_container) in enumerate(self.params):
            g = grad_container[i][j] if j is not None else grad_container[i]
            self.m[idx] = self.beta1 * self.m[idx] + (1 - self.beta1) * g
            self.v[idx] = self.beta2 * self.v[idx] + (1 - self.beta2) * g * g
            m_hat = self.m[idx] / (1 - self.beta1 ** self.t)
            v_hat = self.v[idx] / (1 - self.beta2 ** self.t)
            update = self.lr * m_hat / (math.sqrt(v_hat) + self.eps)
            if j is not None:
                container[i][j] -= update
            else:
                container[i] -= update

    def zero_grad(self):
        for container, i, j, grad_container in self.params:
            if j is not None:
                grad_container[i][j] = 0.0
            else:
                grad_container[i] = 0.0


class DataLoader:
    def __init__(self, data, batch_size=32, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
    def __iter__(self):
        indices = list(range(len(self.data)))
        if self.shuffle:
            random.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start:start + self.batch_size]
            batch = [self.data[i] for i in batch_indices]
            inputs = [item[0] for item in batch]
            targets = [item[1] for item in batch]
            yield inputs, targets
    def __len__(self):
        return (len(self.data) + self.batch_size - 1) // self.batch_size


# ============ GERÇEK VERİ: Breast Cancer Wisconsin ============

data_bunch = load_breast_cancer()
X_raw = data_bunch.data.tolist()
y_raw = data_bunch.target.tolist()  # 0=malignant, 1=benign

print(f"Gerçek veri seti: {len(X_raw)} hasta, {len(X_raw[0])} tıbbi özellik")
print(f"Sınıf dağılımı: {sum(y_raw)} iyi huylu (benign), {len(y_raw)-sum(y_raw)} kötü huylu (malignant)\n")

# Standartlaştırma (ZORUNLU -- özellikler çok farklı ölçekte, örn. "alan" vs "simetri")
n_features = len(X_raw[0])
means = [sum(row[j] for row in X_raw) / len(X_raw) for j in range(n_features)]
stds = []
for j in range(n_features):
    var = sum((row[j] - means[j]) ** 2 for row in X_raw) / len(X_raw)
    stds.append(math.sqrt(var) if var > 0 else 1.0)

X_scaled = [[(row[j] - means[j]) / stds[j] for j in range(n_features)] for row in X_raw]

full_data = [(X_scaled[i], [float(y_raw[i])]) for i in range(len(X_raw))]
random.shuffle(full_data)

split = int(0.8 * len(full_data))
train_data = full_data[:split]
test_data = full_data[split:]

print(f"Eğitim: {len(train_data)}, Test: {len(test_data)}\n")

# ============ MODEL ============

model = Sequential(
    Linear(30, 32),
    ReLU(),
    Dropout(p=0.2),
    Linear(32, 16),
    ReLU(),
    Dropout(p=0.2),
    Linear(16, 1),
    Sigmoid(),
)

criterion = BCELoss()
optimizer = Adam(model.parameters(), lr=0.01)
loader = DataLoader(train_data, batch_size=16, shuffle=True)

model.train()
print("=== Eğitim ===")
for epoch in range(60):
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for batch_inputs, batch_targets in loader:
        for x, t in zip(batch_inputs, batch_targets):
            pred = model.forward(x)
            loss = criterion(pred, t)
            total_loss += loss

            optimizer.zero_grad()
            grad = criterion.backward()
            model.backward(grad)
            optimizer.step()

            predicted_class = 1.0 if pred[0] >= 0.5 else 0.0
            if predicted_class == t[0]:
                total_correct += 1
            total_samples += 1

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples * 100

    if epoch % 10 == 0 or epoch == 59:
        print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Train Accuracy: {accuracy:.1f}%")

# ============ TEST DEĞERLENDİRME (eval modunda -- dropout kapalı!) ============

model.eval()
correct = 0
tp = tn = fp = fn = 0
for x, t in test_data:
    pred = model.forward(x)
    predicted_class = 1.0 if pred[0] >= 0.5 else 0.0
    actual = t[0]
    if predicted_class == actual:
        correct += 1
    if predicted_class == 1 and actual == 1: tp += 1
    if predicted_class == 0 and actual == 0: tn += 1
    if predicted_class == 1 and actual == 0: fp += 1
    if predicted_class == 0 and actual == 1: fn += 1

test_accuracy = correct / len(test_data) * 100
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0

print(f"\n=== Test Sonuçları (model.eval() modunda) ===")
print(f"Test Accuracy: {test_accuracy:.1f}% ({correct}/{len(test_data)})")
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")
print(f"Confusion: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
