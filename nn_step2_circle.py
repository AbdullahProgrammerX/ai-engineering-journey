import math
import random

def sigmoid(x):
    x = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x))

class Layer:
    def __init__(self, n_inputs, n_neurons, weights=None, biases=None):
        if weights is not None:
            self.weights = weights
        else:
            self.weights = [[random.uniform(-1, 1) for _ in range(n_inputs)] for _ in range(n_neurons)]
        self.biases = biases if biases is not None else [0.0] * n_neurons

    def forward(self, inputs):
        self.last_output = []
        for neuron_idx in range(len(self.weights)):
            z = sum(w * x for w, x in zip(self.weights[neuron_idx], inputs)) + self.biases[neuron_idx]
            self.last_output.append(sigmoid(z))
        return self.last_output

class Network:
    def __init__(self, layers):
        self.layers = layers
    def forward(self, inputs):
        current = inputs
        for layer in self.layers:
            current = layer.forward(current)
        return current


random.seed(42)
data = []
for _ in range(200):
    x = random.uniform(-1, 1)
    y = random.uniform(-1, 1)
    label = 1 if (x*x + y*y) < 0.25 else 0
    data.append(([x, y], label))

circle_net = Network([
    Layer(n_inputs=2, n_neurons=8),
    Layer(n_inputs=8, n_neurons=1),
])

correct = 0
for inputs, expected in data:
    result = circle_net.forward(inputs)
    predicted = 1 if result[0] >= 0.5 else 0
    if predicted == expected:
        correct += 1

majority_class_acc = max(sum(1 for _, l in data if l==0), sum(1 for _, l in data if l==1)) / len(data)

print(f"Rastgele ağırlıklarla doğruluk: {correct}/{len(data)} ({100*correct/len(data):.1f}%)")
print(f"Çoğunluk sınıfı baseline'ı: {100*majority_class_acc:.1f}%")
print("\nBu, henüz HİÇ ÖĞRENMEMİŞ bir ağın davranışı -- forward pass çalışıyor ama")
print("ağırlıklar rastgele olduğu için tahminler anlamsız. Lesson 03'te backpropagation")
print("ile bu ağırlıkları ÖĞRENECEĞİZ, sonra bu aynı mimari daire sınırını doğru çizecek.")
