import math
import random

def sigmoid(x):
    x = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x))

class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = [[random.uniform(-1, 1) for _ in range(n_inputs)] for _ in range(n_neurons)]
        self.biases = [0.0] * n_neurons

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
        all_outputs = [inputs]
        for layer in self.layers:
            current = layer.forward(current)
            all_outputs.append(current)
        return current, all_outputs


random.seed(1)
# 2-4-2-1 mimari: 2 girdi, 4 nöronlu gizli katman, 2 nöronlu gizli katman, 1 çıktı
net = Network([
    Layer(n_inputs=2, n_neurons=4),
    Layer(n_inputs=4, n_neurons=2),
    Layer(n_inputs=2, n_neurons=1),
])

xor_data = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]

print("=== 2-4-2-1 Mimari: Her katmanda temsil nasıl dönüşüyor ===\n")
for inputs, expected in xor_data:
    final_output, all_outputs = net.forward(inputs)
    print(f"Girdi: {inputs}")
    print(f"  Katman 1 (4 nöron) çıktısı: {[round(v,3) for v in all_outputs[1]]}")
    print(f"  Katman 2 (2 nöron) çıktısı: {[round(v,3) for v in all_outputs[2]]}")
    print(f"  Katman 3 (1 nöron, final) çıktısı: {round(all_outputs[3][0],3)}")
    print()
