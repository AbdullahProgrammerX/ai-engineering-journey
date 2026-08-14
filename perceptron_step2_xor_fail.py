class Perceptron:
    def __init__(self, n_inputs, learning_rate=0.1):
        self.weights = [0.0] * n_inputs
        self.bias = 0.0
        self.lr = learning_rate

    def predict(self, inputs):
        total = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        return 1 if total >= 0 else 0

    def train(self, training_data, epochs=100):
        for epoch in range(epochs):
            errors = 0
            for inputs, target in training_data:
                prediction = self.predict(inputs)
                error = target - prediction
                if error != 0:
                    errors += 1
                    for i in range(len(self.weights)):
                        self.weights[i] += self.lr * error * inputs[i]
                    self.bias += self.lr * error
            if errors == 0:
                print(f"  {epoch + 1}. epoch'ta yakınsadı")
                return
        print(f"  {epochs} epoch sonunda YAKINSAMADI")


xor_data = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]

print("=== XOR Kapısı (tek perceptron) ===")
p_xor = Perceptron(2)
p_xor.train(xor_data, epochs=1000)
for inputs, expected in xor_data:
    result = p_xor.predict(inputs)
    status = "OK" if result == expected else "YANLIŞ"
    print(f"  {inputs} -> {result} (beklenen {expected}) {status}")

print("\nBu, tek bir perceptronun XOR'u ÖĞRENEMEYECEĞİNİN kesin kanıtı --")
print("1000 epoch bile yetmiyor çünkü matematiksel olarak İMKANSIZ.")
print("XOR, DOĞRUSAL OLARAK AYRILAMAZ (tek bir düz çizgiyle 0'lar ve 1'ler ayrılamaz).")
