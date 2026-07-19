import random

random.seed(42)


class Matrix:
    def __init__(self, data):
        self.data = data
        self.rows = len(data)
        self.cols = len(data[0])

    def __matmul__(self, other):
        result = [[0] * other.cols for _ in range(self.rows)]
        for i in range(self.rows):
            for j in range(other.cols):
                result[i][j] = sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
        return Matrix(result)

    def add_bias(self, bias):
        # bias: 1 x N şeklinde bir Matrix, her satıra ekle
        result = [[self.data[i][j] + bias.data[0][j] for j in range(self.cols)] for i in range(self.rows)]
        return Matrix(result)

    def relu(self):
        result = [[max(0, val) for val in row] for row in self.data]
        return Matrix(result)

    def shape(self):
        return (self.rows, self.cols)

    def __repr__(self):
        return "\n".join(str(row) for row in self.data)


def random_matrix(rows, cols):
    return Matrix([[random.uniform(-1, 1) for _ in range(cols)] for _ in range(rows)])


# Girdi: batch_size=1, input_dim=3
X = Matrix([[0.5, -0.2, 0.1]])
print(f"Girdi X şekli: {X.shape()}")

# Katman 1: input(3) -> hidden(4)
W1 = random_matrix(3, 4)
b1 = random_matrix(1, 4)
print(f"W1 şekli: {W1.shape()}, b1 şekli: {b1.shape()}")

hidden = (X @ W1).add_bias(b1).relu()
print(f"Gizli katman çıktısı şekli: {hidden.shape()}")
print(f"Gizli katman değerleri: {hidden}")

# Katman 2: hidden(4) -> output(2)
W2 = random_matrix(4, 2)
b2 = random_matrix(1, 2)
print(f"\nW2 şekli: {W2.shape()}, b2 şekli: {b2.shape()}")

output = (hidden @ W2).add_bias(b2)
print(f"Çıktı katmanı şekli: {output.shape()}")
print(f"Çıktı değerleri: {output}")

print("\n--- Şekil doğrulaması ---")
assert X.shape() == (1, 3)
assert hidden.shape() == (1, 4)
assert output.shape() == (1, 2)
print("Tüm şekiller doğru: [1,3] -> [1,4] -> [1,2]")
