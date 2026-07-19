class Matrix:
    def __init__(self, data):
        self.data = data  # liste liste: [[a,b],[c,d]]
        self.rows = len(data)
        self.cols = len(data[0])

    def __matmul__(self, other):
        result = [[0] * other.cols for _ in range(self.rows)]
        for i in range(self.rows):
            for j in range(other.cols):
                result[i][j] = sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
        return Matrix(result)

    def determinant_2x2(self):
        a, b = self.data[0]
        c, d = self.data[1]
        return a * d - b * c

    def inverse_2x2(self):
        det = self.determinant_2x2()
        if det == 0:
            raise ValueError("Determinant 0 -- matrisin tersi yok (singular matrix)")
        a, b = self.data[0]
        c, d = self.data[1]
        factor = 1 / det
        return Matrix([
            [d * factor, -b * factor],
            [-c * factor, a * factor]
        ])

    def __repr__(self):
        return "\n".join(str(row) for row in self.data)


def is_identity(m, tol=1e-9):
    for i in range(m.rows):
        for j in range(m.cols):
            expected = 1 if i == j else 0
            if abs(m.data[i][j] - expected) > tol:
                return False
    return True


test_matrices = [
    [[4, 7], [2, 6]],
    [[1, 2], [3, 4]],
    [[2, 0], [0, 2]],
]

for data in test_matrices:
    A = Matrix(data)
    print(f"A =\n{A}")
    try:
        A_inv = A.inverse_2x2()
        result = A @ A_inv
        print(f"A @ A_inv =\n{result}")
        print(f"Birim matris mi? {is_identity(result)}\n")
    except ValueError as e:
        print(f"Hata: {e}\n")

# Determinant 0 olan durum
print("--- Determinant 0 örneği ---")
singular = Matrix([[2, 4], [1, 2]])  # 2. satır 1. satırın yarısı -> lineer bağımlı
print(f"Determinant: {singular.determinant_2x2()}")
try:
    singular.inverse_2x2()
except ValueError as e:
    print(f"Beklenen hata: {e}")
