import numpy as np


class Matrix3x3:
    def __init__(self, data):
        self.data = data

    def determinant(self):
        m = self.data
        return (
            m[0][0] * (m[1][1]*m[2][2] - m[1][2]*m[2][1])
            - m[0][1] * (m[1][0]*m[2][2] - m[1][2]*m[2][0])
            + m[0][2] * (m[1][0]*m[2][1] - m[1][1]*m[2][0])
        )

    def cofactor(self, row, col):
        minor = [
            [self.data[i][j] for j in range(3) if j != col]
            for i in range(3) if i != row
        ]
        minor_det = minor[0][0]*minor[1][1] - minor[0][1]*minor[1][0]
        sign = (-1) ** (row + col)
        return sign * minor_det

    def inverse(self):
        det = self.determinant()
        if det == 0:
            raise ValueError("Determinant 0 -- tersi yok")

        # Kofaktör matrisi
        cofactors = [[self.cofactor(i, j) for j in range(3)] for i in range(3)]
        # Adjugate = kofaktör matrisinin transpozu
        adjugate = [[cofactors[j][i] for j in range(3)] for i in range(3)]
        # Ters = adjugate / determinant
        inverse = [[adjugate[i][j] / det for j in range(3)] for i in range(3)]
        return inverse


A_data = [
    [2, -1, 0],
    [1, 3, 2],
    [0, 1, 4]
]

A = Matrix3x3(A_data)
my_inverse = A.inverse()

print("Kendi hesapladığımız ters:")
for row in my_inverse:
    print([round(x, 4) for x in row])

np_inverse = np.linalg.inv(np.array(A_data))
print("\nNumPy'ın hesapladığı ters:")
print(np.round(np_inverse, 4))

match = np.allclose(my_inverse, np_inverse)
print(f"\nİkisi eşleşiyor mu? {match}")
