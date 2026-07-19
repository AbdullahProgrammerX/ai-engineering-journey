import math

def rotation_2d(theta):
    c, s = math.cos(theta), math.sin(theta)
    return [[c, -s], [s, c]]

def scaling_2d(sx, sy):
    return [[sx, 0], [0, sy]]

def shearing_2d(kx, ky):
    return [[1, kx], [ky, 1]]

def mat_mul(a, b):
    rows_a, cols_b = len(a), len(b[0])
    cols_a = len(a[0])
    return [
        [sum(a[i][k] * b[k][j] for k in range(cols_a)) for j in range(cols_b)]
        for i in range(rows_a)
    ]

def mat_vec_mul(matrix, vector):
    return [
        sum(matrix[i][j] * vector[j] for j in range(len(vector)))
        for i in range(len(matrix))
    ]

def det_2x2(matrix):
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

# 8 nokta, çember üzerinde
circle_points = [
    [math.cos(2 * math.pi * i / 8), math.sin(2 * math.pi * i / 8)]
    for i in range(8)
]

R = rotation_2d(math.radians(30))
S = scaling_2d(1.5, 0.8)
Sh = shearing_2d(0.3, 0)

# Birleşim: önce rotate, sonra scale, sonra shear -> Sh @ S @ R
composed = mat_mul(Sh, mat_mul(S, R))

print("--- Öncesi ve sonrası koordinatlar ---")
transformed_points = []
for i, p in enumerate(circle_points):
    t = mat_vec_mul(composed, p)
    transformed_points.append(t)
    print(f"Nokta {i}: önce=({p[0]:.4f}, {p[1]:.4f})  sonra=({t[0]:.4f}, {t[1]:.4f})")

print("\n--- Determinant doğrulaması ---")
det_composed = det_2x2(composed)
det_R = det_2x2(R)
det_S = det_2x2(S)
det_Sh = det_2x2(Sh)
product = det_R * det_S * det_Sh

print(f"det(composed) = {det_composed:.6f}")
print(f"det(R) * det(S) * det(Sh) = {product:.6f}")
print(f"Eşit mi? {abs(det_composed - product) < 1e-9}")
