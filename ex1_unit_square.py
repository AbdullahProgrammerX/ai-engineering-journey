import math

def rotation_2d(theta):
    c, s = math.cos(theta), math.sin(theta)
    return [[c, -s], [s, c]]

def scaling_2d(sx, sy):
    return [[sx, 0], [0, sy]]

def shearing_2d(kx, ky):
    return [[1, kx], [ky, 1]]

def mat_vec_mul(matrix, vector):
    return [
        sum(matrix[i][j] * vector[j] for j in range(len(vector)))
        for i in range(len(matrix))
    ]

def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

corners = [[0,0], [1,0], [1,1], [0,1]]

def apply_transform(matrix, points):
    return [mat_vec_mul(matrix, p) for p in points]

print("Orijinal köşeler:", corners)

print("\n--- Rotasyon (45 derece) ---")
R = rotation_2d(math.pi / 4)
rotated = apply_transform(R, corners)
print("Dönmüş köşeler:", [[round(x,4) for x in p] for p in rotated])

print("Mesafe korunuyor mu kontrolü:")
for i in range(4):
    j = (i + 1) % 4
    orig_dist = distance(corners[i], corners[j])
    rot_dist = distance(rotated[i], rotated[j])
    print(f"  köşe{i}-köşe{j}: orijinal={orig_dist:.4f}, döndürülmüş={rot_dist:.4f}, eşit mi: {abs(orig_dist-rot_dist) < 1e-9}")

print("\n--- Ölçekleme (2x, 3y) ---")
S = scaling_2d(2, 3)
scaled = apply_transform(S, corners)
print("Ölçeklenmiş köşeler:", scaled)

print("\n--- Shear (kx=1) ---")
Sh = shearing_2d(1, 0)
sheared = apply_transform(Sh, corners)
print("Shear uygulanmış köşeler:", sheared)
