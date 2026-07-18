import numpy as np

class Vector:
    def __init__(self, data):
        self.data = np.array(data, dtype=float)

    def angle_between(self, other):
        cos_theta = np.dot(self.data, other.data) / (
            np.linalg.norm(self.data) * np.linalg.norm(other.data)
        )
        cos_theta = np.clip(cos_theta, -1.0, 1.0)  # sayısal hatalara karşı güvenlik
        return np.degrees(np.arccos(cos_theta))

v1 = Vector([1, 0])
v2 = Vector([0, 1])
print(f"[1,0] ve [0,1] arasındaki açı: {v1.angle_between(v2):.2f} derece")

v3 = Vector([1, 1])
v4 = Vector([1, 0])
print(f"[1,1] ve [1,0] arasındaki açı: {v3.angle_between(v4):.2f} derece")
