import numpy as np

def hessian_eigenvalues(H):
    return np.linalg.eigvalsh(H)

print("--- f(x,y) = x^2 + 3xy + y^2 (dersteki örnek) ---")
H1 = np.array([[2, 3], [3, 2]])
eigs1 = hessian_eigenvalues(H1)
print(f"Hessian:\n{H1}")
print(f"Özdeğerler: {eigs1}")
print(f"Konveks mi? {'Evet (tüm özdeğerler >= 0)' if all(eigs1 >= 0) else 'Hayır (negatif özdeğer var -- eyer noktası veya konkav)'}")

print("\n--- f(x,y) = x^2 + y^2 (bowl/çukur şekli) ---")
H2 = np.array([[2, 0], [0, 2]])
eigs2 = hessian_eigenvalues(H2)
print(f"Hessian:\n{H2}")
print(f"Özdeğerler: {eigs2}")
print(f"Konveks mi? {'Evet' if all(eigs2 >= 0) else 'Hayır'}")

print("\n--- f(x,y) = x^2 - y^2 (eyer noktası -- saddle) ---")
H3 = np.array([[2, 0], [0, -2]])
eigs3 = hessian_eigenvalues(H3)
print(f"Hessian:\n{H3}")
print(f"Özdeğerler: {eigs3}")
print(f"Konveks mi? {'Evet' if all(eigs3 >= 0) else 'Hayır (karışık işaretli özdeğerler -- eyer noktası)'}")
