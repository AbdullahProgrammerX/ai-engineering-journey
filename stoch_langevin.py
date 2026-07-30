import numpy as np

def langevin_dynamics(grad_U, x0, dt, temperature, n_steps, seed=None):
    rng = np.random.RandomState(seed)
    x = np.array(x0, dtype=float)
    trajectory = [x.copy()]
    for _ in range(n_steps):
        noise = rng.randn(*x.shape)
        x = x - dt * grad_U(x) + np.sqrt(2 * temperature * dt) * noise
        trajectory.append(x.copy())
    return np.array(trajectory)

# Çift-çukurlu (double-well) enerji fonksiyonu: U(x) = (x^2 - 1)^2
def U(x):
    return (x**2 - 1)**2

def grad_U(x):
    return 4 * x * (x**2 - 1)

print("--- Düşük sıcaklık (T=0.05): tek bir çukurda kalıyor ---")
traj_low = langevin_dynamics(grad_U, x0=[0.5], dt=0.01, temperature=0.05, n_steps=5000, seed=42)
print(f"Son 10 pozisyon: {np.round(traj_low[-10:].flatten(), 3)}")
print(f"Pozitif çukurda (x>0) kalma oranı: {(traj_low > 0).mean():.2%}")

print("\n--- Yüksek sıcaklık (T=0.5): iki çukur arasında geçiş yapıyor ---")
traj_high = langevin_dynamics(grad_U, x0=[0.5], dt=0.01, temperature=0.5, n_steps=5000, seed=42)
print(f"Son 10 pozisyon: {np.round(traj_high[-10:].flatten(), 3)}")
print(f"Pozitif çukurda (x>0) kalma oranı: {(traj_high > 0).mean():.2%}")
