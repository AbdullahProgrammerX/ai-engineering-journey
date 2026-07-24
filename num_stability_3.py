import math

def clip_by_norm(gradients, max_norm):
    total_norm = math.sqrt(sum(g**2 for g in gradients))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        return [g * scale for g in gradients]
    return gradients

print("=== Gradient Clipping ===")
grads = [10.0, 20.0, 30.0]
clipped = clip_by_norm(grads, max_norm=5.0)
print(f"Orijinal norm: {math.sqrt(sum(g**2 for g in grads)):.2f}")
print(f"Clip sonrası norm: {math.sqrt(sum(g**2 for g in clipped)):.2f}")
print(f"Yön korundu mu: {[round(c/clipped[0],4) for c in clipped]} == {[round(g/grads[0],4) for g in grads]}")

def check_tensor(name, values):
    has_nan = any(math.isnan(v) for v in values)
    has_inf = any(math.isinf(v) for v in values)
    if has_nan or has_inf:
        print(f"UYARI {name}: nan={has_nan} inf={has_inf}")
        return False
    print(f"{name}: temiz")
    return True

print("\n=== NaN/Inf Tespiti ===")
check_tensor("iyi_tensor", [1.0, 2.0, 3.0])
check_tensor("kotu_tensor", [1.0, float('nan'), 3.0])
check_tensor("cirkin_tensor", [1.0, float('inf'), 3.0])
