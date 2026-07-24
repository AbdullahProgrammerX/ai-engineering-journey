import math

print("=== Floating Point Precision ===")
print(f"0.1 + 0.2 = {0.1 + 0.2}")
print(f"0.1 + 0.2 == 0.3? {0.1 + 0.2 == 0.3}")
print(f"Fark: {(0.1 + 0.2) - 0.3:.2e}")

def softmax_naive(logits):
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def softmax_stable(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

print("\n=== Softmax karşılaştırması ===")
safe_logits = [2.0, 1.0, 0.1]
print(f"Güvenli logit'lerde -- Naive:  {softmax_naive(safe_logits)}")
print(f"Güvenli logit'lerde -- Stable: {softmax_stable(safe_logits)}")

dangerous_logits = [100.0, 101.0, 102.0]
print(f"\nTehlikeli logit'lerde -- Stable: {softmax_stable(dangerous_logits)}")
try:
    result = softmax_naive(dangerous_logits)
    print(f"Tehlikeli logit'lerde -- Naive: {result}")
except OverflowError as e:
    print(f"Tehlikeli logit'lerde -- Naive: HATA! {e}")

def logsumexp_naive(values):
    return math.log(sum(math.exp(v) for v in values))

def logsumexp_stable(values):
    c = max(values)
    return c + math.log(sum(math.exp(v - c) for v in values))

print("\n=== Log-Sum-Exp karşılaştırması ===")
safe = [1.0, 2.0, 3.0]
print(f"Güvenli değerler -- Naive:  {logsumexp_naive(safe):.6f}")
print(f"Güvenli değerler -- Stable: {logsumexp_stable(safe):.6f}")

large = [500.0, 501.0, 502.0]
print(f"\nBüyük değerler -- Stable: {logsumexp_stable(large):.6f}")
try:
    print(f"Büyük değerler -- Naive: {logsumexp_naive(large):.6f}")
except OverflowError as e:
    print(f"Büyük değerler -- Naive: HATA! {e}")
