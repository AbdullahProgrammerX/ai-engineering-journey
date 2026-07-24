import math

def softmax_naive(logits):
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def softmax_stable(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

# float64'ü de aşacak kadar büyük logit'ler
truly_dangerous = [1000.0, 1001.0, 1002.0]

print("--- Gerçekten patlatan logit'ler (1000+) ---")
print(f"Stable: {softmax_stable(truly_dangerous)}")

try:
    result = softmax_naive(truly_dangerous)
    print(f"Naive:  {result}")
except OverflowError as e:
    print(f"Naive:  HATA! {e}")
