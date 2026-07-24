import math

def softmax_naive(logits):
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def cross_entropy_naive(true_class, logits):
    probs = softmax_naive(logits)
    return -math.log(probs[true_class])

def cross_entropy_stable(true_class, logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = math.log(sum(math.exp(s) for s in shifted))
    log_prob = shifted[true_class] - log_sum_exp
    return -log_prob

logits = [2.0, 5.0, 1.0]
true_class = 1
print(f"Naive:  {cross_entropy_naive(true_class, logits):.6f}")
print(f"Stable: {cross_entropy_stable(true_class, logits):.6f}")
print("(Bu değerlerde ikisi de aynı çıkar, ama logit'ler büyüdükçe naive versiyon patlar.)")

print("\n--- Tehlikeli logit'lerle ---")
dangerous_logits = [200.0, 500.0, 100.0]
print(f"Stable: {cross_entropy_stable(1, dangerous_logits):.6f}")
try:
    print(f"Naive:  {cross_entropy_naive(1, dangerous_logits):.6f}")
except OverflowError as e:
    print(f"Naive:  HATA! {e}")
