import math
import random

random.seed(42)

def softmax(logits):
    max_l = max(logits)
    exps = [math.exp(z - max_l) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def sample_from_probs(probs):
    r = random.random()
    cumsum = 0
    for i, p in enumerate(probs):
        cumsum += p
        if r < cumsum:
            return i
    return len(probs) - 1

def top_k_sample(logits, k):
    indexed = sorted(enumerate(logits), key=lambda x: -x[1])
    top = indexed[:k]
    top_logits = [l for _, l in top]
    probs = softmax(top_logits)
    idx = sample_from_probs(probs)
    return top[idx][0], probs

def top_p_sample(logits, p):
    probs = softmax(logits)
    indexed = sorted(enumerate(probs), key=lambda x: -x[1])
    cumsum = 0
    selected = []
    for token_idx, prob in indexed:
        cumsum += prob
        selected.append((token_idx, prob))
        if cumsum >= p:
            break
    sel_probs = [pr for _, pr in selected]
    total = sum(sel_probs)
    sel_probs = [pr / total for pr in sel_probs]
    idx = sample_from_probs(sel_probs)
    return selected[idx][0], sel_probs, [t for t, _ in selected]

tokens = ["kedi", "köpek", "araba", "gökyüzü", "mutlu", "üzgün", "mavi", "hızlı"]
logits = [3.0, 2.5, 0.5, 0.2, 1.8, 0.1, -0.5, 1.0]

print("--- Top-k sampling (k=3) ---")
print("Sadece en yüksek 3 logit'e sahip kelime aday olabilir:")
idx, probs = top_k_sample(logits, k=3)
sorted_idx = sorted(range(len(logits)), key=lambda i: -logits[i])[:3]
for i, p in zip(sorted_idx, probs):
    print(f"  {tokens[i]:10s} olasılık={p:.4f}")
print(f"Seçilen kelime: {tokens[idx]}")

print("\n--- Top-p (nucleus) sampling (p=0.8) ---")
print("Kümülatif olasılığı 0.8'e ulaşana kadar kelime eklenir:")
idx, probs, selected_tokens = top_p_sample(logits, p=0.8)
for t, pr in zip(selected_tokens, probs):
    print(f"  {tokens[t]:10s} olasılık={pr:.4f}")
print(f"Seçilen kelime: {tokens[idx]}")
print(f"Toplam aday kelime sayısı: {len(selected_tokens)} / {len(tokens)}")
