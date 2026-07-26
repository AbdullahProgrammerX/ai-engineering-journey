import math
import random

random.seed(42)

def softmax(logits):
    max_l = max(logits)
    exps = [math.exp(z - max_l) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def temperature_sample(logits, temperature):
    scaled = [z / temperature for z in logits]
    return softmax(scaled)

# Örnek: 5 kelimelik bir "sözlük" için model çıktısı (logit'ler)
tokens = ["kedi", "köpek", "araba", "gökyüzü", "mutlu"]
logits = [3.0, 2.5, 0.5, 0.2, 1.8]

print("Aynı logit'ler, farklı temperature değerleriyle:\n")
for temp in [0.1, 0.5, 1.0, 2.0, 5.0]:
    probs = temperature_sample(logits, temp)
    print(f"temperature={temp:.1f}:")
    for tok, p in zip(tokens, probs):
        bar = "#" * int(p * 50)
        print(f"  {tok:10s} {p:.4f}  {bar}")
    print()
