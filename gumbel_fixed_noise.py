import math
import random

def gumbel_sample():
    u = random.random()
    return -math.log(-math.log(u))

def softmax(logits):
    max_l = max(logits)
    exps = [math.exp(z - max_l) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def gumbel_softmax_fixed(probs, temperature, gumbels):
    g = [math.log(p) + noise for p, noise in zip(probs, gumbels)]
    return softmax([x / temperature for x in g])

random.seed(42)
probs = [0.5, 0.3, 0.15, 0.05]

# Gumbel gürültüsünü BİR KERE çek, tüm temperature'larda aynısını kullan
fixed_gumbels = [gumbel_sample() for _ in probs]
print(f"Sabit Gumbel gürültüsü: {[f'{g:.4f}' for g in fixed_gumbels]}\n")

for temp in [2.0, 1.0, 0.5, 0.1, 0.01]:
    result = gumbel_softmax_fixed(probs, temp, fixed_gumbels)
    print(f"temperature={temp:<5.2f}: {[f'{r:.4f}' for r in result]}")
