import math
import random

random.seed(42)

def gumbel_sample():
    u = random.random()
    return -math.log(-math.log(u))

def softmax(logits):
    max_l = max(logits)
    exps = [math.exp(z - max_l) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def gumbel_softmax(probs, temperature):
    gumbels = [math.log(p) + gumbel_sample() for p in probs]
    return softmax([g / temperature for g in gumbels])

probs = [0.5, 0.3, 0.15, 0.05]  # 4 kategorili bir olasılık dağılımı

print("Temperature azaldıkça çıktı 'one-hot' vektöre yaklaşıyor:\n")
for temp in [2.0, 1.0, 0.5, 0.1, 0.01]:
    result = gumbel_softmax(probs, temp)
    print(f"temperature={temp:<5.2f}: {[f'{r:.4f}' for r in result]}")
