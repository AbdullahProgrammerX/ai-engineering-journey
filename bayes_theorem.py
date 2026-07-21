def bayes(prior, likelihood, false_positive_rate):
    evidence = likelihood * prior + false_positive_rate * (1 - prior)
    posterior = likelihood * prior / evidence
    return posterior

# Örnek: 10.000 kişiden 1'inde görülen bir hastalık, test %99 doğru
result = bayes(prior=0.0001, likelihood=0.99, false_positive_rate=0.01)
print(f"P(hasta|test pozitif) = {result:.4f}")
