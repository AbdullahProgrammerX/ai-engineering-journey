import random

random.seed(42)

def reparam_sample(mu, sigma):
    epsilon = random.gauss(0, 1)
    return mu + sigma * epsilon

def reparam_gradient(mu, sigma, epsilon):
    dz_dmu = 1.0
    dz_dsigma = epsilon
    return dz_dmu, dz_dsigma

mu, sigma = 2.0, 0.5
epsilon = random.gauss(0, 1)

z = reparam_sample(mu, sigma)
dz_dmu, dz_dsigma = reparam_gradient(mu, sigma, epsilon)

print(f"mu={mu}, sigma={sigma}, epsilon={epsilon:.4f}")
print(f"z = mu + sigma*epsilon = {z:.4f}")
print(f"\ndz/dmu = {dz_dmu}  (gradyan mu'ya akabiliyor)")
print(f"dz/dsigma = {dz_dsigma:.4f}  (gradyan sigma'ya akabiliyor)")

print("""
Neden önemli: Normalde 'z'yi doğrudan bir dağılımdan örneklersen
(z ~ Normal(mu, sigma)), bu işlem TÜREV ALINAMAZ -- rastgele örnekleme
bir "kesinti" yaratır, gradyan mu ve sigma'ya geri akamaz.

Reparameterization trick, rastgeleliği DIŞARI çıkarıyor:
z = mu + sigma * epsilon  (epsilon sabit bir rastgele sayı, N(0,1)'den)

Şimdi z, mu ve sigma'nın DETERMINISTIK bir fonksiyonu (epsilon zaten
çekilmiş, sabit). Bu yüzden backpropagation, mu ve sigma'ya kadar
gradyanı akıtabiliyor -- bu da VAE (Variational Autoencoder) gibi
modellerin GRADIENT DESCENT ile eğitilebilmesini sağlıyor.
""")
