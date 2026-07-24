import numpy as np

def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)

# Boyutlar: Batch, Head, Time(sequence length), Dim(per head)
B, H, T, D = 2, 4, 8, 16
E = H * D  # toplam embedding boyutu = 64

print(f"Batch={B}, Head={H}, SeqLen={T}, HeadDim={D}, TotalEmbed={E}\n")

X = np.random.randn(B, T, E)
print(f"Girdi X şekli: {X.shape}  (batch, sequence, embedding)")

W_q = np.random.randn(E, E) * 0.02
W_k = np.random.randn(E, E) * 0.02
W_v = np.random.randn(E, E) * 0.02
W_o = np.random.randn(E, E) * 0.02

# 1. Q, K, V projeksiyonu
Q = np.einsum("bte,ek->btk", X, W_q)
K = np.einsum("bte,ek->btk", X, W_k)
V = np.einsum("bte,ek->btk", X, W_v)
print(f"Q/K/V projeksiyon sonrası şekil: {Q.shape}")

# 2. Head'lere böl (multi-head attention)
Q = Q.reshape(B, T, H, D).transpose(0, 2, 1, 3)
K = K.reshape(B, T, H, D).transpose(0, 2, 1, 3)
V = V.reshape(B, T, H, D).transpose(0, 2, 1, 3)
print(f"Head'lere bölündükten sonra Q şekli: {Q.shape}  (batch, head, time, headdim)")

# 3. Attention skorları -- her token'ın her token'a "ne kadar baktığı"
scores = np.einsum("bhtd,bhsd->bhts", Q, K) / np.sqrt(D)
print(f"Attention skorları şekli: {scores.shape}  (her token çifti için bir skor)")

# 4. Softmax ile olasılığa çevir
weights = softmax(scores, axis=-1)

# 5. Ağırlıklı toplam -- her token, diğer tüm token'lardan "bilgi topluyor"
attn_output = np.einsum("bhts,bhsd->bhtd", weights, V)
print(f"Attention çıktısı şekli: {attn_output.shape}")

# 6. Head'leri birleştir
concat = attn_output.transpose(0, 2, 1, 3).reshape(B, T, E)
print(f"Head'ler birleştirildikten sonra: {concat.shape}")

# 7. Son projeksiyon
output = np.einsum("bte,ek->btk", concat, W_o)
print(f"Nihai çıktı şekli: {output.shape}  (girdiyle aynı şekilde!)")
