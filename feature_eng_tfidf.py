import math

def count_vectorize(documents):
    vocab = {}
    idx = 0
    for doc in documents:
        for word in doc.lower().split():
            if word not in vocab:
                vocab[word] = idx
                idx += 1
    vectors = []
    for doc in documents:
        vec = [0] * len(vocab)
        for word in doc.lower().split():
            vec[vocab[word]] += 1
        vectors.append(vec)
    return vectors, vocab

def tfidf(documents):
    n_docs = len(documents)
    vocab = {}
    idx = 0
    for doc in documents:
        for word in doc.lower().split():
            if word not in vocab:
                vocab[word] = idx
                idx += 1

    doc_freq = {}
    for doc in documents:
        seen = set()
        for word in doc.lower().split():
            if word not in seen:
                doc_freq[word] = doc_freq.get(word, 0) + 1
                seen.add(word)

    vectors = []
    for doc in documents:
        words = doc.lower().split()
        word_count = len(words)
        tf_map = {}
        for word in words:
            tf_map[word] = tf_map.get(word, 0) + 1
        vec = [0.0] * len(vocab)
        for word, count in tf_map.items():
            tf = count / word_count
            idf = math.log(n_docs / doc_freq[word])
            vec[vocab[word]] = tf * idf
        vectors.append(vec)
    return vectors, vocab


descriptions = [
    "large modern house with pool",
    "small cozy cottage near downtown",
    "spacious family home with large yard",
    "modern apartment downtown with view",
    "rustic cabin in rural area",
]

print("--- Count Vectorize (basit kelime sayımı) ---")
cv, cv_vocab = count_vectorize(descriptions)
print(f"Kelime hazinesi boyutu: {len(cv_vocab)}")
print(f"Doküman 0 vektörü (ilk 10 boyut): {cv[0][:10]}")

print("\n--- TF-IDF (akıllı ağırlıklandırma) ---")
tf, tf_vocab = tfidf(descriptions)
print(f"Doküman 0: '{descriptions[0]}'")
top_words = sorted(tf_vocab.keys(), key=lambda w: tf[0][tf_vocab[w]], reverse=True)[:3]
print(f"En yüksek TF-IDF skorlu 3 kelime: {top_words}")

print("\nNeden 'with' gibi kelimeler düşük skor alıyor:")
if "with" in tf_vocab:
    idx = tf_vocab["with"]
    print(f"  'with' TF-IDF skoru (doküman 0): {tf[0][idx]:.4f}")
    print(f"  'with' kaç dokümanda geçiyor: {sum(1 for doc in descriptions if 'with' in doc.lower())}/{len(descriptions)}")
print("  Çok dokümanda geçen kelimeler (with, house gibi) DÜŞÜK IDF alır --")
print("  çünkü ayırt edici değiller, her yerde var. Nadir kelimeler YÜKSEK IDF alır.")
