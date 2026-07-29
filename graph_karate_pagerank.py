import networkx as nx
import numpy as np

G = nx.karate_club_graph()

print(f"Düğüm sayısı: {G.number_of_nodes()}")
print(f"Kenar sayısı: {G.number_of_edges()}")

A = nx.adjacency_matrix(G).toarray()
L = nx.laplacian_matrix(G).toarray()

eigenvalues = np.linalg.eigvalsh(L.astype(float))
print(f"\nEn küçük 5 özdeğer: {np.round(eigenvalues[:5], 4)}")
print(f"Bağlı bileşen sayısı: {nx.number_connected_components(G)}")

print("\n--- Topluluk tespiti (community detection) ---")
communities = nx.community.greedy_modularity_communities(G)
print(f"Bulunan topluluk sayısı: {len(communities)}")
for i, comm in enumerate(communities):
    print(f"  Topluluk {i}: {sorted(comm)}")

print("\n--- PageRank -- en 'önemli' düğümler ---")
pr = nx.pagerank(G)
top_nodes = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:5]
print("En yüksek PageRank skoruna sahip 5 düğüm:")
for node, score in top_nodes:
    print(f"  Düğüm {node}: {score:.4f}")

print("\n(Karate Club, gerçek bir sosyal ağdan geliyor -- 1970'lerde bir karate")
print("kulübünde yaşanan bölünmeyi modelliyor. Ünlü bir graf teorisi test verisi.)")
