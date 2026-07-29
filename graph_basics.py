import numpy as np
from collections import deque

class Graph:
    def __init__(self, n_nodes, directed=False):
        self.n = n_nodes
        self.directed = directed
        self.adj = {i: {} for i in range(n_nodes)}

    def add_edge(self, u, v, weight=1.0):
        self.adj[u][v] = weight
        if not self.directed:
            self.adj[v][u] = weight

    def neighbors(self, node):
        return list(self.adj[node].keys())

    def degree(self, node):
        return len(self.adj[node])

    def adjacency_matrix(self):
        A = np.zeros((self.n, self.n))
        for u in range(self.n):
            for v, w in self.adj[u].items():
                A[u][v] = w
        return A

    def degree_matrix(self):
        D = np.zeros((self.n, self.n))
        for i in range(self.n):
            D[i][i] = self.degree(i)
        return D

    def laplacian(self):
        return self.degree_matrix() - self.adjacency_matrix()


def bfs(graph, start):
    visited = set()
    order = []
    distances = {}
    queue = deque([(start, 0)])
    visited.add(start)
    while queue:
        node, dist = queue.popleft()
        order.append(node)
        distances[node] = dist
        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    return order, distances


def dfs(graph, start):
    visited = set()
    order = []
    stack = [start]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        order.append(node)
        for neighbor in reversed(graph.neighbors(node)):
            if neighbor not in visited:
                stack.append(neighbor)
    return order


# Basit bir graf kur: 0-1-2 üçgeni + 2-3-4 zinciri
g = Graph(5)
g.add_edge(0, 1)
g.add_edge(1, 2)
g.add_edge(0, 2)
g.add_edge(2, 3)
g.add_edge(3, 4)

print("Komşuluk matrisi:")
print(g.adjacency_matrix())

print("\nDerece matrisi:")
print(g.degree_matrix())

order_bfs, distances = bfs(g, 0)
print(f"\nBFS sırası (0'dan başlayarak): {order_bfs}")
print(f"Her düğüme mesafe (hop sayısı): {distances}")

order_dfs = dfs(g, 0)
print(f"\nDFS sırası (0'dan başlayarak): {order_dfs}")
