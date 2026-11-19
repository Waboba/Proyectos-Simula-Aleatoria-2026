import random
import numpy as np
import networkx as nx
from graphlet_counts import update_counts_node, update_counts_edge

def arista_nueva(H):
    nodos = list(H.nodes)
    for _ in range(1000):
        a, b = random.sample(nodos, 2)   # dos nodos distintos (sin self-loops)
        if not H.has_edge(a, b):         # solo la aceptamos si NO existe
            return a, b
    return None

def add_node(H, u, CCg):                      # AddNode, líneas 40-50
    nodos = list(H.nodes)
    pesos = np.array([H.degree(x) for x in nodos], dtype=float)
    v = random.choices(nodos, weights=pesos if pesos.sum() > 0 else None)[0]
    clique = max(nx.find_cliques(H, nodes=[v]), key=len)
    if len(clique) >= 4:
        vecinos = clique
    else:
        kv = H.degree(v)
        env = nx.triangles(H, v)
        pnb = ((kv + 1) * CCg / 2 - env) / kv if kv > 0 else 0.0
        pnb = min(max(pnb, 0.0), 1.0)
        vecinos = [v] + [x for x in H.neighbors(v) if random.random() < pnb]
    return [(u, x) for x in vecinos]

def grados_tras(H, op):                       # grados de T sin construir T
    g = dict(H.degree())
    t, u, ar = op["tipo"], op["u"], op["aristas"]
    if t == "add_node":
        g[u] = len(ar)
        for _, x in ar: g[x] += 1
    elif t == "remove_node":
        for _, x in ar: g[x] -= 1
        del g[u]
    elif t == "add_edge":
        a, b = ar[0]; g[a] += 1; g[b] += 1
    else:                                     # remove_edge
        a, b = ar[0]; g[a] -= 1; g[b] -= 1
    return list(g.values())

def aplicar(H, op):
    t = op["tipo"]
    if t == "add_node":
        H.add_node(op["u"]); H.add_edges_from(op["aristas"])
    elif t == "remove_node":
        H.remove_node(op["u"])
    elif t == "add_edge":
        H.add_edges_from(op["aristas"])
    else:
        H.remove_edges_from(op["aristas"])

def dist_grados(grados):
    return np.bincount(grados) / len(grados)

def histograma_bins(P, bins):
    out = np.zeros(len(bins) - 1)
    for i in range(len(bins) - 1):
        out[i] = P[bins[i]:bins[i + 1]].sum()
    out[-1] += P[bins[-1]:].sum()          # grados mayores al último borde
    return out

def dentro(x, E, b):
    return bool(np.all(np.abs(x - E) <= b))

def construir_valid():
    """Códigos de bits válidos para M1..M8, en el orden de four_counts."""
    graphlets = ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8"]
    valid = {g: set() for g in graphlets}
    for k in (3, 4):
        pares = [(i, j) for j in range(k) for i in range(j)]   # bit j(j-1)/2 + i
        for code in range(1 << len(pares)):
            F = nx.Graph()
            F.add_nodes_from(range(k))
            F.add_edges_from(p for b, p in enumerate(pares) if (code >> b) & 1)
            if not nx.is_connected(F):
                continue
            e = F.number_of_edges()
            dmax = max(d for _, d in F.degree())
            if k == 3:   nombre = "M1" if e == 2 else "M2"
            elif e == 3: nombre = "M3" if dmax == 3 else "M4"
            elif e == 4: nombre = "M6" if dmax == 2 else "M5"
            elif e == 5: nombre = "M7"
            else:        nombre = "M8"
            valid[nombre].add(code)
    return graphlets, valid

def contar_T(H, C_H, op, graphlets, valid):
    """C_T a partir de C_H, tocando solo el vecindario local. H no cambia al terminar."""
    t, u, ar = op["tipo"], op["u"], op["aristas"]
    if t in ("add_edge", "remove_edge"):
        # H es el grafo ANTES del cambio; update_counts_edge invierte el bit de la arista
        return C_H + update_counts_edge(H, ar[0], graphlets, valid)
    if t == "remove_node":
        # H todavía contiene u: se restan los graphlets que lo incluyen
        return C_H - update_counts_node(H, u, graphlets, valid)
    # add_node: update_counts_node necesita el nodo dentro, se añade, se mide y se deshace
    H.add_node(u)
    H.add_edges_from(ar)
    try:
        d = update_counts_node(H, u, graphlets, valid)
    finally:
        H.remove_node(u)
    return C_H + d




