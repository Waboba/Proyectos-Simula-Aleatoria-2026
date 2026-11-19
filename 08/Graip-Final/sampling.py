import networkx as nx
import numpy as np
from random import random
import hashlib

_CACHE = {}

def Sample(G, N, n_gl, count_func):
    """
    G : NetworkX graph, of which each edge has an attribute 'probability' (float) between 0 and 1.
    N : int
        Number of samples.
    n_gl : int, {2,8,29} optional
        Cantida de tipos de graphlets a contar
    
    Returns
    -------
    E_nodes : float
        Mean number of nodes.
    std_nodes : float
        Standard deviation on number of nodes.
    E_edges : float
        Mean number of edges.
    std_edges : float
        Standard deviation on number of edges.
    E_graphlets : array
        Mean graphlet frequencies.
    std_graphlets : array
        Standard deviation on graphlet frequencies.
    E_degrees : array
        Mean degree histogram.
    std_degrees : array
        Standard deviation on the degree histogram.
    """

    items = sorted((min(u, v), max(u, v), round(d["probability"], 6))
                   for u, v, d in G.edges(data=True))
    key = hashlib.sha1(repr((items, N, n_gl)).encode()).hexdigest()
    if key in _CACHE:
        return _CACHE[key]

    
    print("Iniciando sampleo.")
            
    E_nodes = 0
    E_nodes_sq = 0      # squared, store for std
    E_edges = 0
    E_edges_sq = 0
    E_graphlets = np.zeros(n_gl)
    E_graphlets_sq = np.zeros(n_gl)
    E_degrees = np.zeros(len(nx.degree_histogram(G)))
    E_degrees_sq = np.zeros(len(E_degrees))
    
    p = 1
    for i in range(N):
        
        # Construct sample graph
        
        Gi = nx.create_empty_copy(G)
        for edge in G.edges():
            r = random()
            if r <= G.edges[edge]["probability"]:
                Gi.add_edge(edge[0], edge[1])
        
        Gi = Gi.subgraph(max(nx.connected_components(Gi), key=len))
        
        # Compute properties
        
        E_nodes += Gi.number_of_nodes()
        E_nodes_sq += Gi.number_of_nodes()**2
        
        E_edges += Gi.number_of_edges()
        E_edges_sq += Gi.number_of_edges()**2
        
        counts = count_func(Gi).astype('float64')
        E_graphlets += counts
        E_graphlets_sq += counts**2
        
        deg_hist = np.array(nx.degree_histogram(Gi))
        deg_hist.resize(len(E_degrees))
        E_degrees += deg_hist
        E_degrees_sq += deg_hist**2
    
    E_nodes /= N
    std_nodes = np.sqrt(E_nodes_sq/N - E_nodes**2)
    
    E_edges /= N
    std_edges = np.sqrt(E_edges_sq/N - E_edges**2)
    
    E_graphlets /= N
    std_graphlets = np.sqrt(E_graphlets_sq/N - E_graphlets**2)
    
    E_degrees /= N
    std_degrees = np.sqrt(E_degrees_sq/N - E_degrees**2)
    
    print("Sampleo terminado.")
            

    resultado = (E_nodes, std_nodes, E_edges, std_edges,
                 E_graphlets, std_graphlets, E_degrees, std_degrees)
    _CACHE[key] = resultado
    return resultado