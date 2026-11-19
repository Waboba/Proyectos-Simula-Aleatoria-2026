"""
wwl_validation.py
=================
Validación de generadores de grafos (p. ej. GRAIP) usando el kernel
Wasserstein-Weisfeiler-Lehman (WWL, Togninalli et al., NeurIPS 2019) dentro de
un MMD^2 (como en la Sec. 4.2.1 del paper de GRAIP, pero con WWL como kernel).

Contenido
---------
1. Utilidades de grafos        : prep_graph, muestreo de mundos posibles
2. Embeddings WL continuos     : Ec. (5) y Def. 4 del paper WWL
3. Wasserstein-1               : POT si está instalado; si no, LP exacto (scipy) o Sinkhorn
4. Bloques de distancias       : paralelos (joblib) y con caché en disco
5. Kernel y estadísticos       : Laplaciano, PSD, MMD^2 insesgado, permutación,
                                 razón de diversidad, test de clasificación (C2ST)
6. Baselines                   : modelo de configuración y BA dual
7. Test de sensibilidad        : intercambios de aristas que preservan el grado
8. Orquestadores               : run_wwl_validation / run_sensitivity / benchmark_pair_time
"""
from __future__ import annotations

import hashlib
import time
from functools import lru_cache
from pathlib import Path

import networkx as nx
import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed
from scipy.optimize import linprog
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
from scipy.stats import wasserstein_distance

try:  # POT (Python Optimal Transport): pip install pot
    import ot  # type: ignore

    HAS_POT = True
except Exception:  # pragma: no cover
    ot = None
    HAS_POT = False


# =============================================================================
# 1. Utilidades de grafos
# =============================================================================
def prep_graph(G: nx.Graph) -> nx.Graph:
    """Grafo simple, sin self-loops, componente conexa más grande, nodos 0..n-1.

    Se aplica igual a los grafos muestreados y a los generados, para que la
    comparación sea justa (GRAIP también trabaja con la componente más grande
    de cada muestra).
    """
    G = nx.Graph(G)
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    if G.number_of_nodes() == 0:
        return G
    comp = max(nx.connected_components(G), key=len)
    return nx.convert_node_labels_to_integers(G.subgraph(comp).copy())


def sample_possible_graphs(Gu: nx.Graph, n: int, seed: int, p_attr: str = "p") -> list[nx.Graph]:
    """Muestrea `n` mundos posibles de un grafo incierto (aristas independientes)."""
    rng = np.random.default_rng(seed)
    edges = list(Gu.edges())
    probs = np.array([Gu[u][v][p_attr] for u, v in edges], dtype=float)
    out = []
    for _ in range(n):
        keep = rng.random(len(edges)) < probs
        G = nx.Graph()
        G.add_nodes_from(Gu.nodes())
        G.add_edges_from(e for e, k in zip(edges, keep) if k)
        out.append(prep_graph(G))
    return out


# =============================================================================
# 2. Embeddings WL continuos (paper WWL, Ec. 5 y Def. 4)
# =============================================================================
def fit_standardizer(graphs: list[nx.Graph]) -> tuple[float, float]:
    """Media y desv. estándar del grado sobre el conjunto de referencia."""
    deg = np.concatenate([np.array([d for _, d in G.degree()], dtype=float) for G in graphs])
    sigma = deg.std()
    return float(deg.mean()), float(sigma if sigma > 0 else 1.0)


def wl_embedding(G: nx.Graph, h_max: int, mu: float, sigma: float) -> np.ndarray:
    """Embedding de nodos (n x (h_max+1)): [a^0, a^1, ..., a^h_max].

    a^0(v)   = (grado(v) - mu) / sigma
    a^{h+1}  = 1/2 * ( a^h(v) + (1/deg(v)) * sum_{u in N(v)} a^h(u) )
    """
    nodes = list(G.nodes())
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, weight=None, format="csr", dtype=float)
    deg = np.asarray(A.sum(axis=1)).ravel()
    safe = np.where(deg > 0, deg, 1.0)[:, None]
    a = ((deg - mu) / sigma)[:, None]
    feats = [a]
    for _ in range(h_max):
        a_new = 0.5 * (a + (A @ a) / safe)
        a = np.where(deg[:, None] > 0, a_new, a)
        feats.append(a)
    return np.hstack(feats)


def embed_graphs(graphs: list[nx.Graph], h_max: int, mu: float, sigma: float) -> list[np.ndarray]:
    return [wl_embedding(G, h_max, mu, sigma) for G in graphs]


# =============================================================================
# 3. Wasserstein-1 entre dos conjuntos de nodos (masa uniforme, dist. euclidiana)
# =============================================================================
@lru_cache(maxsize=64)
def _transport_constraints(n: int, m: int):
    rows = sp.kron(sp.identity(n), np.ones((1, m)), format="csr")  # sum_j P_ij
    cols = sp.kron(np.ones((1, n)), sp.identity(m), format="csr")  # sum_i P_ij
    return sp.vstack([rows, cols[:-1]], format="csr")  # una restricción es redundante


def _emd_linprog(M: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Costo exacto de transporte óptimo vía LP (HiGHS). Más lento que POT."""
    n, m = M.shape
    A_eq = _transport_constraints(n, m)
    b_eq = np.concatenate([a, b[:-1]])
    res = linprog(M.ravel(), A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    if res.status != 0:
        raise RuntimeError(f"linprog falló: {res.message}")
    return float(res.fun)


def _sinkhorn_cost(M, a, b, eps, n_iter=1000, tol=1e-8) -> float:
    """Sinkhorn en el dominio log. Devuelve <P, M> (sin el término de entropía)."""
    la, lb = np.log(a), np.log(b)
    f = np.zeros_like(a)
    g = np.zeros_like(b)
    for _ in range(n_iter):
        f = eps * (la - logsumexp((g[None, :] - M) / eps, axis=1))
        g = eps * (lb - logsumexp((f[:, None] - M) / eps, axis=0))
        P = np.exp((f[:, None] + g[None, :] - M) / eps)
        if np.abs(P.sum(axis=1) - a).sum() < tol:
            break
    P = np.exp((f[:, None] + g[None, :] - M) / eps)
    return float((P * M).sum())


def wasserstein1(Xa: np.ndarray, Xb: np.ndarray, solver: str = "emd", reg: float = 0.05) -> float:
    """W1 entre dos nubes de puntos con masa uniforme.

    solver = "emd"      : exacto (POT si existe; si no, LP de scipy).
    solver = "sinkhorn" : aproximado, `reg` es epsilon relativo al costo medio.
    Con una sola columna (h=0) se usa la forma cerrada 1D.
    """
    if Xa.shape[1] == 1:
        return float(wasserstein_distance(Xa[:, 0], Xb[:, 0]))
    n, m = len(Xa), len(Xb)
    M = cdist(Xa, Xb)
    a = np.full(n, 1.0 / n)
    b = np.full(m, 1.0 / m)
    if solver == "emd":
        return float(ot.emd2(a, b, M)) if HAS_POT else _emd_linprog(M, a, b)
    if solver == "sinkhorn":
        eps = reg * max(float(M.mean()), 1e-12)
        return _sinkhorn_cost(M, a, b, eps)
    raise ValueError(f"solver desconocido: {solver}")


# =============================================================================
# 4. Bloques de distancias (paralelo + caché)
# =============================================================================
def _pack(embs: list[np.ndarray]):
    sizes = np.array([len(e) for e in embs])
    offs = np.concatenate([[0], np.cumsum(sizes)])
    return np.vstack(embs), offs


def _fingerprint(pack) -> str:
    X, offs = pack
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(X).tobytes())
    h.update(offs.tobytes())
    return h.hexdigest()[:10]


def _chunk(idx_pairs, packA, packB, ncol, solver, reg):
    XA, oa = packA
    XB, ob = packB
    out = np.empty(len(idx_pairs))
    for k, (i, j) in enumerate(idx_pairs):
        out[k] = wasserstein1(XA[oa[i]: oa[i + 1], :ncol], XB[ob[j]: ob[j + 1], :ncol], solver, reg)
    return out


def distance_block(embA, embB, h, same=False, solver="emd", reg=0.05, n_jobs=1, cache_dir=None, tag=""):
    """Matriz D[i, j] = W1(emb_A[i], emb_B[j]) usando las columnas 0..h de los embeddings.

    same=True: A y B son el mismo conjunto -> solo se calcula el triángulo superior.
    Si cache_dir está dado, el resultado se guarda en disco con una huella del contenido.
    """
    ncol = h + 1
    pa = _pack(embA)
    pb = pa if same else _pack(embB)
    nA, nB = len(embA), len(embB)
    path = None
    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        path = Path(cache_dir) / f"D_{tag}_h{h}_{solver}_{_fingerprint(pa)}_{_fingerprint(pb)}.npy"
        if path.exists():
            D = np.load(path)
            if D.shape == (nA, nB):
                return D
    if same:
        ii, jj = np.triu_indices(nA, k=1)
    else:
        ii, jj = np.meshgrid(np.arange(nA), np.arange(nB), indexing="ij")
        ii, jj = ii.ravel(), jj.ravel()
    pairs = np.stack([ii, jj], axis=1)
    D = np.zeros((nA, nB))
    if len(pairs):
        n_chunks = max(1, min(len(pairs), n_jobs * 8))
        parts = [np.arange(k, len(pairs), n_chunks) for k in range(n_chunks)]  # intercalado: balancea carga
        res = Parallel(n_jobs=n_jobs)(
            delayed(_chunk)(pairs[p], pa, pb, ncol, solver, reg) for p in parts
        )
        for p, r in zip(parts, res):
            D[pairs[p, 0], pairs[p, 1]] = r
        if same:
            D = D + D.T
    if path is not None:
        np.save(path, D)
    return D


# =============================================================================
# 5. Kernel y estadísticos
# =============================================================================
def offdiag(D: np.ndarray) -> np.ndarray:
    return D[np.triu_indices_from(D, k=1)]


def median_lambda(D_pool: np.ndarray) -> float:
    """Heurística de la mediana: lambda = 1 / mediana(D) sobre pares distintos."""
    med = float(np.median(offdiag(D_pool)))
    return 1.0 / med if med > 0 else 1.0


def laplacian_kernel(D: np.ndarray, lam: float) -> np.ndarray:
    """K_WWL = exp(-lambda * D_W)  (Def. 5 del paper WWL)."""
    return np.exp(-lam * D)


def psd_clip(K: np.ndarray, tol: float = 1e-8) -> tuple[np.ndarray, float]:
    """Devuelve (K proyectada a PSD si hace falta, autovalor mínimo original)."""
    Ks = 0.5 * (K + K.T)
    w, V = np.linalg.eigh(Ks)
    if w.min() >= -tol:
        return Ks, float(w.min())
    return (V * np.clip(w, 0, None)) @ V.T, float(w.min())


def mmd2_unbiased(Kxx, Kyy, Kxy) -> float:
    """MMD^2 insesgado (Ec. 9 del paper GRAIP)."""
    m, n = len(Kxx), len(Kyy)
    sxx = (Kxx.sum() - np.trace(Kxx)) / (m * (m - 1))
    syy = (Kyy.sum() - np.trace(Kyy)) / (n * (n - 1))
    return float(sxx + syy - 2.0 * Kxy.mean())


def permutation_test(K: np.ndarray, m: int, n_perm: int = 1000, seed: int = 0) -> dict:
    """Test de permutación para MMD^2 sobre la matriz de kernel conjunta K.

    Las primeras `m` filas/columnas de K son el grupo X; el resto, el grupo Y.
    p = (1 + #{null >= obs}) / (1 + n_perm).
    """
    N = K.shape[0]
    n = N - m
    K0 = K - np.diag(np.diag(K))
    base = np.zeros(N)
    base[:m] = 1.0

    def stat(A):
        Bm = 1.0 - A
        AK = A @ K0
        sxx = (AK * A).sum(axis=1) / (m * (m - 1))
        syy = ((Bm @ K0) * Bm).sum(axis=1) / (n * (n - 1))
        sxy = (AK * Bm).sum(axis=1) / (m * n)
        return sxx + syy - 2.0 * sxy

    obs = float(stat(base[None, :])[0])
    rng = np.random.default_rng(seed)
    null = stat(rng.permuted(np.tile(base, (n_perm, 1)), axis=1))
    p = (1.0 + np.sum(null >= obs - 1e-12)) / (n_perm + 1.0)
    return dict(obs=obs, p=float(p), null_mean=float(null.mean()), null_std=float(null.std(ddof=1)))


def diversity_ratio(D_yy: np.ndarray, D_xx: np.ndarray) -> float:
    """rho = (distancia media intra-generados) / (distancia media intra-referencia).
    rho ~ 1: variabilidad similar a la de los muestreados. Análogo de la
    métrica de 'aleatoriedad' del paper GRAIP."""
    return float(offdiag(D_yy).mean() / offdiag(D_xx).mean())


def diversity_ratio_ci(D_yy, D_xx, n_boot: int = 500, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Intervalo bootstrap (percentil) de rho remuestreando grafos con reposición.
    Los pares formados por el mismo grafo repetido se excluyen (distancia 0 artificial)."""
    rng = np.random.default_rng(seed)

    def mean_pair(D, idx):
        mask = idx[:, None] != idx[None, :]
        return D[np.ix_(idx, idx)][mask].mean()

    ny, nx_ = len(D_yy), len(D_xx)
    out = np.empty(n_boot)
    for b in range(n_boot):
        out[b] = mean_pair(D_yy, rng.integers(0, ny, ny)) / mean_pair(D_xx, rng.integers(0, nx_, nx_))
    return float(np.nanquantile(out, alpha / 2)), float(np.nanquantile(out, 1 - alpha / 2))


def c2st(K: np.ndarray, y: np.ndarray, n_splits: int = 10, seed: int = 0) -> float:
    """Accuracy de un SVM con kernel precalculado (CV anidada para C) que intenta
    distinguir X de Y. ~0.5 => indistinguibles."""
    from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
    from sklearn.svm import SVC

    inner = GridSearchCV(SVC(kernel="precomputed"), {"C": [1e-2, 1e-1, 1.0, 10.0, 100.0]}, cv=3)
    outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return float(cross_val_score(inner, K, y, cv=outer).mean())


# =============================================================================
# 6. Baselines
# =============================================================================
def config_model_graph(deg_seq, seed: int) -> nx.Graph:
    deg = [int(d) for d in deg_seq]
    if sum(deg) % 2:
        deg[int(np.argmax(deg))] -= 1
    return prep_graph(nx.configuration_model(deg, seed=seed))


def make_config_graphs(Gu: nx.Graph, n_graphs: int, seed: int) -> list[nx.Graph]:
    """Control negativo: cada grafo usa la secuencia de grados de un mundo posible
    independiente (distribución de grados correcta, sin estructura de orden superior)."""
    out = []
    for i in range(n_graphs):
        donor = sample_possible_graphs(Gu, 1, seed + 500_000 + i)[0]
        out.append(config_model_graph([d for _, d in donor.degree()], seed + 900_000 + i))
    return out


def make_dual_ba_graphs(ref_graphs: list[nx.Graph], n_graphs: int, seed: int) -> list[nx.Graph]:
    """BA dual como en el paper: n ~ Normal(E[n], s_n) y misma razón media m/n."""
    rng = np.random.default_rng(seed)
    ns = np.array([G.number_of_nodes() for G in ref_graphs], dtype=float)
    ratio = float(np.mean([G.number_of_edges() / G.number_of_nodes() for G in ref_graphs]))
    m1 = max(1, int(np.floor(ratio)))
    m2 = m1 + 1
    p = float(np.clip(m1 + 1 - ratio, 0.0, 1.0))
    out = []
    for _ in range(n_graphs):
        n = max(m2 + 2, int(round(rng.normal(ns.mean(), ns.std()))))
        out.append(prep_graph(nx.dual_barabasi_albert_graph(n, m1, m2, p, seed=int(rng.integers(2**31 - 1)))))
    return out


# =============================================================================
# 7. Perturbación para el test de sensibilidad
# =============================================================================
def rewire_graph(G: nx.Graph, frac: float, seed: int) -> nx.Graph:
    """Aplica round(frac * m) intercambios de aristas que preservan el grado."""
    H = nx.Graph(G)
    nswap = int(round(frac * H.number_of_edges()))
    if nswap > 0 and H.number_of_nodes() >= 4:
        try:
            nx.double_edge_swap(H, nswap=nswap, max_tries=nswap * 100, seed=seed)
        except nx.NetworkXException:
            pass  # intercambios parciales si no se logran todos
    return prep_graph(H)


# =============================================================================
# 8. Orquestadores
# =============================================================================
def _analyze(D_rr, D_xy, D_yy, lam, n_perm, seed, do_c2st):
    m, n = D_rr.shape[0], D_yy.shape[0]
    Dpool = np.block([[D_rr, D_xy], [D_xy.T, D_yy]])
    K, min_eig = psd_clip(laplacian_kernel(Dpool, lam))
    perm = permutation_test(K, m, n_perm=n_perm, seed=seed)
    row = dict(
        mmd2=perm["obs"],
        p_perm=perm["p"],
        z=(perm["obs"] - perm["null_mean"]) / perm["null_std"] if perm["null_std"] > 0 else float("nan"),
        rho=diversity_ratio(D_yy, D_rr),
        min_eig=min_eig,
        lam=lam,
    )
    row["rho_lo"], row["rho_hi"] = diversity_ratio_ci(D_yy, D_rr, seed=seed)
    row["c2st_acc"] = c2st(K, np.r_[np.zeros(m), np.ones(n)], seed=seed) if do_c2st else float("nan")
    return row


def run_wwl_validation(
    X_ref, X_cal, models, h_list=(0, 1, 2, 3), baseline="config", solver="emd", reg=0.05,
    n_jobs=1, cache_dir=None, n_perm=1000, seed=0, do_c2st=True, verbose=True,
):
    """Evalúa cada conjunto de grafos generados contra la referencia X_ref con
    MMD^2 (kernel WWL), test de permutación, razón de diversidad y C2ST.

    X_ref, X_cal : dos conjuntos independientes de grafos muestreados de la red
                   probabilística. X_cal se usa como "generador ideal" (calibración).
    models       : dict nombre -> lista de grafos generados.
    Devuelve una lista de dicts (una fila por modelo y por h).
    """
    h_max = max(h_list)
    mu, sigma = fit_standardizer(X_ref)
    E_ref = embed_graphs(X_ref, h_max, mu, sigma)
    E_cal = embed_graphs(X_cal, h_max, mu, sigma)
    E_mod = {k: embed_graphs([prep_graph(g) for g in v], h_max, mu, sigma) for k, v in models.items()}
    rows = []
    for h in h_list:
        t0 = time.time()
        kw = dict(solver=solver, reg=reg, n_jobs=n_jobs, cache_dir=cache_dir)
        D_rr = distance_block(E_ref, E_ref, h, same=True, tag="ref-ref", **kw)
        D_rc = distance_block(E_ref, E_cal, h, tag="ref-cal", **kw)
        D_cc = distance_block(E_cal, E_cal, h, same=True, tag="cal-cal", **kw)
        lam = median_lambda(np.block([[D_rr, D_rc], [D_rc.T, D_cc]]))
        h_rows = [dict(model="ideal(X_cal)", h=h, **_analyze(D_rr, D_rc, D_cc, lam, n_perm, seed, do_c2st))]
        for name, E in E_mod.items():
            D_xy = distance_block(E_ref, E, h, tag=f"ref-{name}", **kw)
            D_yy = distance_block(E, E, h, same=True, tag=f"{name}-{name}", **kw)
            h_rows.append(dict(model=name, h=h, **_analyze(D_rr, D_xy, D_yy, lam, n_perm, seed, do_c2st)))
        base = next((r["mmd2"] for r in h_rows if r["model"] == baseline), None)
        for r in h_rows:
            r["skill_vs_baseline"] = (
                1.0 - r["mmd2"] / base if (base is not None and base > 0 and r["model"] != baseline) else float("nan")
            )
        rows += h_rows
        if verbose:
            print(f"  [h={h}] listo en {time.time() - t0:.1f}s (lambda={lam:.4g})", flush=True)
    return rows


def run_sensitivity(X_ref, X_cal, fracs=(0.1, 0.5, 2.0), sens_n=50, seed=0, **kw):
    """Test de sensibilidad: perturba X_cal con intercambios de aristas que
    preservan el grado y mide cuánto sube el MMD^2 respecto de X_ref.
    La fila 'ideal(X_cal)' corresponde a 0 intercambios."""
    s = min(sens_n, len(X_ref), len(X_cal))
    models = {
        f"swap_{f:g}m": [rewire_graph(G, f, seed + i) for i, G in enumerate(X_cal[:s])] for f in fracs
    }
    return run_wwl_validation(X_ref[:s], X_cal[:s], models, baseline=None, seed=seed, **kw)


def benchmark_pair_time(graphs, h_list, solver="emd", reg=0.05, n_pairs=20, seed=0) -> dict:
    """Mide segundos por par de grafos para cada h (para extrapolar el tiempo total)."""
    mu, sigma = fit_standardizer(graphs)
    embs = embed_graphs(graphs, max(h_list), mu, sigma)
    rng = np.random.default_rng(seed)
    pairs = [tuple(rng.choice(len(graphs), 2, replace=False)) for _ in range(n_pairs)]
    out = {}
    for h in h_list:
        t0 = time.time()
        for i, j in pairs:
            wasserstein1(embs[i][:, : h + 1], embs[j][:, : h + 1], solver, reg)
        out[h] = (time.time() - t0) / n_pairs
    return out


def write_csv(rows: list[dict], path) -> None:
    import csv

    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
