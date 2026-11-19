#!/usr/bin/env python3
"""
run_validation.py - Validación de GRAIP con WWL
================================================
MMD^2 con kernel WWL + test de permutación + razón de diversidad + C2ST +
test de sensibilidad, comparando GRAIP contra baselines (configuración, BA dual)
y contra un "generador ideal" (segundo conjunto de grafos muestreados).

Uso típico
----------
# 0) Prueba de humo sin datos reales ni tu GRAIP (usa fake_graip.py):
python run_validation.py --synthetic 60 2 --name demo --graip-spec fake_graip:generate --n-graphs 20

# 1) Estimar el tiempo ANTES de correr lo pesado (mide 20 pares y extrapola):
python run_validation.py --edgelist ecoli.tsv --name ecoli --graip-spec my_graip_adapter:generate --benchmark

# 2) Validación completa (reanudable; todo se guarda en results/<name>/cache):
python run_validation.py --edgelist ecoli.tsv --name ecoli --graip-spec my_graip_adapter:generate \
       --n-graphs 100 --h-list 0 1 2 3 --n-jobs 8

# 3) Dosis-respuesta en n_g: genera con n_g = 3, 4, 5 (cada uno con su --graip-name) o pasa
#    listas de grafos ya generados (pickle de list[nx.Graph]) como modelos extra:
python run_validation.py --edgelist ecoli.tsv --name ecoli --graph-file graip_ng3=ng3.pkl \
       --graph-file graip_ng4=ng4.pkl --graph-file graip_ng5=ng5.pkl

Formato de --edgelist: una arista por línea, "u v p" (separado por espacios, tabs o comas),
con p en (0, 1]. Líneas que empiezan con # se ignoran. Usa un --name distinto por red:
la caché (grafos muestreados, generados y matrices de distancias) cuelga de ese nombre.

IMPORTANTE: los .pkl de grafos son pickles; carga solo archivos que tú mismo generaste.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import os
import pickle
import sys
import time
from math import comb
from pathlib import Path

import networkx as nx
import numpy as np
from joblib import Parallel, delayed

import wwl_validation as wv


# ----------------------------------------------------------------------------- E/S
def read_uncertain_edgelist(path) -> nx.Graph:
    G = nx.Graph()
    with open(path) as f:
        for ln, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 3:
                raise ValueError(f"{path}:{ln + 1}: se esperan 3 columnas (u v p)")
            try:
                p = float(parts[2])
            except ValueError:
                if ln == 0:  # encabezado
                    continue
                raise
            u, v = parts[0], parts[1]
            if u == v:
                continue
            if not 0 < p <= 1:
                raise ValueError(f"{path}:{ln + 1}: p={p} fuera de (0, 1]")
            if G.has_edge(u, v):
                p = max(p, G[u][v]["p"])
            G.add_edge(u, v, p=p)
    return G


def synthetic_uncertain(n: int, m_attach: int, seed: int) -> nx.Graph:
    rng = np.random.default_rng(seed)
    G = nx.barabasi_albert_graph(n, m_attach, seed=seed)
    for u, v in G.edges():
        G[u][v]["p"] = float(rng.uniform(0.05, 1.0))
    return G


def network_fingerprint(Gu: nx.Graph) -> str:
    items = sorted((min(str(u), str(v)), max(str(u), str(v)), round(d["p"], 6)) for u, v, d in Gu.edges(data=True))
    return hashlib.sha1(repr(items).encode()).hexdigest()[:8]


def atomic_pickle(obj, path: Path) -> None:
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


def load_graph_file(path) -> list[nx.Graph]:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return [g if isinstance(g, nx.Graph) else nx.Graph(list(g)) for g in obj]


def load_callable(spec: str):
    """'modulo:funcion' -> función. Añade el directorio actual al path (también para los procesos hijos)."""
    mod, _, fn = spec.partition(":")
    cwd = str(Path.cwd())
    sys.path.insert(0, cwd)
    os.environ["PYTHONPATH"] = cwd + os.pathsep + os.environ.get("PYTHONPATH", "")
    return getattr(importlib.import_module(mod), fn or "generate")


# ------------------------------------------------------------------ generación (reanudable)
def generate_cached(name, gen_fn, Gu, n_graphs, seed, n_jobs, cache_dir: Path, chunk=None):
    """Genera grafos con gen_fn(Gu, seed) -> nx.Graph, guardando tras cada tanda.
    Si el proceso se interrumpe, al relanzar continúa donde quedó."""
    path = cache_dir / f"graphs_{name}.pkl"
    graphs = pickle.load(open(path, "rb")) if path.exists() else []
    chunk = chunk or max(1, n_jobs)
    t0 = time.time()
    start0 = len(graphs)
    while len(graphs) < n_graphs:
        start = len(graphs)
        k = min(chunk, n_graphs - start)
        new = Parallel(n_jobs=n_jobs)(delayed(gen_fn)(Gu, seed + start + i) for i in range(k))
        graphs += [wv.prep_graph(g) for g in new]
        atomic_pickle(graphs, path)
        done = len(graphs) - start0
        el = time.time() - t0
        print(f"  [{name}] {len(graphs)}/{n_graphs} grafos "
              f"({el / done:.1f} s/grafo, ~{el / done * (n_graphs - len(graphs)) / 60:.1f} min restantes)", flush=True)
    return graphs[:n_graphs]


def get_cached(name, build, n_graphs, cache_dir: Path):
    path = cache_dir / f"graphs_{name}.pkl"
    if path.exists():
        g = pickle.load(open(path, "rb"))
        if len(g) >= n_graphs:
            return g[:n_graphs]
    g = build()
    atomic_pickle(g, path)
    return g


# ------------------------------------------------------------------------- benchmark
def run_benchmark(args, X_all, n_models):
    h_list = sorted(args.h_list)
    t = wv.benchmark_pair_time(X_all[: min(len(X_all), 30)], h_list, args.solver, args.reg)
    N = args.n_graphs
    s = min(args.sens_n, N)
    F = 0 if args.no_sensitivity else len(args.sens_fracs)
    main_pairs = N * (2 * N - 1) + n_models * (N * N + comb(N, 2))
    sens_pairs = (s * (2 * s - 1) + F * (s * s + comb(s, 2))) if F else 0
    print(f"\nBenchmark (solver={args.solver}, POT={'sí' if wv.HAS_POT else 'NO (usa LP de scipy, mucho más lento)'})")
    print(f"  N={N} grafos por conjunto, {n_models} modelos generados, sensibilidad: {F} niveles con {s} grafos\n")
    print(f"  {'h':>2} {'s/par':>9} {'pares':>9} {'1 núcleo':>10} {f'{args.n_jobs} núcleos':>10}")
    tot = 0.0
    for h in h_list:
        pairs = main_pairs + sens_pairs
        sec = t[h] * pairs
        tot += sec
        print(f"  {h:>2} {t[h]:>9.4f} {pairs:>9d} {sec / 3600:>9.2f}h {sec / 3600 / args.n_jobs:>9.2f}h")
    print(f"\n  Total WWL: ~{tot / 3600:.2f} h con 1 núcleo, ~{tot / 3600 / args.n_jobs:.2f} h con {args.n_jobs} núcleos.")
    print("  NO incluye el tiempo de generar grafos con GRAIP (mídelo con tu implementación; suele dominar).")


# ----------------------------------------------------------------------------- salida
def print_table(rows, title):
    print(f"\n=== {title} ===")
    print(f"{'modelo':<18}{'h':>2} {'MMD2':>10} {'p_perm':>7} {'z':>7} {'rho [IC95%]':>19} {'C2ST':>6} {'skill':>7} {'min_eig':>9}")
    for r in rows:
        rho = f"{r['rho']:.2f} [{r['rho_lo']:.2f},{r['rho_hi']:.2f}]"
        print(f"{r['model']:<18}{r['h']:>2} {r['mmd2']:>10.5f} {r['p_perm']:>7.3f} {r['z']:>7.2f} "
              f"{rho:>19} {r['c2st_acc']:>6.2f} {r.get('skill_vs_baseline', float('nan')):>7.2f} {r['min_eig']:>9.1e}")


SURFACE, INK1, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CATEG = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE_ORDINAL = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]  # pasos 250..650 de la rampa azul


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=AXIS, labelcolor=INK2, labelsize=8.5, length=3)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _direct_labels(ax, ends, x):
    """Etiquetas al final de cada serie (texto en tinta secundaria), sin solaparse."""
    ax.figure.canvas.draw()
    ys = np.array([ax.transData.transform((x, y))[1] for y, _ in ends])
    order = np.argsort(ys)
    pos = ys[order].copy()
    for k in range(1, len(pos)):
        pos[k] = max(pos[k], pos[k - 1] + 21)
    pos -= (pos.mean() - ys.mean())
    for k, p in zip(order, pos):
        ydat = ax.transData.inverted().transform((0, p))[1]
        ax.annotate(ends[k][1], (x, ydat), xytext=(9, 0), textcoords="offset points", va="center",
                    fontsize=8.5, color=INK2, annotation_clip=False)


def _plot_series(ax, xs, series, xlabels=None):
    """series: lista de (nombre, color, ys, filled_mask, dashed)."""
    handles = []
    for name, color, ys, filled, dashed in series:
        ax.plot(xs, ys, color=color, linewidth=1.6, linestyle="--" if dashed else "-", zorder=2)
        ax.scatter(xs, ys, s=46, facecolors=[color if f else SURFACE for f in filled], edgecolors=color,
                   linewidths=1.6, zorder=3)
        handles.append(ax.plot([], [], color=color, linewidth=1.6, linestyle="--" if dashed else "-",
                               marker="o", markersize=6, markerfacecolor=color, markeredgecolor=color, label=name)[0])
    ax.axhline(0, color=AXIS, linewidth=1.0, zorder=1)
    if xlabels is not None:
        ax.set_xticks(xs)
        ax.set_xticklabels(xlabels)
    span = xs[-1] - xs[0] if len(xs) > 1 else 1
    ax.set_xlim(xs[0] - 0.05 * span, xs[-1] + (0.30 if len(series) <= 4 else 0.08) * span)
    if len(series) <= 4:
        _direct_labels(ax, [(s[2][-1], s[0]) for s in series], xs[-1])
    ax.legend(handles=handles, frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper center",
              bbox_to_anchor=(0.5, -0.17), ncol=min(4, len(handles)), handletextpad=0.4, columnspacing=1.4)


def plot_summary(rows, sens_rows, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h_list = sorted({r["h"] for r in rows})
    names = []
    for r in rows:
        if r["model"] not in names and not r["model"].startswith("ideal"):
            names.append(r["model"])
    preferred = [n for n in ("graip", "config", "ba") if n in names]
    names = preferred + [n for n in names if n not in preferred]  # el color sigue a la entidad
    if len(names) > len(CATEG):
        raise ValueError("Más de 8 modelos: agrupa o facetea en vez de generar más colores.")
    color = {n: CATEG[i] for i, n in enumerate(names)}

    ncols = 2 if sens_rows else 1
    fig, axes = plt.subplots(1, ncols, figsize=(5.6 * ncols + 0.6, 4.9), dpi=150, facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    # el layout se fija ANTES de dibujar: las etiquetas directas se posicionan con la geometría final
    fig.subplots_adjust(bottom=0.27, left=0.09, right=0.97, top=0.92, wspace=0.28)
    ax = axes[0]
    _style(ax)
    series = []
    for n in ["ideal(X_cal)"] + names:
        rr = sorted([r for r in rows if r["model"] == n], key=lambda r: r["h"])
        if not rr:
            continue
        series.append((n, MUTED if n.startswith("ideal") else color[n], [r["mmd2"] for r in rr],
                       [r["p_perm"] < 0.05 for r in rr], n.startswith("ideal")))
    _plot_series(ax, h_list, series, xlabels=[str(h) for h in h_list])
    ax.set_title("MMD² (kernel WWL) vs. profundidad WL", loc="left", fontsize=10.5, color=INK1)
    ax.set_xlabel("h (iteraciones de WL)", fontsize=9, color=INK2)
    ax.set_ylabel("MMD² insesgado (menor = más parecido)", fontsize=9, color=INK2)

    if sens_rows:
        ax2 = axes[1]
        _style(ax2)
        hs = sorted({r["h"] for r in sens_rows})
        labels = []
        for r in sens_rows:
            if r["h"] == hs[0]:
                labels.append(r["model"])
        xl = ["0" if l.startswith("ideal") else l.replace("swap_", "") for l in labels]
        idx = np.linspace(0, len(BLUE_ORDINAL) - 1, len(hs)).round().astype(int)
        ser2 = []
        for k, h in enumerate(hs):
            ys = [next(r for r in sens_rows if r["h"] == h and r["model"] == l)["mmd2"] for l in labels]
            ps = [next(r for r in sens_rows if r["h"] == h and r["model"] == l)["p_perm"] for l in labels]
            ser2.append((f"h={h}", BLUE_ORDINAL[idx[k]], ys, [p < 0.05 for p in ps], False))
        _plot_series(ax2, list(range(len(labels))), ser2, xlabels=xl)
        ax2.set_title("Sensibilidad: swaps que preservan el grado", loc="left", fontsize=10.5, color=INK1)
        ax2.set_xlabel("intercambios de aristas aplicados a X_cal (m = nº de aristas)", fontsize=9, color=INK2)
        ax2.set_ylabel("MMD² insesgado", fontsize=9, color=INK2)

    fig.text(0.012, 0.012, "Marcador relleno: p < 0.05 en el test de permutación (distinguible de la referencia). "
             "Hueco: no distinguible.   ideal(X_cal): segundo conjunto de grafos muestreados.",
             fontsize=8, color=MUTED)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--edgelist", help="archivo 'u v p' con la red probabilística")
    src.add_argument("--synthetic", nargs=2, type=int, metavar=("N", "M"), help="red BA incierta sintética (n, m)")
    ap.add_argument("--name", required=True, help="nombre de la red (carpeta de resultados y caché)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--graip-spec", help="'modulo:funcion' con generate(uncertain_graph, seed) -> nx.Graph")
    ap.add_argument("--graip-name", default="graip", help="nombre del modelo en los resultados/caché")
    ap.add_argument("--graph-file", action="append", default=[], metavar="NOMBRE=ARCHIVO.pkl",
                    help="modelo extra: pickle con list[nx.Graph] (repetible)")
    ap.add_argument("--baselines", nargs="*", default=["config", "ba"], choices=["config", "ba"])
    ap.add_argument("--n-graphs", type=int, default=100, help="N: grafos por conjunto (paper GRAIP: 100)")
    ap.add_argument("--h-list", nargs="+", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--solver", choices=["emd", "sinkhorn"], default="emd")
    ap.add_argument("--reg", type=float, default=0.05, help="epsilon relativo de Sinkhorn")
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--gen-chunk", type=int, default=None, help="grafos por tanda al generar (default: n-jobs)")
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--no-c2st", action="store_true")
    ap.add_argument("--no-sensitivity", action="store_true")
    ap.add_argument("--sens-fracs", nargs="+", type=float, default=[0.1, 0.5, 2.0])
    ap.add_argument("--sens-n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--benchmark", action="store_true", help="solo estima el tiempo y termina")
    args = ap.parse_args()
    args.h_list = sorted(set(args.h_list))

    Gu = read_uncertain_edgelist(args.edgelist) if args.edgelist else synthetic_uncertain(*args.synthetic, args.seed)
    out = Path(args.out) / args.name
    cache = out / f"cache_{network_fingerprint(Gu)}"
    cache.mkdir(parents=True, exist_ok=True)
    N = args.n_graphs
    print(f"Red '{args.name}': {Gu.number_of_nodes()} nodos, {Gu.number_of_edges()} aristas, "
          f"p media = {np.mean([d['p'] for *_, d in Gu.edges(data=True)]):.2f}")
    print(f"Solver: {args.solver} | POT: {'sí' if wv.HAS_POT else 'no (LP de scipy)'} | n_jobs={args.n_jobs}")

    # 1) referencia: 2N mundos posibles (N de referencia + N para calibración "ideal")
    X_all = get_cached("X", lambda: wv.sample_possible_graphs(Gu, 2 * N, args.seed), 2 * N, cache)
    X_ref, X_cal = X_all[:N], X_all[N: 2 * N]

    n_models = (1 if args.graip_spec else 0) + len(args.baselines) + len(args.graph_file)
    if args.benchmark:
        run_benchmark(args, X_all, n_models)
        return

    # 2) grafos generados por cada modelo
    models: dict[str, list[nx.Graph]] = {}
    if args.graip_spec:
        gen = load_callable(args.graip_spec)
        print(f"Generando con {args.graip_spec} ...")
        models[args.graip_name] = generate_cached(args.graip_name, gen, Gu, N, args.seed + 1_000_000,
                                                  args.n_jobs, cache, args.gen_chunk)
    if "config" in args.baselines:
        models["config"] = get_cached("config", lambda: wv.make_config_graphs(Gu, N, args.seed + 2_000_000), N, cache)
    if "ba" in args.baselines:
        models["ba"] = get_cached("ba", lambda: wv.make_dual_ba_graphs(X_ref, N, args.seed + 3_000_000), N, cache)
    for spec in args.graph_file:
        nm, _, path = spec.partition("=")
        models[nm] = load_graph_file(path)[:N]
        if len(models[nm]) < N:
            print(f"  aviso: {nm} tiene {len(models[nm])} < {N} grafos")
    if not models:
        sys.exit("No hay modelos que evaluar: usa --graip-spec, --baselines o --graph-file.")

    # 3) validación WWL
    common = dict(solver=args.solver, reg=args.reg, n_jobs=args.n_jobs, cache_dir=cache,
                  n_perm=args.n_perm, seed=args.seed, do_c2st=not args.no_c2st)
    print("\nValidación WWL (MMD², permutación, diversidad, C2ST) ...")
    baseline = "config" if "config" in models else None
    rows = wv.run_wwl_validation(X_ref, X_cal, models, h_list=args.h_list, baseline=baseline, **common)
    wv.write_csv(rows, out / "results.csv")
    print_table(rows, f"Resultados ({args.name})")

    sens_rows = []
    if not args.no_sensitivity:
        print("\nTest de sensibilidad (swaps que preservan el grado) ...")
        sens_rows = wv.run_sensitivity(X_ref, X_cal, fracs=args.sens_fracs, sens_n=args.sens_n,
                                       h_list=args.h_list, **{**common, "do_c2st": False})
        wv.write_csv(sens_rows, out / "sensitivity.csv")
        print_table(sens_rows, "Sensibilidad")

    plot_summary(rows, sens_rows, out / "summary.png")
    print(f"\nListo. Archivos en {out}/: results.csv" + (", sensitivity.csv" if sens_rows else "") + ", summary.png")


if __name__ == "__main__":
    main()
