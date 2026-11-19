"""
Geometry and non-Gaussianity of the critical double random current (free boundary, no sources).

  * outermost clusters and their outer boundaries (conjecturally / by DCQ: CLE_4 in the limit)
  * box-counting dimension of the largest outermost cluster (7/4) and of its outer boundary (3/2)
  * finite-size scaling of its mass (L^(7/4)) and of its outer-boundary length (L^(3/2))
  * non-Gaussianity of h: excess kurtosis of
        - low Fourier (Dirichlet sine) modes of h, normalised so that the GFF of Theorem 1.3 gives
          variance 1 and kurtosis 0 (these are pairings (h, f) with smooth f, as in the theorem)
        - block averages of h over blocks of size L/32, L/16, L/8

Uses the sampler of drc_sim.py.  Library only: runs are started from custom_runs.py.  Same safety features as
drc_sim.run_all: independent chains, checkpoints every few minutes, resume by re-running.

Outer boundary: the clusters are drawn on the doubled grid (vertex (i,j) -> pixel (2i+1, 2j+1), an
edge -> the pixel between its endpoints, 1 pixel of padding).  The exterior is the 4-connected component
of the empty pixels that contains the padding.  A cluster is outermost iff it touches the exterior, and
its outer boundary is the set of its pixels next to the exterior (two disjoint clusters can never enclose
a region together, so this is the same as using the exterior of the cluster alone).
"""

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy import ndimage
from scipy.fft import dstn
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from . import drc_sim as d

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "drc_geo_runs")
N_MODES = 4                       # sine modes (j, k), 1 <= j, k <= N_MODES
BLOCK_FRACTIONS = (32, 16, 8)     # block size = L / fraction

CONFIG = {
    64:   dict(chains=12, samples=32000, burn_in=300),
    128:  dict(chains=12, samples=32000, burn_in=300),
    256:  dict(chains=12, samples=16000, burn_in=500),
    512:  dict(chains=12, samples=8000,  burn_in=500),
    1024: dict(chains=12, samples=4000,  burn_in=500),
}


# ----------------------------------------------------------------------------
# outer boundaries
# ----------------------------------------------------------------------------
def pixel_grid(L):
    """grid(L) plus the doubled-grid pixel (row, col) of every edge."""
    g = d.grid(L)
    i, j = g["u"] // L, g["u"] % L
    hor = np.arange(len(g["u"])) < g["n_h"]
    g["er"] = np.where(hor, 2 * i + 1, 2 * i + 2)
    g["ec"] = np.where(hor, 2 * j + 2, 2 * j + 1)
    return g


def geometry(support, g):
    """Clusters of the support and their outer boundaries.
    Returns lab (cluster of every vertex), sizes (vertices per cluster),
    cid ((2L+1)^2 image: cluster id of each pixel, -1 = empty) and bnd (outer-boundary pixels)."""
    L = g["L"]
    P = 2 * L + 1
    u, v = g["u"][support], g["v"][support]
    adj = coo_matrix((np.ones(len(u), dtype=np.int8), (u, v)), shape=(L * L, L * L))
    n, lab = connected_components(adj, directed=False)
    sizes = np.bincount(lab, minlength=n)
    cid = np.full((P, P), -1, dtype=np.int32)
    cid[g["er"][support], g["ec"][support]] = lab[u]
    nontrivial = np.zeros(n, dtype=bool)
    nontrivial[lab[u]] = True
    vi = np.flatnonzero(nontrivial[lab])
    cid[2 * (vi // L) + 1, 2 * (vi % L) + 1] = lab[vi]
    occ = cid >= 0
    comp, _ = ndimage.label(~occ)                      # 4-connectivity
    ext = comp == comp[0, 0]
    near = np.zeros_like(ext)
    near[1:] |= ext[:-1]
    near[:-1] |= ext[1:]
    near[:, 1:] |= ext[:, :-1]
    near[:, :-1] |= ext[:, 1:]
    return lab, sizes, cid, occ & near


def box_counts(rows, cols, K):
    """Number of boxes of side 2^k (k = 0..K) hit by the points (lattice units, integer arrays)."""
    out = np.empty(K + 1)
    for k in range(K + 1):
        out[k] = len(np.unique(((rows >> k) << 21) + (cols >> k)))
    return out


def largest_outermost(support, g):
    """Mass, outer-boundary length (edges) and box counts of the largest outermost cluster."""
    L = g["L"]
    K = int(np.log2(L)) - 1
    lab, sizes, cid, bnd = geometry(support, g)
    outer = np.unique(cid[bnd])
    k = outer[np.argmax(sizes[outer])]
    vtx = np.flatnonzero(lab == k)
    br, bc = np.nonzero(bnd & (cid == k))
    blen = int(np.sum((br + bc) % 2 == 1))            # edge pixels have one even coordinate
    return dict(mass=int(sizes[k]), blen=blen,
                n_cluster=box_counts(vtx // L, vtx % L, K),
                n_boundary=box_counts((br - 1) // 2, (bc - 1) // 2, K))


# ----------------------------------------------------------------------------
# non-Gaussianity of h
# ----------------------------------------------------------------------------
def mode_normalisation(L, M=N_MODES):
    j = np.arange(1, M + 1)
    c = 2 - 2 * np.cos(np.pi * j / L)
    return np.sqrt(np.pi * (c[:, None] + c[None, :]))   # z_jk = hhat_jk * this -> N(0,1) for the GFF


def modes(h, norm):
    M = norm.shape[0]
    return dstn(np.asarray(h, dtype=float), type=1, norm="ortho")[:M, :M] * norm


def blocks(h, L, fraction):
    b = L // fraction
    n = (L - 1) // b
    o = ((L - 1) - n * b) // 2
    return np.asarray(h[o:o + n * b, o:o + n * b], dtype=float).reshape(n, b, n, b).mean(axis=(1, 3))


def _moments(S, n):
    """variance and excess kurtosis from power sums S[1..4] (arrays of the same shape)."""
    mu = S[1] / n
    m2 = S[2] / n - mu ** 2
    m4 = S[4] / n - 4 * mu * S[3] / n + 6 * mu ** 2 * S[2] / n - 3 * mu ** 4
    return m2, m4 / m2 ** 2 - 3


# ----------------------------------------------------------------------------
# one chain with checkpoints
# ----------------------------------------------------------------------------
def chain_path(out_dir, L, chain):
    return os.path.join(out_dir, f"geoL{L}_chain{chain:02d}.pkl")


def run_chain(L, chain, samples, burn_in, seed=7000, out_dir=OUT_DIR, geom_every=10,
              checkpoint_minutes=5.0, max_seconds=None):
    os.makedirs(out_dir, exist_ok=True)
    path = chain_path(out_dir, L, chain)
    g = pixel_grid(L)
    x, y = np.tanh(d.J_CRIT), 1.0 / np.cosh(d.J_CRIT)
    K = int(np.log2(L)) - 1
    norm = mode_normalisation(L)
    rng = np.random.default_rng(np.random.SeedSequence([seed, L, chain, 1]))

    if os.path.exists(path):
        with open(path, "rb") as f:
            st = pickle.load(f)
        rng.bit_generator.state = st["rng"]
    else:
        taus = []
        for _ in range(2):
            t = (rng.integers(0, 2, size=g["n_faces"], dtype=np.int8) * 2 - 1).astype(np.int8)
            t[0] = 1
            taus.append(t)
        st = dict(L=L, chain=chain, burn_done=0, n_done=0, seconds=0.0, tau1=taus[0], tau2=taus[1],
                  rng=None, geom_every=geom_every,
                  modes=np.zeros((5, N_MODES, N_MODES)),
                  blocks={f: np.zeros((5, (L - 1) // (L // f), (L - 1) // (L // f)))
                          for f in BLOCK_FRACTIONS if L // f >= 1},
                  n_geo=0, geo=dict(mass=np.zeros(3), blen=np.zeros(3),
                                    log_ncl=np.zeros((2, K + 1)), log_nbd=np.zeros((2, K + 1))))
    st["burn_in"], st["target"] = burn_in, samples
    t_start = t_last = time.time()

    def checkpoint(force=False):
        nonlocal t_last
        now = time.time()
        wait = 60 * checkpoint_minutes if st["seconds"] > 0 else 60
        if force or now - t_last > wait:
            st["rng"] = rng.bit_generator.state
            st["seconds"] += now - t_last
            t_last = now
            d._save(st, path)

    def out_of_time():
        return max_seconds is not None and time.time() - t_start > max_seconds

    while st["burn_done"] < burn_in and not out_of_time():
        st["tau1"] = d.sw_step(st["tau1"], g, x, rng)
        st["tau2"] = d.sw_step(st["tau2"], g, x, rng)
        st["burn_done"] += 1
        checkpoint()

    while st["burn_done"] >= burn_in and st["n_done"] < samples and not out_of_time():
        st["tau1"] = d.sw_step(st["tau1"], g, x, rng)
        st["tau2"] = d.sw_step(st["tau2"], g, x, rng)
        a1, b1 = d.current_omega(st["tau1"], g, y, rng)
        a2, b2 = d.current_omega(st["tau2"], g, y, rng)
        support = a1 | b1 | a2 | b2
        h, _, _ = d.nesting(a1 ^ a2, support, g, rng)

        z = modes(h, norm)
        for p in range(5):
            st["modes"][p] += z ** p
        for f, S in st["blocks"].items():
            m = blocks(h, L, f)
            for p in range(5):
                S[p] += m ** p

        if st["n_done"] % geom_every == 0:
            r = largest_outermost(support, g)
            G = st["geo"]
            G["mass"] += [1, np.log(r["mass"]), r["mass"]]
            G["blen"] += [1, np.log(r["blen"]), r["blen"]]
            lc, lb = np.log(r["n_cluster"]), np.log(r["n_boundary"])
            G["log_ncl"] += [lc, lc ** 2]
            G["log_nbd"] += [lb, lb ** 2]
            st["n_geo"] += 1
        st["n_done"] += 1
        checkpoint()

    checkpoint(force=True)
    return f"geo L={L} chain {chain}: {st['n_done']}/{samples} samples, {st['seconds'] / 60:.1f} min"


def _task(args):
    return run_chain(*args[0], **args[1])


def run_all(config=CONFIG, workers=None, seed=7000, out_dir=OUT_DIR, **kwargs):
    """Run every chain of `config` on `workers` processes.  Interrupt and re-run to resume."""
    workers = workers or os.cpu_count()
    tasks = [((L, ch, cfg["samples"], cfg["burn_in"]), dict(seed=seed, out_dir=out_dir, **kwargs))
             for L, cfg in sorted(config.items(), reverse=True) for ch in range(cfg["chains"])]
    print(f"{len(tasks)} chains on {workers} processes, output in {os.path.abspath(out_dir)}", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(_task, t) for t in tasks]):
            print(time.strftime("%H:%M:%S"), fut.result(), flush=True)


# ----------------------------------------------------------------------------
# results
# ----------------------------------------------------------------------------
def load_chains(out_dir=OUT_DIR):
    chains = {}
    if not os.path.isdir(out_dir):
        return chains
    for name in sorted(os.listdir(out_dir)):
        if name.startswith("geoL") and name.endswith(".pkl"):
            with open(os.path.join(out_dir, name), "rb") as f:
                st = pickle.load(f)
            chains.setdefault(st["L"], []).append(st)
    return chains


def status(out_dir=OUT_DIR):
    chains = load_chains(out_dir)
    print(f"reading {out_dir}  ({sum(len(v) for v in chains.values())} checkpoint files)")
    for L, sts in sorted(chains.items()):
        done, target = sum(s["n_done"] for s in sts), sum(s["target"] for s in sts)
        print(f"L={L:5d}: {len(sts):2d} chains, {done:7d}/{target} samples ({100 * done / max(1, target):5.1f}%), "
              f"{sum(s['n_geo'] for s in sts)} geometry samples, {sum(s['seconds'] for s in sts) / 3600:.2f} core-hours")


def _mean_se(z):
    z = np.asarray(z, dtype=float)
    return z.mean(axis=0), (z.std(axis=0, ddof=1) / np.sqrt(len(z)) if len(z) > 1 else np.full(z.shape[1:], np.nan))


def _fit_slope(logn, k_lo, k_hi):
    """D from log N(2^k) = c - D k log 2 over k_lo..k_hi."""
    k = np.arange(k_lo, k_hi + 1)
    return -np.polyfit(k * np.log(2), logn[k_lo:k_hi + 1], 1)[0]


def summarize(out_dir=OUT_DIR, min_samples=200, fit_from=2, fit_to_fraction=16):
    """Prints and returns, per L (mean +- standard error over chains):
    modes: variance (GFF: 1) and excess kurtosis (GFF: 0) of the normalised sine modes
    blocks: excess kurtosis of block averages (interior blocks), block = L/32, L/16, L/8
    geometry: box-counting dimension of the largest outermost cluster and of its outer boundary,
              fitted for box sides 2^fit_from .. L/fit_to_fraction; local slopes; mass and boundary length."""
    res = {}
    for L, sts in sorted(load_chains(out_dir).items()):
        sts = [s for s in sts if s["n_done"] >= min_samples]
        if len(sts) < 2:
            continue
        r = dict(chains=len(sts), samples=sum(s["n_done"] for s in sts), n_geo=sum(s["n_geo"] for s in sts))
        var_m, kur_m = zip(*[_moments(s["modes"], s["n_done"]) for s in sts])
        r["mode_var"], r["mode_var_se"] = _mean_se(var_m)
        r["mode_kurt"], r["mode_kurt_se"] = _mean_se(kur_m)
        r["block_kurt"] = {}
        for f in BLOCK_FRACTIONS:
            if f not in sts[0]["blocks"]:
                continue
            per = []
            for s in sts:
                _, kur = _moments(s["blocks"][f], s["n_done"])
                n = kur.shape[0]
                q = n // 4
                per.append(kur[q:n - q, q:n - q].mean())
            r["block_kurt"][f] = _mean_se(per)
        geo = [s for s in sts if s["n_geo"] >= 2]
        if len(geo) >= 2:
            K = int(np.log2(L)) - 1
            k_hi = min(K, int(np.log2(L // fit_to_fraction)))
            lcl = np.array([s["geo"]["log_ncl"][0] / s["n_geo"] for s in geo])
            lbd = np.array([s["geo"]["log_nbd"][0] / s["n_geo"] for s in geo])
            r["eps"] = 2.0 ** np.arange(K + 1)
            r["local_cluster"] = _mean_se(-np.diff(lcl, axis=1) / np.log(2))
            r["local_boundary"] = _mean_se(-np.diff(lbd, axis=1) / np.log(2))
            if k_hi > fit_from:                      # need at least two box sizes in the window
                r["D_cluster"] = _mean_se([_fit_slope(z, fit_from, k_hi) for z in lcl])
                r["D_boundary"] = _mean_se([_fit_slope(z, fit_from, k_hi) for z in lbd])
                r["fit_range"] = (2 ** fit_from, 2 ** k_hi)
            r["log_mass"] = _mean_se([s["geo"]["mass"][1] / s["n_geo"] for s in geo])
            r["log_blen"] = _mean_se([s["geo"]["blen"][1] / s["n_geo"] for s in geo])
            r["mass"] = _mean_se([s["geo"]["mass"][2] / s["n_geo"] for s in geo])
            r["blen"] = _mean_se([s["geo"]["blen"][2] / s["n_geo"] for s in geo])
        res[L] = r

    print("Non-Gaussianity of h (GFF: variance ratio 1, excess kurtosis 0)")
    for L, r in res.items():
        mv, ms = r["mode_var"], r["mode_var_se"]
        mk, mks = r["mode_kurt"], r["mode_kurt_se"]
        line = "  ".join(f"({j + 1},{j + 1}): var {mv[j, j]:.3f}+-{ms[j, j]:.3f} kurt {mk[j, j]:+.3f}+-{mks[j, j]:.3f}"
                         for j in range(min(3, N_MODES)))
        print(f"  L={L:5d} ({r['samples']} samples) modes  {line}")
        bl = "  ".join(f"L/{f}: {m:+.3f}+-{s:.3f}" for f, (m, s) in r["block_kurt"].items())
        print(f"  {'':12s} block-average kurtosis  {bl}")
    print("Geometry of the largest outermost cluster (cluster 7/4 = 1.75, outer boundary 3/2 = 1.5)")
    for L, r in res.items():
        if "D_cluster" in r:
            print(f"  L={L:5d} ({r['n_geo']} samples): box counting, boxes {r['fit_range'][0]}..{r['fit_range'][1]}: "
                  f"cluster {r['D_cluster'][0]:.3f}+-{r['D_cluster'][1]:.3f}, boundary "
                  f"{r['D_boundary'][0]:.3f}+-{r['D_boundary'][1]:.3f} | mass {r['mass'][0]:.0f}, boundary length {r['blen'][0]:.0f}")
    Ls = [L for L in res if "log_mass" in res[L]]
    for a, b in zip(Ls[:-1], Ls[1:]):
        dm = (res[b]["log_mass"][0] - res[a]["log_mass"][0]) / np.log(b / a)
        db = (res[b]["log_blen"][0] - res[a]["log_blen"][0]) / np.log(b / a)
        sm = np.hypot(res[a]["log_mass"][1], res[b]["log_mass"][1]) / np.log(b / a)
        sb = np.hypot(res[a]["log_blen"][1], res[b]["log_blen"][1]) / np.log(b / a)
        print(f"  L {a}->{b}: mass exponent {dm:.3f}+-{sm:.3f}, boundary-length exponent {db:.3f}+-{sb:.3f}")
    return res


# ----------------------------------------------------------------------------
# plots
# ----------------------------------------------------------------------------
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#d6d5d0"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
RAMP = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b")   # blue ramp, light -> dark = small -> large L


def plot_dimensions(res):
    """Local box-counting slopes against box size / L, one line per L, for cluster and outer boundary."""
    import matplotlib.pyplot as plt
    Ls = [L for L in res if "local_cluster" in res[L]]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, key, target, name in ((axes[0], "local_cluster", 1.75, "largest outermost cluster"),
                                  (axes[1], "local_boundary", 1.5, "its outer boundary")):
        for i, L in enumerate(Ls):
            m, s = res[L][key]
            e = np.sqrt(res[L]["eps"][:-1] * res[L]["eps"][1:]) / L
            ax.errorbar(e, m, yerr=s, color=RAMP[i % len(RAMP)], marker="o", ms=4, lw=2, capsize=2, label=f"L = {L}")
        ax.axhline(target, color=INK2, lw=1, ls="--")
        ax.set_xscale("log")
        ax.set_xlabel("box size / L")
        ax.set_ylabel("local box-counting slope")
        ax.set_title(f"{name} (dashed: {target})")
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def plot_kurtosis(res):
    """Excess kurtosis against L: low sine modes and block averages (GFF: 0)."""
    import matplotlib.pyplot as plt
    Ls = np.array(sorted(res))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for i, j in enumerate(range(min(3, N_MODES))):
        m = [res[L]["mode_kurt"][j, j] for L in Ls]
        s = [res[L]["mode_kurt_se"][j, j] for L in Ls]
        axes[0].errorbar(Ls, m, yerr=s, color=SERIES[i], marker="o", ms=5, lw=2, capsize=3, label=f"mode ({j + 1},{j + 1})")
    axes[0].set_title("sine modes of h")
    for i, f in enumerate(BLOCK_FRACTIONS):
        Lf = [L for L in Ls if f in res[L]["block_kurt"]]
        m = [res[L]["block_kurt"][f][0] for L in Lf]
        s = [res[L]["block_kurt"][f][1] for L in Lf]
        axes[1].errorbar(Lf, m, yerr=s, color=SERIES[i], marker="o", ms=5, lw=2, capsize=3, label=f"block L/{f}")
    axes[1].set_title("block averages of h")
    for ax in axes:
        ax.axhline(0, color=INK2, lw=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks(Ls)
        ax.set_xticklabels(Ls)
        ax.set_xlabel("L")
        ax.set_ylabel("excess kurtosis (GFF: 0)")
        ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def draw_boundaries(L, omega1, omega2, h=None, window=None, min_size=None, dpi=150):
    """Outermost clusters (grey) with their outer boundaries (ink; the three largest in colour),
    and, if h is given, h on the faces with the same boundaries.  window = (row, col, size) in
    lattice units to zoom (recommended for L >= 512).  Only clusters with >= min_size vertices."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    g = pixel_grid(L)
    lab, sizes, cid, bnd = geometry(omega1 | omega2, g)
    outer = np.unique(cid[bnd])
    min_size = min_size if min_size is not None else max(10, int(0.002 * L ** 1.75))
    keep = outer[sizes[outer] >= min_size]
    top = keep[np.argsort(sizes[keep])[::-1][:3]]
    kept = np.zeros(len(sizes), dtype=bool)
    kept[keep] = True
    pix = cid >= 0
    in_keep = pix & kept[np.where(pix, cid, 0)]
    img = np.ones(cid.shape + (3,))
    img[in_keep] = to_rgb(MUTED)
    bk = bnd & in_keep
    img[bk] = to_rgb(INK)
    for c, col in zip(top, SERIES):
        img[bnd & (cid == c)] = to_rgb(col)
    if window is not None:
        r0, c0, w = window
        sl = (slice(2 * r0, 2 * (r0 + w) + 1), slice(2 * c0, 2 * (c0 + w) + 1))
    else:
        sl = (slice(None), slice(None))
    n_pan = 2 if h is not None else 1
    fig, axes = plt.subplots(1, n_pan, figsize=(7.5 * n_pan, 7.5), dpi=dpi, squeeze=False)
    axes = axes[0]
    axes[0].imshow(img[sl], interpolation="nearest")
    axes[0].set_title(f"outermost clusters >= {min_size} vertices, outer boundaries (3 largest in colour), L = {L}",
                      fontsize=9)
    if h is not None:
        hp = np.zeros(cid.shape)
        hp[2:2 * L, 2:2 * L] = np.kron(h, np.ones((2, 2)))
        v = max(1, np.abs(h).max())
        axes[1].imshow(hp[sl], cmap="RdBu", vmin=-v, vmax=v, interpolation="nearest")
        over = np.zeros(cid.shape + (4,))
        over[bk] = (0, 0, 0, 1)
        axes[1].imshow(over[sl], interpolation="nearest")
        axes[1].set_title("h on the faces with the same outer boundaries (black)", fontsize=9)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    return fig


def benchmark(Ls=(64, 128, 256, 512, 1024), seconds=8.0, config=CONFIG, workers=None, geom_every=10):
    """Time per sample (including the extra measurements) and projected duration of `config`."""
    workers = workers or os.cpu_count()
    x, y = np.tanh(d.J_CRIT), 1.0 / np.cosh(d.J_CRIT)
    total = 0.0
    for L in Ls:
        g = pixel_grid(L)
        rng = np.random.default_rng(L)
        norm = mode_normalisation(L)
        t1 = np.ones(g["n_faces"], dtype=np.int8)
        t2 = t1.copy()
        for _ in range(5):
            t1, t2 = d.sw_step(t1, g, x, rng), d.sw_step(t2, g, x, rng)
        n, t0 = 0, time.time()
        while time.time() - t0 < seconds or n < 3:
            t1, t2 = d.sw_step(t1, g, x, rng), d.sw_step(t2, g, x, rng)
            a1, b1 = d.current_omega(t1, g, y, rng)
            a2, b2 = d.current_omega(t2, g, y, rng)
            sup = a1 | b1 | a2 | b2
            h, _, _ = d.nesting(a1 ^ a2, sup, g, rng)
            modes(h, norm)
            for f in BLOCK_FRACTIONS:
                blocks(h, L, f)
            if n % geom_every == 0:
                largest_outermost(sup, g)
            n += 1
        per = (time.time() - t0) / n
        line = f"L={L:5d}: {1000 * per:8.2f} ms per sample"
        if L in config:
            cfg = config[L]
            core_h = cfg["chains"] * (cfg["samples"] + 0.7 * cfg["burn_in"]) * per / 3600
            total += core_h
            line += f" | {cfg['chains']} x {cfg['samples']} = {core_h:5.1f} core-hours"
        print(line, flush=True)
    print(f"total {total:.1f} core-hours = about {total / workers:.1f} hours on {workers} processes")
