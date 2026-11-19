"""
Long-run simulation of the critical double random current on the L x L square
grid (free boundary, no sources) and its nesting-field height function.

Each chain is an independent process-safe unit: it keeps running sums, writes a
checkpoint file every few minutes (atomic write) and resumes from it if the
run is interrupted.  Library only: runs are started from custom_runs.py (project folder), results are read by the monitor notebooks.
"""

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # one thread per process; parallelism = processes

import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

J_CRIT = 0.5 * np.log(1 + np.sqrt(2))
# results folder lives next to this file, so the run (Anaconda Prompt) and the notebooks
# read/write the same place whatever their working directories are
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "drc_runs")

# Standard run: chains x samples per chain, burn-in sweeps.  Rerunning with a larger "samples" extends the
# existing chains instead of starting again; the extensions are listed in custom_runs.py, not here.
CONFIG = {
    32:   dict(chains=24, samples=42000, burn_in=300),
    64:   dict(chains=24, samples=42000, burn_in=300),
    128:  dict(chains=24, samples=42000, burn_in=500),
    256:  dict(chains=24, samples=42000, burn_in=500),
    512:  dict(chains=12, samples=16000, burn_in=500),
    1024: dict(chains=12, samples=4000,  burn_in=500),
}


# ----------------------------------------------------------------------------
# lattice, chain, nesting field
# ----------------------------------------------------------------------------
def grid(L):
    """Edges (u, v) and the two faces (f1, f2) of each edge. Face 0 = outer face,
    face 1 + a*(L-1) + b = unit square with top-left vertex (a, b). Horizontal edges first."""
    def inner(a, b):
        ok = (a >= 0) & (b >= 0) & (a <= L - 2) & (b <= L - 2)
        return np.where(ok, 1 + a * (L - 1) + b, 0)

    i, j = (t.ravel() for t in np.meshgrid(np.arange(L), np.arange(L - 1), indexing="ij"))
    hu, hf1, hf2 = i * L + j, inner(i - 1, j), inner(i, j)
    i, j = (t.ravel() for t in np.meshgrid(np.arange(L - 1), np.arange(L), indexing="ij"))
    vu, vf1, vf2 = i * L + j, inner(i, j - 1), inner(i, j)
    return dict(L=L, n_h=len(hu), n_faces=1 + (L - 1) ** 2,
                u=np.concatenate([hu, vu]), v=np.concatenate([hu + 1, vu + L]),
                f1=np.concatenate([hf1, vf1]), f2=np.concatenate([hf2, vf2]),
                v_row=vu // L, v_col=vu % L)


def sw_step(tau, g, x, rng):
    """One Swendsen-Wang sweep on the face signs (outer face fixed to +1)."""
    f1, f2 = g["f1"], g["f2"]
    bond = (tau[f1] == tau[f2]) & (rng.random(len(f1)) < 1.0 - x)
    adj = coo_matrix((np.ones(bond.sum(), dtype=np.int8), (f1[bond], f2[bond])),
                     shape=(g["n_faces"], g["n_faces"]))
    n_clusters, label = connected_components(adj, directed=False)
    sign = (rng.integers(0, 2, size=n_clusters, dtype=np.int8) * 2 - 1).astype(np.int8)
    sign[label[0]] = 1
    return sign[label]


def current_omega(tau, g, y, rng):
    """omega = (odd edges, even nonzero edges) of one sourceless current (Lemma 3.1)."""
    odd = tau[g["f1"]] != tau[g["f2"]]
    even = (~odd) & (rng.random(len(odd)) < 1.0 - y)
    return odd, even


def nesting(omega1, support, g, rng):
    """Nesting field h, number of odd clusters n_odd (both (L-1, L-1)) and the
    number of vertices of the largest cluster."""
    L = g["L"]
    adj = coo_matrix((np.ones(support.sum(), dtype=np.int8), (g["u"][support], g["v"][support])),
                     shape=(L * L, L * L))
    n_comp, lab = connected_components(adj, directed=False)
    eps = rng.integers(0, 2, size=n_comp) * 2 - 1
    largest = int(np.bincount(lab).max())

    h = np.zeros((L - 1, L - 1), dtype=np.int32)
    n_odd = np.zeros((L - 1, L - 1), dtype=np.int32)
    idx = np.flatnonzero(omega1[g["n_h"]:])              # vertical odd edges
    if len(idx) == 0:
        return h, n_odd, largest
    a, j = g["v_row"][idx], g["v_col"][idx]
    c = lab[a * L + j]
    order = np.lexsort((j, c, a))
    a, j, c = a[order], j[order], c[order]
    new = np.r_[True, (a[1:] != a[:-1]) | (c[1:] != c[:-1])]
    starts = np.flatnonzero(new)
    sizes = np.diff(np.r_[starts, len(a)])
    assert np.all(sizes % 2 == 0)
    k = np.arange(len(a)) - np.repeat(starts, sizes)
    first = np.flatnonzero(k % 2 == 0)
    for out, w in ((h, eps[c[first]]), (n_odd, np.ones(len(first), dtype=np.int64))):
        D = np.zeros((L - 1, L), dtype=np.int64)
        np.add.at(D, (a[first], j[first]), w)
        np.add.at(D, (a[first], j[first + 1]), -w)
        out[:] = np.cumsum(D, axis=1)[:, :L - 1]
    return h, n_odd, largest


# ----------------------------------------------------------------------------
# one chain with checkpoints
# ----------------------------------------------------------------------------
def chain_path(out_dir, L, chain):
    return os.path.join(out_dir, f"L{L}_chain{chain:02d}.pkl")


def _save(state, path):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)                      # atomic: a crash never leaves a half-written file


def run_chain(L, chain, samples, burn_in, seed=2024, out_dir=OUT_DIR,
              checkpoint_minutes=5.0, n_snapshots=3, block=500, max_seconds=None):
    """Run (or resume) one chain until it has `samples` samples. Returns a short status line."""
    os.makedirs(out_dir, exist_ok=True)
    path = chain_path(out_dir, L, chain)
    g = grid(L)
    x, y = np.tanh(J_CRIT), 1.0 / np.cosh(J_CRIT)
    c = L // 2 - 1                                              # centre face (L even)
    rng = np.random.default_rng(np.random.SeedSequence([seed, L, chain]))

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
        st = dict(L=L, chain=chain, seed=seed, burn_done=0, n_done=0, seconds=0.0,
                  tau1=taus[0], tau2=taus[1], rng=None,
                  sum_nodd=0.0, sum_nodd2=0.0, sum_h=0.0, sum_h2=0.0,
                  sum_big=0.0, sum_big2=0.0, hist_nodd=np.zeros(32, dtype=np.int64),
                  field_nodd=np.zeros((L - 1, L - 1)), cov_row=np.zeros(L - 1),
                  blocks=[], block_sum=0.0, block_n=0, snapshots=0)
    st["burn_in"], st["target"] = burn_in, samples

    t_start = t_last = time.time()

    def checkpoint(force=False):
        nonlocal t_last
        now = time.time()
        wait = 60 * checkpoint_minutes if st["seconds"] > 0 else 60   # first file after ~1 min
        if force or now - t_last > wait:
            st["rng"] = rng.bit_generator.state
            st["seconds"] += now - t_last
            t_last = now
            _save(st, path)

    def out_of_time():
        return max_seconds is not None and time.time() - t_start > max_seconds

    while st["burn_done"] < burn_in and not out_of_time():
        st["tau1"] = sw_step(st["tau1"], g, x, rng)
        st["tau2"] = sw_step(st["tau2"], g, x, rng)
        st["burn_done"] += 1
        checkpoint()

    snap_every = max(1, samples // max(1, n_snapshots))
    while st["burn_done"] >= burn_in and st["n_done"] < samples and not out_of_time():
        st["tau1"] = sw_step(st["tau1"], g, x, rng)
        st["tau2"] = sw_step(st["tau2"], g, x, rng)
        a1, b1 = current_omega(st["tau1"], g, y, rng)
        a2, b2 = current_omega(st["tau2"], g, y, rng)
        omega1 = a1 ^ a2
        support = a1 | b1 | a2 | b2
        h, n_odd, largest = nesting(omega1, support, g, rng)

        k, hc = int(n_odd[c, c]), int(h[c, c])
        st["sum_nodd"] += k
        st["sum_nodd2"] += k * k
        st["sum_h"] += hc
        st["sum_h2"] += hc * hc
        st["sum_big"] += largest
        st["sum_big2"] += largest * largest
        st["hist_nodd"][min(k, 31)] += 1
        st["field_nodd"] += n_odd
        st["cov_row"] += h[c, :] * hc
        st["block_sum"] += k
        st["block_n"] += 1
        if st["block_n"] == block:
            st["blocks"].append(st["block_sum"] / block)
            st["block_sum"], st["block_n"] = 0.0, 0

        if chain == 0 and st["snapshots"] < n_snapshots and st["n_done"] % snap_every == 0:
            np.savez_compressed(os.path.join(out_dir, f"L{L}_snapshot{st['snapshots']}.npz"),
                                L=L, omega1=np.packbits(omega1), omega2=np.packbits(support & ~omega1),
                                n_edges=len(omega1), h=h.astype(np.int8), n_odd=n_odd.astype(np.int8))
            st["snapshots"] += 1
        st["n_done"] += 1
        checkpoint()

    checkpoint(force=True)
    return f"L={L} chain {chain}: {st['n_done']}/{samples} samples, {st['seconds'] / 60:.1f} min"


# ----------------------------------------------------------------------------
# launcher, progress, results
# ----------------------------------------------------------------------------
def _task(args):
    return run_chain(*args[0], **args[1])


def run_all(config=CONFIG, workers=None, seed=2024, out_dir=OUT_DIR, **kwargs):
    """Run every chain of `config` on `workers` processes (default: all cores).
    Safe to interrupt and call again: finished chains are skipped, others resume."""
    workers = workers or os.cpu_count()
    tasks = [((L, ch, cfg["samples"], cfg["burn_in"]), dict(seed=seed, out_dir=out_dir, **kwargs))
             for L, cfg in sorted(config.items(), reverse=True) for ch in range(cfg["chains"])]
    print(f"{len(tasks)} chains on {workers} processes, output in {os.path.abspath(out_dir)}", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(_task, t) for t in tasks]):
            print(time.strftime("%H:%M:%S"), fut.result(), flush=True)


def load_chains(out_dir=OUT_DIR):
    chains = {}
    if not os.path.isdir(out_dir):
        return chains
    for name in sorted(os.listdir(out_dir)):
        if name.endswith(".pkl"):
            with open(os.path.join(out_dir, name), "rb") as f:
                st = pickle.load(f)
            chains.setdefault(st["L"], []).append(st)
    return chains


def status(out_dir=OUT_DIR):
    """Progress table from the checkpoint files (can be called while the run is going)."""
    chains = load_chains(out_dir)
    print(f"reading {out_dir}  ({sum(len(v) for v in chains.values())} checkpoint files)")
    if not chains:
        print("no checkpoints yet: the first ones appear about a minute after the run starts "
              "(larger sizes are run first, so L=1024 shows up first)")
    for L, sts in sorted(chains.items()):
        done = sum(s["n_done"] for s in sts)
        target = sum(s["target"] for s in sts)
        burning = sum(s["burn_done"] < s["burn_in"] for s in sts)
        hours = sum(s["seconds"] for s in sts) / 3600
        print(f"L={L:5d}: {len(sts):2d} chains started, {done:8d}/{target} samples "
              f"({100 * done / max(1, target):5.1f}%), {burning} still in burn-in, {hours:.2f} core-hours")


def summarize(out_dir=OUT_DIR, min_samples=100):
    """Var h(centre) = E[N_odd(centre)] and largest-cluster size per L, mean +- standard
    error over chains; slopes against ln L and a fractal-dimension fit."""
    res = {}
    for L, sts in sorted(load_chains(out_dir).items()):
        sts = [s for s in sts if s["n_done"] >= min_samples]
        if len(sts) < 2:
            continue
        var = np.array([s["sum_nodd"] / s["n_done"] for s in sts])
        var_h2 = np.array([s["sum_h2"] / s["n_done"] for s in sts])
        mean_h = np.array([s["sum_h"] / s["n_done"] for s in sts])
        big = np.array([s["sum_big"] / s["n_done"] for s in sts])
        se = lambda z: z.std(ddof=1) / np.sqrt(len(z))
        res[L] = dict(chains=len(sts), samples=sum(s["n_done"] for s in sts),
                      var=var.mean(), var_se=se(var), var_h2=var_h2.mean(), mean_h=mean_h.mean(),
                      mean_h_se=se(mean_h), big=big.mean(), big_se=se(big),
                      field=sum(s["field_nodd"] for s in sts) / sum(s["n_done"] for s in sts),
                      cov_row=sum(s["cov_row"] for s in sts) / sum(s["n_done"] for s in sts))
        r = res[L]
        print(f"L={L:5d}: {r['chains']:2d} chains, {r['samples']:8d} samples | Var h(centre) = "
              f"{r['var']:.4f} +- {r['var_se']:.4f} (from h^2: {r['var_h2']:.4f}) | E h = {r['mean_h']:+.4f} "
              f"+- {r['mean_h_se']:.4f} | largest cluster = {r['big']:.1f} +- {r['big_se']:.1f} vertices")
    if not res:
        print("not enough data yet (need >= 2 chains with >= min_samples samples for some L)")
    Ls = sorted(res)
    for a, b in zip(Ls[:-1], Ls[1:]):
        s = (res[b]["var"] - res[a]["var"]) / np.log(b / a)
        s_se = np.hypot(res[a]["var_se"], res[b]["var_se"]) / np.log(b / a)
        d = np.log(res[b]["big"] / res[a]["big"]) / np.log(b / a)
        print(f"L {a}->{b}: variance slope per ln L = {s:.4f} +- {s_se:.4f} (1/(2 pi^2) = {1 / (2 * np.pi ** 2):.4f})"
              f" | largest-cluster exponent = {d:.3f} (7/4 = 1.75)")
    return res


def load_snapshot(path):
    """Returns L, omega1, omega2 (boolean, edge order of grid(L)), h, n_odd."""
    z = np.load(path)
    n = int(z["n_edges"])
    return (int(z["L"]), np.unpackbits(z["omega1"])[:n].astype(bool),
            np.unpackbits(z["omega2"])[:n].astype(bool), z["h"].astype(int), z["n_odd"].astype(int))


def cluster_labels(L, omega1, omega2):
    """Cluster label of every vertex as an (L, L) array, and cluster sizes (in vertices)."""
    g = grid(L)
    sup = omega1 | omega2
    adj = coo_matrix((np.ones(sup.sum(), dtype=np.int8), (g["u"][sup], g["v"][sup])), shape=(L * L, L * L))
    _, lab = connected_components(adj, directed=False)
    return lab.reshape(L, L), np.bincount(lab)


def benchmark(Ls=(32, 64, 128, 256, 512, 1024), seconds=8.0, config=CONFIG, workers=None):
    """Time per sample on this machine and the projected duration of `config`."""
    workers = workers or os.cpu_count()
    x, y = np.tanh(J_CRIT), 1.0 / np.cosh(J_CRIT)
    total = 0.0
    for L in Ls:
        g = grid(L)
        rng = np.random.default_rng(L)
        t1 = np.ones(g["n_faces"], dtype=np.int8)
        t2 = t1.copy()
        for _ in range(5):
            t1, t2 = sw_step(t1, g, x, rng), sw_step(t2, g, x, rng)
        n, t0 = 0, time.time()
        while time.time() - t0 < seconds or n < 3:
            t1, t2 = sw_step(t1, g, x, rng), sw_step(t2, g, x, rng)
            a1, b1 = current_omega(t1, g, y, rng)
            a2, b2 = current_omega(t2, g, y, rng)
            nesting(a1 ^ a2, a1 | b1 | a2 | b2, g, rng)
            n += 1
        per = (time.time() - t0) / n
        line = f"L={L:5d}: {1000 * per:8.2f} ms per sample"
        if L in config:
            cfg = config[L]
            core_h = cfg["chains"] * (cfg["samples"] + 0.7 * cfg["burn_in"]) * per / 3600
            total += core_h
            line += f" | {cfg['chains']} x {cfg['samples']} samples = {core_h:6.1f} core-hours"
        print(line, flush=True)
    print(f"total {total:.1f} core-hours = about {total / workers:.1f} hours on {workers} processes "
          f"(if every process gets its own core)")
