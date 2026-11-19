"""
Theorem 1.3 on its own terms: the law of  h(f) = integral of f(x) h(x) dx  for test functions f
(continuous, compact support in D) against the law of  pi^(-1/2) * h_D(f),  h_D = Dirichlet GFF.

Domain: D = (0,1)^2, mesh 1/L.  The L x L vertices sit at ((i + 1/2)/L, (j + 1/2)/L), so the
(L-1)^2 faces are the squares of side 1/L centred at ((b+1)/L, (a+1)/L), a = row, b = column.
    h(f)  =  sum over faces  f(centre of the face) * h(face) / L^2
GFF side (exact, no sampling needed): (pi^(-1/2) h_D(f_1), ..., pi^(-1/2) h_D(f_k)) is a centred
Gaussian vector with covariance  C_ij = (1/pi) * double integral of f_i(z) G_D(z,w) f_j(w).
    gff_covariance(L, funcs)      the same for the lattice GFF at mesh 1/L (what h is compared with at finite L)
    gff_covariance(L_ref, funcs)  with a large L_ref: the continuum value
gff_values() pushes sampled GFF fields through the same sum (end-to-end check); gff_exact() draws the same law directly from N(0, C).

Library only: runs are started from custom_runs.py.  Runs are checkpointed and resumable.
"""

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.fft import dstn

from . import drc_sim as d

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "drc_testfun_runs")


# ----------------------------------------------------------------------------
# test functions
# ----------------------------------------------------------------------------
def bump(x0, y0, r):
    """Smooth bump exp(1 - 1/(1 - |z - z0|^2 / r^2)) supported in the disc of radius r around (x0, y0)."""
    def f(x, y):
        s = ((x - x0) ** 2 + (y - y0) ** 2) / r ** 2
        out = np.zeros_like(s)
        m = s < 1
        out[m] = np.exp(1.0 - 1.0 / (1.0 - s[m]))
        return out
    return f


def default_test_functions():
    b1, b2 = bump(0.35, 0.5, 0.15), bump(0.65, 0.5, 0.15)
    return {
        "bump centre r=0.25": bump(0.5, 0.5, 0.25),
        "bump centre r=0.10": bump(0.5, 0.5, 0.10),
        "bump (0.3,0.3) r=0.15": bump(0.3, 0.3, 0.15),
        "dipole (left - right)": lambda x, y: b1(x, y) - b2(x, y),
    }


def face_points(L):
    t = np.arange(1, L) / L
    return np.meshgrid(t, t)                      # X[a, b] = (b+1)/L, Y[a, b] = (a+1)/L


def weights(L, funcs):
    """W[i] = f_i(face centres) / L^2, so that h(f_i) = sum(W[i] * h)."""
    X, Y = face_points(L)
    W = np.array([f(X, Y) for f in funcs.values()], dtype=float) / L ** 2
    edge = max(np.abs(W[:, 0]).max(), np.abs(W[:, -1]).max(), np.abs(W[:, :, 0]).max(), np.abs(W[:, :, -1]).max())
    if edge > 0:
        print("warning: a test function does not vanish at the boundary of D (Theorem 1.3 needs compact support)")
    return W


# ----------------------------------------------------------------------------
# GFF side
# ----------------------------------------------------------------------------
def _eigenvalues(L):
    j = np.arange(1, L)
    c = 2 - 2 * np.cos(np.pi * j / L)
    return c[:, None] + c[None, :]


def gff_covariance(L, funcs):
    """Covariance matrix of (pi^(-1/2) GFF(f_i))_i for the lattice Dirichlet GFF at mesh 1/L (exact)."""
    W = weights(L, funcs)
    lam = _eigenvalues(L)
    What = [dstn(w, type=1, norm="ortho") for w in W]
    k = len(What)
    C = np.empty((k, k))
    for i in range(k):
        for j in range(i, k):
            C[i, j] = C[j, i] = np.sum(What[i] * What[j] / lam) / np.pi
    return C


def gff_values(L, funcs, n, seed=0):
    """h_GFF(f_i) for n sampled lattice GFF fields (times pi^(-1/2)), shape (n, k)."""
    W = weights(L, funcs).reshape(len(funcs), -1)
    lam = np.sqrt(_eigenvalues(L))
    rng = np.random.default_rng(seed)
    out = np.empty((n, len(funcs)))
    for s in range(n):
        field = dstn(rng.standard_normal((L - 1, L - 1)) / lam, type=1, norm="ortho") / np.sqrt(np.pi)
        out[s] = W @ field.ravel()
    return out


def gff_exact(L, funcs, n, seed=0):
    """n exact samples of (pi^(-1/2) GFF(f_i))_i: the vector is centred Gaussian with covariance C
    (gff_covariance), so Cholesky of C gives it at the cost of a k x k matrix: n can be 10^6 and costs nothing.
    Same law as gff_values (which pushes full lattice fields through the weights), only much cheaper."""
    C = gff_covariance(L, funcs)
    A = np.linalg.cholesky(C)
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, len(funcs))) @ A.T


# ----------------------------------------------------------------------------
# double random current side: chains with checkpoints
# ----------------------------------------------------------------------------
def _paths(out_dir, name, L):
    base = os.path.join(out_dir, f"{name}_L{L}")
    return base + "_weights.npy", base + "_names.json", base + "_chain{:02d}.pkl"


def run_chain(L, chain, samples, burn_in, name, seed=4000, out_dir=OUT_DIR, checkpoint_minutes=5.0,
              max_seconds=None):
    wpath, _, cpath = _paths(out_dir, name, L)
    path = cpath.format(chain)
    W = np.load(wpath)
    Wf = W.reshape(len(W), -1)
    g = d.grid(L)
    x, y = np.tanh(d.J_CRIT), 1.0 / np.cosh(d.J_CRIT)
    rng = np.random.default_rng(np.random.SeedSequence([seed, L, chain, 2]))
    if os.path.exists(path):
        with open(path, "rb") as f:
            st = pickle.load(f)
        rng.bit_generator.state = st["rng"]
        if st["values"].shape[0] < samples:
            st["values"] = np.vstack([st["values"], np.zeros((samples - st["values"].shape[0], len(W)))])
    else:
        taus = []
        for _ in range(2):
            t = (rng.integers(0, 2, size=g["n_faces"], dtype=np.int8) * 2 - 1).astype(np.int8)
            t[0] = 1
            taus.append(t)
        st = dict(L=L, chain=chain, burn_done=0, n_done=0, seconds=0.0, tau1=taus[0], tau2=taus[1],
                  rng=None, values=np.zeros((samples, len(W))))
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
        h, _, _ = d.nesting(a1 ^ a2, a1 | b1 | a2 | b2, g, rng)
        st["values"][st["n_done"]] = Wf @ h.ravel().astype(float)
        st["n_done"] += 1
        checkpoint()
    checkpoint(force=True)
    return f"{name} L={L} chain {chain}: {st['n_done']}/{samples} samples, {st['seconds'] / 60:.1f} min"


def _task(args):
    return run_chain(*args[0], **args[1])


def run(L, funcs=None, chains=6, samples=5000, burn_in=500, workers=None, name="default", seed=4000,
        out_dir=OUT_DIR, **kwargs):
    """Sample h(f_i) for every test function, on `chains` independent chains.  Re-run to resume or,
    with a larger `samples`, to extend.  Use a new `name` for a different set of test functions."""
    funcs = funcs or default_test_functions()
    os.makedirs(out_dir, exist_ok=True)
    wpath, npath, _ = _paths(out_dir, name, L)
    W = weights(L, funcs)
    if os.path.exists(wpath):
        if not np.array_equal(np.load(wpath), W):
            raise ValueError(f"run '{name}' at L={L} was started with other test functions; choose another name")
    else:
        np.save(wpath, W)
        with open(npath, "w") as f:
            json.dump(list(funcs), f)
    workers = workers or os.cpu_count()
    tasks = [((L, ch, samples, burn_in, name), dict(seed=seed, out_dir=out_dir, **kwargs)) for ch in range(chains)]
    print(f"{len(tasks)} chains on {workers} processes, output in {out_dir}", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(_task, t) for t in tasks]):
            print(time.strftime("%H:%M:%S"), fut.result(), flush=True)


def load(L, name="default", out_dir=OUT_DIR):
    """names, and the list (one entry per chain) of arrays h(f) of shape (samples done, k)."""
    _, npath, cpath = _paths(out_dir, name, L)
    with open(npath) as f:
        names = json.load(f)
    vals, ch = [], 0
    while os.path.exists(cpath.format(ch)):
        with open(cpath.format(ch), "rb") as f:
            st = pickle.load(f)
        vals.append(st["values"][:st["n_done"]])
        ch += 1
    return names, vals


# ----------------------------------------------------------------------------
# comparison
# ----------------------------------------------------------------------------
def _tau_int(z):
    """Integrated autocorrelation time (in samples) with Sokal's automatic window."""
    z = z - z.mean()
    n = len(z)
    f = np.fft.rfft(z, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n] / np.arange(n, 0, -1)
    rho = acf / acf[0]
    tau = 0.5
    for t in range(1, n // 4):
        tau += rho[t]
        if t >= 5 * tau:
            break
    return max(tau, 0.5)


def _mean_se(z):
    z = np.asarray(z, dtype=float)
    return z.mean(axis=0), z.std(axis=0, ddof=1) / np.sqrt(len(z))


def compare(L, name="default", funcs=None, L_ref=2048, out_dir=OUT_DIR):
    """Prints, for every test function: Var h(f) against the GFF variance (lattice at this L and
    continuum), skewness, excess kurtosis (GFF: 0), autocorrelation time, a Kolmogorov-Smirnov
    distance to the Gaussian; then the correlations between test functions.  Returns a dict."""
    from scipy.stats import kstest, kurtosis, skew
    funcs = funcs or default_test_functions()
    names, vals = load(L, name, out_dir)
    assert names == list(funcs), "pass the same test functions that were used in run()"
    vals = [v for v in vals if len(v) >= 50]          # chains that have not produced samples yet are skipped
    if len(vals) < 2:
        raise ValueError(f"L = {L}: fewer than 2 chains with 50 samples, nothing to compare yet")
    C = gff_covariance(L, funcs)
    C_ref = gff_covariance(L_ref, funcs) if L_ref else C
    n = sum(len(v) for v in vals)
    print(f"L = {L}: {len(vals)} chains, {n} samples.   Errors: standard error over chains.")
    out = dict(names=names, C=C, C_ref=C_ref, values=vals)
    for i, nm in enumerate(names):
        var, var_se = _mean_se([v[:, i].var() for v in vals])
        ku, ku_se = _mean_se([kurtosis(v[:, i]) for v in vals])
        sk, sk_se = _mean_se([skew(v[:, i]) for v in vals])
        tau_h = np.mean([_tau_int(v[:, i]) for v in vals])
        tau_v = np.mean([_tau_int(v[:, i] ** 2) for v in vals])      # h(f)^2 carries the correlation between samples
        tau = max(tau_h, tau_v)
        thin = int(np.ceil(2 * tau))
        z = np.concatenate([v[::thin, i] for v in vals]) / np.sqrt(C[i, i])
        ks = kstest(z, "norm")
        print(f"\n{nm}")
        print(f"   Var h(f)            = {var:.4e} +- {var_se:.1e}")
        print(f"   Var GFF(f) / pi     = {C[i, i]:.4e} (lattice, mesh 1/{L})    {C_ref[i, i]:.4e} (continuum, mesh 1/{L_ref})")
        print(f"   ratio h / GFF       = {var / C[i, i]:.4f} +- {var_se / C[i, i]:.4f} (lattice)    "
              f"{var / C_ref[i, i]:.4f} (continuum)")
        print(f"   skewness            = {sk:+.3f} +- {sk_se:.3f}     excess kurtosis = {ku:+.3f} +- {ku_se:.3f}   (GFF: 0, 0)")
        print(f"   autocorrelation time = {tau_h:.1f} samples for h(f), {tau_v:.1f} for h(f)^2 (the larger one sets the thinning)")
        print(f"   KS distance to N(0, Var GFF), thinning {thin} ({len(z)} values): D = {ks.statistic:.4f}, p = {ks.pvalue:.2f}")
    if len(names) > 1:
        emp = np.mean([np.corrcoef(v.T) for v in vals], axis=0)
        emp_se = np.std([np.corrcoef(v.T) for v in vals], axis=0, ddof=1) / np.sqrt(len(vals))
        sd = np.sqrt(np.diag(C_ref))
        th = C_ref / np.outer(sd, sd)
        print("\ncorrelations between test functions   (h: mean +- se over chains | GFF continuum)")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                print(f"   {names[i]}  x  {names[j]}:  {emp[i, j]:+.4f} +- {emp_se[i, j]:.4f} | {th[i, j]:+.4f}")
    return out


def figure(L, name="default", funcs=None, n_gff=None, n_fields=0, out_dir=OUT_DIR):
    """Row 1: the test functions.  Row 2: histogram of h(f) (blue) with the exact Gaussian law of
    pi^(-1/2) GFF(f) (line) and exact GFF samples (orange; n_gff of them, default = as many as h samples;
    n_fields > 0 adds a small end-to-end check with full lattice GFF fields, green).  Row 3: quantile-quantile plot."""
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    blue, orange, ink, ink2 = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
    funcs = funcs or default_test_functions()
    names, vals = load(L, name, out_dir)
    vals = [v for v in vals if len(v) >= 50]
    if not vals:
        raise ValueError(f"L = {L}: no chain has 50 samples yet")
    C = gff_covariance(L, funcs)
    n_h = sum(len(v) for v in vals)
    n_gff = n_h if n_gff is None else n_gff
    gv = gff_exact(L, funcs, n_gff, seed=1) if n_gff else None
    gf = gff_values(L, funcs, n_fields, seed=2) if n_fields else None
    X, Y = face_points(min(L, 256))
    k = len(names)
    fig, axes = plt.subplots(3, k, figsize=(4.2 * k, 11.5), squeeze=False)
    for i, nm in enumerate(names):
        F = list(funcs.values())[i](X, Y)
        m = np.abs(F).max()
        axes[0, i].imshow(F, cmap="RdBu", vmin=-m, vmax=m, extent=(0, 1, 1, 0))
        axes[0, i].set_title(nm, fontsize=10)
        axes[0, i].set_xticks([0, 0.5, 1])
        axes[0, i].set_yticks([0, 0.5, 1])
        z = np.concatenate([v[:, i] for v in vals])
        s = np.sqrt(C[i, i])
        lim = 1.05 * max(np.abs(z).max(), 4 * s)
        bins = np.linspace(-lim, lim, 61)
        ax = axes[1, i]
        ax.hist(z, bins=bins, density=True, color=blue, alpha=0.85, label=f"h(f), {len(z)} samples")
        if gv is not None:
            ax.hist(gv[:, i], bins=bins, density=True, histtype="step", lw=1.8, color=orange,
                    label=f"exact GFF law, {n_gff} samples")
        if gf is not None:
            ax.hist(gf[:, i], bins=bins, density=True, histtype="step", lw=1.2, color="#2b9a4a",
                    label=f"GFF lattice fields, {n_fields} samples")
        xs = np.linspace(-lim, lim, 400)
        ax.plot(xs, norm.pdf(xs, scale=s), color=ink, lw=1.5, label="exact law of π^(-1/2) GFF(f)")
        ax.set_yscale("log")
        ax.set_ylim(0.3 / (len(z) * (bins[1] - bins[0])), 3 * norm.pdf(0, scale=s))
        ax.set_xlabel("value of the pairing with f")
        ax.locator_params(axis="x", nbins=5)
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
        ax = axes[2, i]
        q = (np.arange(1, len(z) + 1) - 0.5) / len(z)
        ax.plot(norm.ppf(q) * s, np.sort(z), color=blue, lw=2)
        ax.plot([-lim, lim], [-lim, lim], color=ink2, lw=1, ls="--")
        ax.set_xlabel("quantiles of π^(-1/2) GFF(f)")
        ax.set_ylabel("quantiles of h(f)" if i == 0 else "")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.locator_params(nbins=5)
    fig.suptitle(f"L = {L}", color=ink2)
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


# ============================================================================
# batch runner, progress and time reports, precision diagnostics
# ============================================================================
import glob
import re
import threading

NAME = "main"
# Standard run, per L: chains x samples per chain (12 x 8334 = 100,008 samples) and burn-in sweeps.
# Extensions (more samples on the same chains) are listed in custom_runs.py, not here.
CONFIG = {
    256: dict(chains=12, samples=8334, burn_in=500),
    512: dict(chains=12, samples=8334, burn_in=500),
    1024: dict(chains=12, samples=8334, burn_in=500),
}


def _hms(seconds):
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.2f} h"


def _timing_path(out_dir, name):
    return os.path.join(out_dir, f"{name}_timing.json")


def _read_timing(out_dir, name):
    try:
        with open(_timing_path(out_dir, name)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return []


def _write_timing(out_dir, name, log):
    path = _timing_path(out_dir, name)
    with open(path + ".tmp", "w") as f:
        json.dump(log, f, indent=1)
    os.replace(path + ".tmp", path)


def _start_log(out_dir, name, L, workers, c):
    log = _read_timing(out_dir, name)
    log.append(dict(L=L, wall_s=0.0, workers=workers, chains=c["chains"], samples=c["samples"],
                    when=time.strftime("%Y-%m-%d %H:%M:%S")))
    _write_timing(out_dir, name, log)
    return len(log) - 1


def _update_log(out_dir, name, idx, wall):
    log = _read_timing(out_dir, name)
    if idx < len(log):
        log[idx]["wall_s"] = round(wall, 1)
        _write_timing(out_dir, name, log)


def available_Ls(name=NAME, out_dir=OUT_DIR):
    """Sizes L of the run `name` that have at least one chain checkpoint."""
    Ls = []
    for p in glob.glob(os.path.join(out_dir, f"{name}_L*_weights.npy")):
        m = re.search(rf"{re.escape(name)}_L(\d+)_weights\.npy$", os.path.basename(p))
        if m and os.path.exists(_paths(out_dir, name, int(m.group(1)))[2].format(0)):
            Ls.append(int(m.group(1)))
    return sorted(Ls)


def _states(L, name, out_dir):
    _, _, cpath = _paths(out_dir, name, L)
    sts, ch = [], 0
    while os.path.exists(cpath.format(ch)):
        with open(cpath.format(ch), "rb") as f:
            sts.append(pickle.load(f))
        ch += 1
    return sts


def run_all(config=None, funcs=None, name=NAME, workers=None, seed=4000, out_dir=OUT_DIR, **kwargs):
    """Runs every size of `config` (default CONFIG), smallest L first, each on a pool of `workers` processes
    (default: all cores).  Safe to interrupt and start again (chains resume); with a larger `samples` the chains
    are extended.  The wall-clock time of every size is logged to <name>_timing.json (see time_report)."""
    config = config or CONFIG
    funcs = funcs or default_test_functions()
    os.makedirs(out_dir, exist_ok=True)
    workers = workers or os.cpu_count()
    t_all = time.time()
    for L in sorted(config):
        c = config[L]
        print(f"\n=== L = {L}: {c['chains']} chains x {c['samples']} samples "
              f"({c['chains'] * c['samples']:,} in total), burn-in {c['burn_in']} ===", flush=True)
        t0 = time.time()
        idx = _start_log(out_dir, name, L, workers, c)
        stop = threading.Event()

        def tick():                      # the wall-clock entry is refreshed every minute, so it survives a closed window
            while not stop.wait(60):
                _update_log(out_dir, name, idx, time.time() - t0)

        thread = threading.Thread(target=tick, daemon=True)
        thread.start()
        try:
            run(L, funcs, chains=c["chains"], samples=c["samples"], burn_in=c["burn_in"], workers=workers,
                name=name, seed=seed, out_dir=out_dir, **kwargs)
        finally:
            stop.set()
            thread.join()
            _update_log(out_dir, name, idx, time.time() - t0)
        print(f"L = {L} finished in {_hms(time.time() - t0)}", flush=True)
    print(f"\nAll sizes finished in {_hms(time.time() - t_all)}.\n", flush=True)
    time_report(name, out_dir)


def status(name=NAME, out_dir=OUT_DIR, workers=None):
    """Progress of the run: samples done, chains still in burn-in, core time used and an estimate of the time left.
    Can be called at any moment, also while the run is going."""
    workers = workers or os.cpu_count()
    Ls = available_Ls(name, out_dir)
    if not Ls:
        print(f"No checkpoint of run '{name}' in {out_dir} yet.  The first one appears about one minute after "
              f"the start; if nothing shows up, check the prompt that runs custom_runs.py.")
        return
    print(f"run '{name}'  ({time.strftime('%H:%M:%S')})   estimates assume {workers} processes")
    print(f"{'L':>5} {'chains':>7} {'samples done / target':>24} {'%':>6} {'in burn-in':>11} {'core time':>10} {'time left':>10}")
    left_all = 0.0
    for L in Ls:
        sts = _states(L, name, out_dir)
        done = sum(s["n_done"] for s in sts)
        target = sum(s["target"] for s in sts)
        burning = sum(s["burn_done"] < s["burn_in"] for s in sts)
        core = sum(s["seconds"] for s in sts)
        units = done + 0.7 * sum(min(s["burn_done"], s["burn_in"]) for s in sts)     # a burn-in sweep costs ~0.7 of a sample
        per = core / units if units else float("nan")
        expected = CONFIG[L]["chains"] if L in CONFIG else len(sts)
        missing = max(expected - len(sts), 0)
        per_chain = sts[0]["target"] + 0.7 * sts[0]["burn_in"]
        left_units = (target - done) + 0.7 * sum(s["burn_in"] - min(s["burn_done"], s["burn_in"]) for s in sts) \
            + missing * per_chain
        left = per * left_units / workers
        left_all += 0 if np.isnan(left) else left
        target_tot = target + missing * sts[0]["target"]
        print(f"{L:5d} {len(sts):3d}/{expected:<3d} {done:>13,d} / {target_tot:<9,d} {100 * done / target_tot:6.1f} "
              f"{burning:11d} {_hms(core):>10} {_hms(left):>10}")
    print(f"\nestimated time left for everything started or expected: {_hms(left_all)} "
          f"(a size that has not started yet is not counted unless it appears in CONFIG)")


def time_report(name=NAME, out_dir=OUT_DIR, workers=None):
    """Time used by every size and in total.  core time = sum over chains of their own running time (read from the
    checkpoints, exact, adds up over resumes).  wall-clock = measured time of the run_all calls, summed over all
    sessions (the real time you waited).  Returns a dict."""
    Ls = available_Ls(name, out_dir)
    if not Ls:
        print(f"No checkpoint of run '{name}' in {out_dir} yet.")
        return {}
    log = _read_timing(out_dir, name)
    out, tot_core, tot_wall, tot_samples = {}, 0.0, 0.0, 0
    print(f"time used by run '{name}'")
    print(f"{'L':>5} {'samples':>10} {'core time':>11} {'wall-clock':>12} {'s / sample / core':>19}")
    for L in Ls:
        sts = _states(L, name, out_dir)
        done = sum(s["n_done"] for s in sts)
        core = sum(s["seconds"] for s in sts)
        walls = [e["wall_s"] for e in log if e["L"] == L]
        wall = sum(walls) if walls else None
        per = core / done if done else float("nan")
        tot_core += core
        tot_samples += done
        tot_wall += wall or 0.0
        out[L] = dict(samples=done, core_s=core, wall_s=wall, s_per_sample_core=per)
        print(f"{L:5d} {done:10,d} {_hms(core):>11} {(_hms(wall) if wall is not None else 'not logged'):>12} {per:19.4f}")
    print(f"{'TOTAL':>5} {tot_samples:10,d} {_hms(tot_core):>11} {_hms(tot_wall):>12}")
    print("(core time includes burn-in; wall-clock counts only runs started with run_all; if the window was closed during a run, up to one minute of that session is missing)")
    out["total"] = dict(samples=tot_samples, core_s=tot_core, wall_s=tot_wall)
    return out


# ----------------------------------------------------------------------------
# how well do the data settle the question?  precision against number of samples, trend with L
# ----------------------------------------------------------------------------
def _prefix_moments(v, ns):
    """Variance and excess kurtosis of the first n values of v, for every n in ns (power sums)."""
    c = [np.cumsum(v ** p) for p in (1, 2, 3, 4)]
    var, ku = [], []
    for n in ns:
        m, m2, m3, m4 = (c[p][n - 1] / n for p in range(4))
        mu2 = m2 - m * m
        mu4 = m4 - 4 * m * m3 + 6 * m * m * m2 - 3 * m ** 4
        var.append(mu2)
        ku.append(mu4 / mu2 ** 2 - 3)
    return np.array(var), np.array(ku)


def running_estimates(L, name=NAME, funcs=None, out_dir=OUT_DIR, n_points=25, n_min=200):
    """Variance ratio h/GFF and excess kurtosis of h(f), using only the first n samples of every chain, for n growing
    from n_min to the shortest chain.  Errors: standard error over chains."""
    funcs = funcs or default_test_functions()
    names, vals = load(L, name, out_dir)
    assert names == list(funcs), "pass the same test functions that were used in run()"
    vals = [v for v in vals if len(v) >= n_min]
    if len(vals) < 2:
        raise ValueError(f"L = {L}: fewer than 2 chains with {n_min} samples")
    nmax = min(len(v) for v in vals)
    ns = np.unique(np.round(np.geomspace(n_min, nmax, n_points)).astype(int))
    C = gff_covariance(L, funcs)
    k, nc = len(names), len(vals)
    ratio, kurt = np.zeros((nc, len(ns), k)), np.zeros((nc, len(ns), k))
    for c, v in enumerate(vals):
        for i in range(k):
            var, ku = _prefix_moments(v[:nmax, i], ns)
            ratio[c, :, i], kurt[c, :, i] = var / C[i, i], ku
    se = lambda a: a.std(axis=0, ddof=1) / np.sqrt(nc)
    return dict(names=names, N=ns * nc, chains=nc, ratio=ratio.mean(0), ratio_se=se(ratio), kurt=kurt.mean(0),
                kurt_se=se(kurt))


def plot_running(L, name=NAME, funcs=None, out_dir=OUT_DIR):
    """Variance ratio and excess kurtosis against the total number of samples: do the estimates still move, and do
    the error bands still shrink like 1/sqrt(N)?"""
    import matplotlib.pyplot as plt
    cols = ["#2a78d6", "#eb6834", "#2f9e6e", "#8b5cc2"]
    r = running_estimates(L, name, funcs, out_dir)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    for i, nm in enumerate(r["names"]):
        col = cols[i % len(cols)]
        for a, key, ref in ((ax[0], "ratio", 1.0), (ax[1], "kurt", 0.0)):
            a.plot(r["N"], r[key][:, i], color=col, lw=1.8, label=nm)
            a.fill_between(r["N"], r[key][:, i] - r[key + "_se"][:, i], r[key][:, i] + r[key + "_se"][:, i],
                           color=col, alpha=0.15, lw=0)
    ax[0].axhline(1, color="#0b0b0b", lw=1, ls="--")
    ax[1].axhline(0, color="#0b0b0b", lw=1, ls="--")
    ax[0].set_ylabel("Var h(f) / Var GFF(f)  (lattice)")
    ax[1].set_ylabel("excess kurtosis of h(f)")
    for a in ax:
        a.set_xscale("log")
        a.set_xlabel("total number of samples used")
    ax[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"L = {L}, {r['chains']} chains: bands are ± 1 standard error over chains", color="#52514e")
    fig.tight_layout()
    return fig


def trend(name=NAME, funcs=None, out_dir=OUT_DIR, min_samples=200):
    """Variance ratio, skewness and excess kurtosis of h(f) for every size L found (all samples, standard error over
    chains).  Returns dict L -> dict."""
    funcs = funcs or default_test_functions()
    from scipy.stats import kurtosis, skew
    rows = {}
    for L in available_Ls(name, out_dir):
        names, vals = load(L, name, out_dir)
        vals = [v for v in vals if len(v) >= min_samples]
        if len(vals) < 2 or names != list(funcs):
            continue
        C = gff_covariance(L, funcs)
        k = len(names)
        ratio = np.array([[v[:, i].var() / C[i, i] for i in range(k)] for v in vals])
        sk = np.array([[skew(v[:, i]) for i in range(k)] for v in vals])
        ku = np.array([[kurtosis(v[:, i]) for i in range(k)] for v in vals])
        se = lambda a: a.std(axis=0, ddof=1) / np.sqrt(len(a))
        rows[L] = dict(names=names, samples=sum(len(v) for v in vals), chains=len(vals), ratio=ratio.mean(0),
                       ratio_se=se(ratio), skew=sk.mean(0), skew_se=se(sk), kurt=ku.mean(0), kurt_se=se(ku))
    return rows


def trend_table(rows, project_to=None):
    """Prints the trend.  `dev` = (ratio - 1) / standard error: a deviation that stays large while the error shrinks
    is a finite-L effect, which more samples will not remove.  With project_to=N, shows the standard error expected
    with N samples in total."""
    if not rows:
        print("no size with at least 2 chains and enough samples yet")
        return
    names = next(iter(rows.values()))["names"]
    for i, nm in enumerate(names):
        print(f"\n{nm}")
        head = f"{'L':>6} {'samples':>9} {'ratio h/GFF':>16} {'dev':>6} {'skewness':>14} {'excess kurtosis':>17}"
        print(head + (f" {'se(ratio) at ' + format(project_to, ','):>22}" if project_to else ""))
        for L, r in rows.items():
            line = (f"{L:6d} {r['samples']:9,d} {r['ratio'][i]:9.4f} ± {r['ratio_se'][i]:.4f} "
                    f"{(r['ratio'][i] - 1) / r['ratio_se'][i]:6.1f} {r['skew'][i]:+8.3f} ± {r['skew_se'][i]:.3f} "
                    f"{r['kurt'][i]:+9.3f} ± {r['kurt_se'][i]:.3f}")
            if project_to:
                line += f" {r['ratio_se'][i] * np.sqrt(r['samples'] / project_to):22.4f}"
            print(line)


def plot_trend(rows):
    """Variance ratio, skewness and excess kurtosis of h(f) against L (GFF values: 1, 0, 0)."""
    import matplotlib.pyplot as plt
    cols = ["#2a78d6", "#eb6834", "#2f9e6e", "#8b5cc2"]
    names = next(iter(rows.values()))["names"]
    Ls = list(rows)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    for i, nm in enumerate(names):
        for a, key, off in zip(ax, ("ratio", "skew", "kurt"), (0, 0, 0)):
            a.errorbar(Ls, [rows[L][key][i] for L in Ls], yerr=[rows[L][key + "_se"][i] for L in Ls],
                       color=cols[i % len(cols)], marker="o", ms=5, lw=1.6, capsize=3, label=nm)
    for a, ref, lab in zip(ax, (1, 0, 0), ("Var h(f) / Var GFF(f)", "skewness", "excess kurtosis")):
        a.axhline(ref, color="#0b0b0b", lw=1, ls="--")
        a.set_xscale("log", base=2)
        a.set_xticks(Ls)
        a.set_xticklabels(Ls)
        a.set_xlabel("L")
        a.set_ylabel(lab)
    ax[0].legend(frameon=False, fontsize=8)
    fig.suptitle("h(f) against the GFF: bars are ± 1 standard error over chains, dashed line = GFF value",
                 color="#52514e")
    fig.tight_layout()
    return fig
