"""
Master-coupling height function H of Duminil-Copin--Lis--Qian (Theorem 2.1), built from the
nesting field h of a sample of the critical double random current.

Conventions (identical to drc_sim.py)
  * L x L vertices, (L-1) x (L-1) inner faces + 1 outer face (index 0, h = 0, tau_dag = +1).
  * omega1 = odd edges of the double current, omega2 = even non-zero edges, edge order of grid(L).
  * h[a, b] = nesting field on the inner face whose top-left vertex is (a, b).

What the theorem says (Theorem 2.1 + eq. (2.5), (2.6) of the paper)
  H lives on faces U and vertices V,  H(u) - H(v) = (1/2) tau_dag_u tau_v  for every incident (u, v),
  tau_dag_u = (-1)^H(u) is the spin whose interfaces are the odd edges,  tau_v is constant on the
  clusters of n and the cluster spins are i.i.d. fair signs.  H|faces is a nesting field of n with
  labels eps_C = tau_C tau_dag_C, H|vertices has the law of the *wired* nesting field of the dual current.

How H is computed here (no dual current needed)
  1. tau_dag from the parity of h (checked against the odd edges).
  2. On an odd edge e with faces u, u' the relation forces  tau_v = tau_dag_u (h(u) - h(u'))  for both
     end points, so every cluster that contains an odd edge gets its spin from h (checked: all odd edges of
     a cluster agree).  Clusters without odd edge get a fresh fair sign (they carry no information in h).
  3. H(v) = h(u) - (1/2) tau_dag_u tau_v, identical for the four faces around v (checked).
With the cluster spins fixed this way tau_C is a deterministic function of (n, eps), eps_C = -tau_C tau_dag_C
(overall sign is a convention, eps -> -eps is a symmetry), so H|faces = h exactly, and the pair (n, tau) has the
law required by the theorem.  What is only true *in law* is that H|vertices is the wired nesting field of
the dual current n_dag (which this module does not sample).
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from . import drc_sim as ds


def height_H(L, omega1, omega2, h, rng=None, check=True):
    """Returns dict(H_faces (L-1,L-1), H_vertices (L,L), tau_dag (L-1,L-1), tau (L,L), labels (L,L))."""
    g = ds.grid(L)
    rng = np.random.default_rng() if rng is None else rng
    h = np.asarray(h, dtype=np.int64)

    hf = np.r_[0, h.ravel()]                              # face-indexed, face 0 = outer
    td = np.where(hf % 2 == 0, 1, -1)                     # tau_dag = (-1)^h
    if check:
        assert np.array_equal(td[g["f1"]] != td[g["f2"]], omega1), "parity of h does not match the odd edges"

    sup = omega1 | omega2
    adj = coo_matrix((np.ones(sup.sum(), dtype=np.int8), (g["u"][sup], g["v"][sup])), shape=(L * L, L * L))
    n_comp, lab = connected_components(adj, directed=False)

    # cluster spins from the odd edges
    odd = np.flatnonzero(omega1)
    f1, f2 = g["f1"][odd], g["f2"][odd]
    te = td[f1] * (hf[f1] - hf[f2])                       # must be +-1
    spin = np.zeros(n_comp, dtype=np.int64)
    if len(odd):
        assert np.all(np.abs(te) == 1), "h jumps by something else than +-1 across an odd edge"
        cl = lab[g["u"][odd]]
        spin[cl] = te                                     # later assignments overwrite; checked below
        if check:
            assert np.array_equal(spin[cl], te), "odd edges of one cluster disagree on tau_C"
    free = spin == 0                                       # clusters with no odd edge
    spin[free] = rng.integers(0, 2, size=int(free.sum())) * 2 - 1
    tau = spin[lab].reshape(L, L)

    # H on vertices from the four surrounding faces
    hp = np.zeros((L + 1, L + 1))
    hp[1:L, 1:L] = h
    tp = np.ones((L + 1, L + 1), dtype=np.int64)
    tp[1:L, 1:L] = td[1:].reshape(L - 1, L - 1)
    cand = [hp[i:i + L, j:j + L] - 0.5 * tp[i:i + L, j:j + L] * tau for i in (0, 1) for j in (0, 1)]
    if check:
        for k in (1, 2, 3):
            assert np.array_equal(cand[0], cand[k]), "the four faces around a vertex disagree on H(v)"
    return dict(H_faces=h.astype(float), H_vertices=cand[0], tau_dag=td[1:].reshape(L - 1, L - 1),
                tau=tau, labels=lab.reshape(L, L))


def sample_stream(L, n, seed=0, burn_in=200, thin=1):
    """Yield n samples (omega1, omega2, h) of the critical double current on the L x L grid
    (same chain as drc_sim.run_chain: two Swendsen-Wang face chains, outer face fixed)."""
    g = ds.grid(L)
    x, y = np.tanh(ds.J_CRIT), 1.0 / np.cosh(ds.J_CRIT)
    rng = np.random.default_rng(np.random.SeedSequence([9100, L, seed]))
    taus = []
    for _ in range(2):
        t = (rng.integers(0, 2, size=g["n_faces"], dtype=np.int8) * 2 - 1).astype(np.int8)
        t[0] = 1
        taus.append(t)
    t1, t2 = taus
    for _ in range(burn_in):
        t1, t2 = ds.sw_step(t1, g, x, rng), ds.sw_step(t2, g, x, rng)
    for _ in range(n):
        for _ in range(thin):
            t1, t2 = ds.sw_step(t1, g, x, rng), ds.sw_step(t2, g, x, rng)
        a1, b1 = ds.current_omega(t1, g, y, rng)
        a2, b2 = ds.current_omega(t2, g, y, rng)
        omega1 = a1 ^ a2
        support = a1 | b1 | a2 | b2
        h, _, _ = ds.nesting(omega1, support, g, rng)
        yield omega1, support & ~omega1, h, rng


def _levels(res):
    m = max(np.abs(res["H_faces"]).max(), np.abs(res["H_vertices"]).max())
    m = np.ceil(m * 2) / 2
    return np.arange(-m - 0.25, m + 0.5, 0.5)


def plot_H(res, path=None, window=None, title=None):
    """Three panels: H on faces, H on vertices, and both together on a window of the lattice
    (faces = squares, vertices = discs, same colours).  window = (row0, col0, size) in vertex units."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm

    Hf, Hv = res["H_faces"], res["H_vertices"]
    L = Hv.shape[0]
    lev = _levels(res)
    cmap = plt.get_cmap("RdBu_r", len(lev) - 1)
    norm = BoundaryNorm(lev, cmap.N)
    if window is None:
        s = min(24, L - 1)
        window = (L // 2 - s // 2, L // 2 - s // 2, s)
    r0, c0, s = window

    fig = plt.figure(figsize=(15, 5.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 3)
    for k, (arr, name) in enumerate(((Hf, "H on faces (= h)"), (Hv, "H on vertices"))):
        ax = fig.add_subplot(gs[0, k])
        off = 0.5 if k == 0 else 0.0
        n = arr.shape[0]
        im = ax.imshow(arr, cmap=cmap, norm=norm, interpolation="nearest",
                       extent=(off - 0.5, off + n - 0.5, off + n - 0.5, off - 0.5))
        ax.add_patch(plt.Rectangle((c0 - 0.5 + off, r0 - 0.5 + off), s, s, fill=False, ec="k", lw=1))
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
    ax = fig.add_subplot(gs[0, 2])
    fw = Hf[r0:r0 + s, c0:c0 + s]
    ax.pcolormesh(np.arange(c0, c0 + s + 1) , np.arange(r0, r0 + s + 1), fw, cmap=cmap, norm=norm,
                  edgecolors="none")
    rr, cc = np.mgrid[r0:r0 + s + 1, c0:c0 + s + 1]
    sel = (rr < L) & (cc < L)
    sc = ax.scatter(cc[sel], rr[sel], c=Hv[rr[sel], cc[sel]], cmap=cmap, norm=norm, s=max(8, 900 / s),
                    edgecolors="k", linewidths=0.4, zorder=3)
    ax.set_xlim(c0 - 0.5, c0 + s + 0.5)
    ax.set_ylim(r0 + s + 0.5, r0 - 0.5)
    ax.set_aspect("equal")
    ax.set_title("window: faces (squares) + vertices (discs)")
    ax.set_xticks([])
    ax.set_yticks([])
    cb = fig.colorbar(sc, ax=fig.axes, shrink=0.8, ticks=lev[:-1] + 0.25)
    cb.set_label("H")
    if title:
        fig.suptitle(title)
    if path:
        fig.savefig(path, dpi=140)
        plt.close(fig)
    return fig


def variance_profile(L, n=2000, seed=0, burn_in=200, thin=1):
    """Var of H(face) and H(vertex) at the centre of the L x L grid (mean ~ 0 by symmetry)."""
    c = L // 2
    fa, ve = [], []
    for o1, o2, h, rng in sample_stream(L, n, seed, burn_in, thin):
        r = height_H(L, o1, o2, h, rng)
        fa.append(r["H_faces"][c - 1, c - 1])
        ve.append(r["H_vertices"][c, c])
    fa, ve = np.array(fa), np.array(ve)
    return fa.var(), ve.var(), fa.mean(), ve.mean(), np.corrcoef(fa, ve)[0, 1]
