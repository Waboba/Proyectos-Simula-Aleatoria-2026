"""
One sample of the critical double random current (sourceless, free boundary) on the L x L grid, with the
pictures that go with it.  Library only: the notebook sample_drc.ipynb calls these functions.

    case = sample(L, seed=0)                 # draw one sample; everything below takes `case`
    summary(case)                            # counts of odd / even / zero edges, clusters, h and H ranges
    plot_case(case, view="tau")              # face signs tau_dag and the edges      (small L or a window)
    plot_case(case, view="h")                # nesting field h with its values
    plot_case(case, view="H")                # H on faces (squares) and on vertices (discs), with the edges
    decode_table(case)                       # what H alone tells about each edge (compare with the truth)
    integer_values(case)                     # integer values n_e given the odd / even / zero classes
    save_edges_csv(case, path)               # one row per edge

Edge classes of the double current n:  odd (n_e odd, the set omega_1),  even (n_e even and > 0, omega_2),
zero (n_e = 0).  Faces: tau_dag = (-1)^h is the sign whose interfaces are the odd edges (outer face +1).
"""
import csv

import numpy as np

from . import drc_sim as ds
from . import drc_height as dh

INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def sample(L, seed=0, burn_in=500, beta_ratio=1.0, values=False):
    """One sample.  beta_ratio = beta / beta_c (1 = critical).  Returns a dict with
    L, seed, beta_ratio, omega1 (odd edges), omega2 (even non-zero edges), h, tau_dag, H (see drc_height.height_H)
    and, with values=True, the integer values n (see integer_values)."""
    g = ds.grid(L)
    J = ds.J_CRIT * beta_ratio
    x, y = np.tanh(J), 1.0 / np.cosh(J)
    rng = np.random.default_rng(np.random.SeedSequence([9300, L, seed]))
    t = []
    for _ in range(2):
        s = (rng.integers(0, 2, size=g["n_faces"], dtype=np.int8) * 2 - 1).astype(np.int8)
        s[0] = 1
        t.append(s)
    for _ in range(burn_in + 1):
        t = [ds.sw_step(s, g, x, rng) for s in t]
    a1, b1 = ds.current_omega(t[0], g, y, rng)
    a2, b2 = ds.current_omega(t[1], g, y, rng)
    omega1 = a1 ^ a2
    support = a1 | b1 | a2 | b2
    h, _, _ = ds.nesting(omega1, support, g, rng)
    case = dict(L=L, seed=seed, beta_ratio=beta_ratio, J=J, omega1=omega1, omega2=support & ~omega1, h=h)
    H = dh.height_H(L, case["omega1"], case["omega2"], h, rng=rng)
    case["H"] = H
    case["tau_dag"] = H["tau_dag"]
    case["clusters"] = H["labels"]
    if values:
        case["n"] = integer_values(case, rng)
    return case


def edge_class(case):
    """0 = zero edge, 1 = odd edge, 2 = even non-zero edge, for every edge (edge order of drc_sim.grid)."""
    return np.where(case["omega1"], 1, np.where(case["omega2"], 2, 0))


def integer_values(case, rng=None, kmax=40):
    """Integer values n_e of the double current n = n1 + n2 given the classes: the edges are independent and
    n_e is Poisson(2J) conditioned on its class (odd / even and > 0 / zero).  (Poisson(2J) because n is the sum of
    two independent currents with Poisson(J) weights; checked against drawing n1 and n2 separately and adding.)"""
    from scipy.stats import poisson
    rng = np.random.default_rng() if rng is None else rng
    k = np.arange(kmax + 1)
    p = poisson.pmf(k, 2 * case["J"])
    cls = edge_class(case)
    n = np.zeros(len(cls), dtype=np.int64)
    for c, mask in ((1, k % 2 == 1), (2, (k % 2 == 0) & (k > 0))):
        w = np.where(mask, p, 0.0)
        idx = np.flatnonzero(cls == c)
        n[idx] = rng.choice(k, size=len(idx), p=w / w.sum())
    g = ds.grid(case["L"])
    deg = np.bincount(g["u"], weights=n, minlength=case["L"] ** 2) + np.bincount(g["v"], weights=n, minlength=case["L"] ** 2)
    assert np.all(deg % 2 == 0), "sourceless current must have even degree at every vertex"
    return n


def summary(case):
    cls = edge_class(case)
    L = case["L"]
    lab = case["clusters"]
    size = np.bincount(lab.ravel())
    H = case["H"]
    print(f"L = {L}, seed {case['seed']}, beta/beta_c = {case['beta_ratio']:.3f}")
    print(f"edges: {len(cls)} = {np.sum(cls == 1)} odd + {np.sum(cls == 2)} even + {np.sum(cls == 0)} zero")
    print(f"clusters with 2+ vertices: {np.sum(size >= 2)};  largest: {size.max()} vertices;  "
          f"isolated vertices: {np.sum(size == 1)}")
    print(f"h on faces: {case['h'].min()} .. {case['h'].max()},  at the centre face: {case['h'][L // 2 - 1, L // 2 - 1]}")
    print(f"H on vertices: {H['H_vertices'].min()} .. {H['H_vertices'].max()}")


# ----------------------------------------------------------------------------------------------------------
# what H alone says about the edges
# ----------------------------------------------------------------------------------------------------------
def decode_from_H(case):
    """For every edge, what can be read from H (and h = H on faces) alone:
         faces on the two sides have different h        -> 'odd'
         same h, the two end vertices have different H   -> 'zero'
         same h, same H at both ends                     -> 'even or zero' (H does not see the difference)
       returns an int array: 1 = odd, 0 = zero, 3 = even or zero."""
    L = case["L"]
    g = ds.grid(L)
    hf = np.r_[0, case["h"].ravel()]
    Hv = case["H"]["H_vertices"].ravel()
    differ_h = hf[g["f1"]] != hf[g["f2"]]
    differ_H = Hv[g["u"]] != Hv[g["v"]]
    return np.where(differ_h, 1, np.where(differ_H, 0, 3))


def decode_table(case):
    """Prints how the true classes (rows) are read from H alone (columns)."""
    cls, dec = edge_class(case), decode_from_H(case)
    names = {1: "odd", 2: "even", 0: "zero"}
    cols = {1: "read as odd", 0: "read as zero", 3: "even or zero (undecided)"}
    print(f"{'true class':<12}" + "".join(f"{c:>26}" for c in cols.values()))
    for k, nm in names.items():
        print(f"{nm:<12}" + "".join(f"{np.sum((cls == k) & (dec == j)):>26,}" for j in cols))
    wrong = np.sum((dec == 1) & (cls != 1)) + np.sum((dec == 0) & (cls != 0))
    print(f"edges read wrongly: {wrong}   (a rule that says 'odd' or 'zero' is never wrong; 'undecided' is not an error)")


# ----------------------------------------------------------------------------------------------------------
# pictures
# ----------------------------------------------------------------------------------------------------------
def _window(case, window):
    L = case["L"]
    if window is None:
        s = L - 1 if L <= 32 else 24
        window = (L // 2 - s // 2, L // 2 - s // 2, s) if L > 32 else (0, 0, s)
    r0, c0, s = window
    s = min(s, L - 1 - max(r0, c0))
    return r0, c0, s


def plot_case(case, view="H", window=None, edges=True, annotate=None, title=None, path=None):
    """view = 'tau' (face signs), 'h' (nesting field, integers), 'H' (faces + vertices).
    window = (top row, left column, size) in lattice units; default: the whole grid if L <= 32, else 24x24
    in the middle.  edges=False hides the edges (to check what H alone shows).  annotate: write the values
    in the squares (and discs); default: when the window has 12 faces or less per side."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    L = case["L"]
    r0, c0, s = _window(case, window)
    annotate = (s <= 12) if annotate is None else annotate
    g = ds.grid(L)
    cls = edge_class(case)
    h, tau, H = case["h"], case["tau_dag"], case["H"]

    fig, ax = plt.subplots(figsize=(9.5, 9), dpi=100)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    xs, ys = np.arange(c0, c0 + s + 1), np.arange(r0, r0 + s + 1)
    sub_h = h[r0:r0 + s, c0:c0 + s]

    legend = []
    sc = None
    if view == "tau":
        cmap = ListedColormap([ORANGE, BLUE])
        ax.pcolormesh(xs, ys, tau[r0:r0 + s, c0:c0 + s], cmap=cmap, vmin=-1, vmax=1, alpha=0.30, edgecolors="none")
        legend += [Patch(facecolor=BLUE, alpha=0.5, label="face sign +"),
                   Patch(facecolor=ORANGE, alpha=0.5, label="face sign −")]
        if annotate:
            for a in range(s):
                for b in range(s):
                    ax.text(c0 + b + 0.5, r0 + a + 0.5, "+" if tau[r0 + a, c0 + b] > 0 else "−",
                            ha="center", va="center", fontsize=8, color=INK2)
    else:
        arrs = [np.abs(h).max()] + ([np.abs(H["H_vertices"]).max()] if view == "H" else [])
        m = np.ceil(max(arrs, default=1) * 2) / 2
        step = 0.5 if view == "H" else 1.0
        lev = np.arange(-m - step / 2, m + step, step)
        cmap = plt.get_cmap("RdBu_r", len(lev) - 1)
        norm = BoundaryNorm(lev, cmap.N)
        ax.pcolormesh(xs, ys, sub_h, cmap=cmap, norm=norm, edgecolors="none")
        if annotate:
            for a in range(s):
                for b in range(s):
                    ax.text(c0 + b + 0.5, r0 + a + 0.5, f"{sub_h[a, b]:+d}" if sub_h[a, b] else "0",
                            ha="center", va="center", fontsize=8, color=INK)

    # edges: zero (light), even (dashed), odd (thick), only those inside the window
    if edges:
        r_u, c_u, r_v, c_v = g["u"] // L, g["u"] % L, g["v"] // L, g["v"] % L
        inside = (np.minimum(r_u, r_v) >= r0) & (np.maximum(r_u, r_v) <= r0 + s) & \
                 (np.minimum(c_u, c_v) >= c0) & (np.maximum(c_u, c_v) <= c0 + s)
        for k, kw in ((0, dict(color="#c9c8c3", lw=0.8, zorder=2)),
                      (2, dict(color=INK2, lw=1.8, ls=(0, (3, 2)), zorder=3)),
                      (1, dict(color=INK, lw=3.2, zorder=4, capstyle="round"))):
            m_ = inside & (cls == k)
            segs = np.stack([np.stack([c_u[m_], r_u[m_]], 1), np.stack([c_v[m_], r_v[m_]], 1)], 1)
            ax.add_collection(LineCollection(segs, **kw))
        legend += [Line2D([0], [0], color=INK, lw=3, label="odd edge"),
                   Line2D([0], [0], color=INK2, lw=1.8, ls=(0, (3, 2)), label="even edge (n_e even, > 0)"),
                   Line2D([0], [0], color="#c9c8c3", lw=1, label="zero edge")]

    if view == "H":
        rr, cc = np.mgrid[r0:r0 + s + 1, c0:c0 + s + 1]
        Hv = H["H_vertices"][rr, cc]
        sc = ax.scatter(cc.ravel(), rr.ravel(), c=Hv.ravel(), cmap=cmap, norm=norm, s=max(12, 1500 / s),
                        edgecolors="k", linewidths=0.5, zorder=5)
        if annotate:
            for r_, c_, v_ in zip(rr.ravel(), cc.ravel(), Hv.ravel()):
                ax.text(c_, r_, f"{v_:+.1f}", ha="center", va="center", fontsize=5.5, color=INK, zorder=6)
        cb = fig.colorbar(sc, ax=ax, shrink=0.7, ticks=lev[:-1] + step / 2)
        cb.set_label("H  (squares: faces, integer;  discs: vertices, half-integer)")
    else:
        ax.scatter(*np.mgrid[c0:c0 + s + 1, r0:r0 + s + 1].reshape(2, -1), s=5, color=INK2, zorder=5)
        if view == "h":
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            fig.colorbar(sm, ax=ax, shrink=0.7, ticks=lev[:-1] + step / 2).set_label("h")

    ax.set_xlim(c0 - 0.3, c0 + s + 0.3)
    ax.set_ylim(r0 + s + 0.3, r0 - 0.3)
    ax.set_aspect("equal")
    ax.axis("off")
    names = {"tau": "face signs τ† = (−1)^h", "h": "nesting field h on the faces",
             "H": "H on faces (squares) and vertices (discs)"}
    ax.set_title(title or f"{names[view]}   L = {L}, window rows {r0}..{r0 + s}, columns {c0}..{c0 + s}"
                 f"{'' if edges else ',  edges hidden'}", fontsize=10, color=INK)
    if legend:
        ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.005), ncol=5, frameon=False,
                  fontsize=8, labelcolor=INK2)
    if path:
        fig.savefig(path, dpi=140, bbox_inches="tight")
    return fig


def save_edges_csv(case, path):
    """One row per edge: vertices (row, column), faces, tau_dag and h of the faces, H of the vertices,
    class (odd / even / zero) and, if the sample has them, the integer value n_e."""
    L = case["L"]
    g = ds.grid(L)
    hf = np.r_[0, case["h"].ravel()]
    tf = np.where(hf % 2 == 0, 1, -1)
    Hv = case["H"]["H_vertices"].ravel()
    cls = edge_class(case)
    nm = {0: "zero", 1: "odd", 2: "even"}
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["edge", "vertex1(row,col)", "vertex2(row,col)", "face1", "face2", "tau_dag1", "tau_dag2",
                    "h1", "h2", "H(vertex1)", "H(vertex2)", "class"] + (["n_e"] if "n" in case else []))
        for e in range(len(cls)):
            u, v, f1, f2 = int(g["u"][e]), int(g["v"][e]), int(g["f1"][e]), int(g["f2"][e])
            w.writerow([e, divmod(u, L), divmod(v, L), f1, f2, int(tf[f1]), int(tf[f2]), int(hf[f1]), int(hf[f2]),
                        Hv[u], Hv[v], nm[int(cls[e])]] + ([int(case["n"][e])] if "n" in case else []))
