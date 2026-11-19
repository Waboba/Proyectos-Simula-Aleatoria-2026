"""
Comparison of the height function h of the double random current with the Gaussian free
field of Theorem 1.3:   h  ->  pi^(-1/2) * GFF   (Dirichlet, G_H = (1/2 pi) log|(x - conj y)/(x - y)|).

The GFF is a random distribution, not a function, and h converges to it only in law, so there
is no pointwise coupling.  What can be compared:
  1. covariances      E[h(a) h(b)]  vs  (1/pi) G(a, b)         (cov_figure)
  2. block averages   (h averaged over b x b faces)  vs  the same averages of a lattice GFF
                      sample divided by sqrt(pi)                (compare_figure, surface_figure)

The lattice GFF is the Gaussian field on the (L-1) x (L-1) faces with covariance (-Laplacian_D)^-1
(simple-random-walk Green function, zero outside the square), whose Green function is
(1/2 pi) log(1/distance) + const, i.e. exactly the normalisation of G_H.
"""

import numpy as np
from scipy.fft import dstn


def _eigenvalues(L):
    j = np.arange(1, L)
    c = 2 - 2 * np.cos(np.pi * j / L)
    return c[:, None] + c[None, :]                      # (L-1, L-1)


def gff_sample(L, rng, scale=1 / np.sqrt(np.pi)):
    """One sample of scale * GFF on the (L-1, L-1) faces (Dirichlet), covariance scale^2 (-Delta_D)^-1.
    scale = pi^(-1/2) is the field of Theorem 1.3."""
    xi = rng.standard_normal((L - 1, L - 1))
    return scale * dstn(xi / np.sqrt(_eigenvalues(L)), type=1, norm="ortho")


def green_function(L, a, b):
    """G(face (a,b), every face) as an (L-1, L-1) array: (-Delta_D)^-1 delta_(a,b)."""
    d = np.zeros((L - 1, L - 1))
    d[a, b] = 1.0
    return dstn(dstn(d, type=1, norm="ortho") / _eigenvalues(L), type=1, norm="ortho")


def block_average(f, b):
    """Average over disjoint b x b blocks (the array is cropped to a multiple of b)."""
    n = (f.shape[0] // b) * b
    return f[:n, :n].reshape(n // b, b, n // b, b).mean(axis=(1, 3))


def _clim(*fields, q=99.5):
    return float(np.percentile(np.abs(np.concatenate([f.ravel() for f in fields])), q))


def compare_figure(h, L, block=4, seed=0, cmap="RdBu"):
    """h and an independent pi^(-1/2) GFF sample, both averaged over block x block faces,
    same colour scale.  Look for the same texture (blob sizes, contrast), not the same blobs."""
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(seed)
    hb = block_average(np.asarray(h, dtype=float), block)
    gb = block_average(gff_sample(L, rng), block)
    v = _clim(hb, gb)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.2))
    for ax, f, title in zip(axes, (hb, gb), (f"h, averaged over {block}x{block} faces, L = {L}",
                                             f"pi^(-1/2) x GFF sample, same averaging")):
        im = ax.imshow(f, cmap=cmap, vmin=-v, vmax=v, interpolation="nearest")
        ax.set_title(title)
        ax.axis("off")
    fig.colorbar(im, ax=axes, shrink=0.7)
    print(f"std of the displayed fields:  h {hb.std():.3f}   GFF {gb.std():.3f}")
    return fig


def surface_figure(h, L, block=16, seed=0, cmap="RdBu"):
    """The same two fields as 3D surfaces (coarse blocks so that the surface is readable)."""
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(seed)
    hb = block_average(np.asarray(h, dtype=float), block)
    gb = block_average(gff_sample(L, rng), block)
    v = _clim(hb, gb)
    n = hb.shape[0]
    X, Y = np.meshgrid(np.arange(n), np.arange(n))
    fig = plt.figure(figsize=(15, 6.5))
    for k, (f, title) in enumerate(((hb, f"h, {block}x{block} block averages"),
                                    (gb, "pi^(-1/2) x GFF sample"))):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        ax.plot_surface(X, Y, f, cmap=cmap, vmin=-v, vmax=v, rstride=1, cstride=1,
                        linewidth=0, antialiased=True)
        ax.set_zlim(-2.2 * v, 2.2 * v)
        ax.set_title(title)
        ax.set_axis_off()
    return fig


def cov_figure(res, L):
    """E[h(centre row, j) h(centre)] from summarize() against (1/pi) G, along the centre row."""
    import matplotlib.pyplot as plt
    c = L // 2 - 1
    emp = np.asarray(res[L]["cov_row"])
    th = green_function(L, c, c)[c, :] / np.pi
    r = np.abs(np.arange(L - 1) - c)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    axes[0].plot(np.arange(L - 1) - c, th, color="#52514e", lw=1.5, label="(1/π) G, lattice Dirichlet")
    axes[0].plot(np.arange(L - 1) - c, emp, ".", ms=3, color="#2a78d6", label="simulation")
    axes[0].set_xlabel("distance from the centre along the centre row")
    axes[0].set_ylabel("E[h(x) h(centre)]")
    axes[0].legend(frameon=False)
    m = r >= 1
    axes[1].plot(r[m], (emp - th)[m], ".", ms=3, color="#2a78d6")
    axes[1].axhline(0, color="#52514e", lw=1)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("distance")
    axes[1].set_ylabel("simulation - theory")
    return fig


# ----------------------------------------------------------------------------
# many samples: values and maxima of h against the GFF, variance over the whole lattice
# ----------------------------------------------------------------------------
def gff_variance_map(L):
    """Var of pi^(-1/2) GFF at every face: (1/pi) diag((-Delta_D)^-1), shape (L-1, L-1)."""
    a = np.arange(1, L)
    S2 = (2.0 / L) * np.sin(np.pi * np.outer(a, a) / L) ** 2         # s_j(a)^2, rows = position
    return S2 @ (1.0 / _eigenvalues(L)) @ S2.T / np.pi


def h_centre_distribution(L, out_dir=None):
    """Exact P(h(centre) = m) from the N_odd(centre) histogram saved by the main run
    (given N_odd = k, h is a sum of k independent fair signs)."""
    from . import drc_sim as d
    from scipy.stats import binom
    sts = d.load_chains(out_dir or d.OUT_DIR)[L]
    hist = sum(s["hist_nodd"] for s in sts).astype(float)
    pN = hist / hist.sum()
    kmax = int(np.flatnonzero(hist)[-1])
    m = np.arange(-kmax, kmax + 1)
    pm = np.zeros(len(m))
    for k in range(kmax + 1):
        j = np.arange(k + 1)
        pm[(2 * j - k) + kmax] += pN[k] * binom.pmf(j, k, 0.5)
    return dict(n=int(hist.sum()), pN=pN[:kmax + 1], counts_N=hist[:kmax + 1].astype(int), m=m, pm=pm)


def h_ensemble(L, n, seed=0, block=None, burn_in=200):
    """n samples of h (double random current, consecutive SW sweeps) and n independent samples of
    pi^(-1/2) GFF on the same faces.  Keeps, for both: the value at the centre, the maximum of |field|,
    and the interior block averages (block = L/16 by default)."""
    from . import drc_sim as d
    block = block or max(2, L // 16)
    g = d.grid(L)
    x, y = np.tanh(d.J_CRIT), 1.0 / np.cosh(d.J_CRIT)
    rng = np.random.default_rng(seed)
    t = []
    for _ in range(2):
        s = (rng.integers(0, 2, size=g["n_faces"], dtype=np.int8) * 2 - 1).astype(np.int8)
        s[0] = 1
        t.append(s)
    for _ in range(burn_in):
        t = [d.sw_step(s, g, x, rng) for s in t]
    c = L // 2 - 1
    nb = (L - 1) // block
    q = nb // 4
    e = dict(L=L, n=n, block=block, h_centre=np.empty(n), h_max=np.empty(n), g_centre=np.empty(n),
             g_max=np.empty(n), h_blocks=[], g_blocks=[])
    for k in range(n):
        t = [d.sw_step(s, g, x, rng) for s in t]
        a1, b1 = d.current_omega(t[0], g, y, rng)
        a2, b2 = d.current_omega(t[1], g, y, rng)
        h = d.nesting(a1 ^ a2, a1 | b1 | a2 | b2, g, rng)[0].astype(float)
        f = gff_sample(L, rng)
        for name, field in (("h", h), ("g", f)):
            e[name + "_centre"][k] = field[c, c]
            e[name + "_max"][k] = np.abs(field).max()
            e[name + "_blocks"].append(block_average(field, block)[q:nb - q, q:nb - q].ravel())
    e["h_blocks"], e["g_blocks"] = np.concatenate(e["h_blocks"]), np.concatenate(e["g_blocks"])
    return e


def ensemble_summary(e):
    from scipy.stats import kurtosis
    print(f"L = {e['L']}, {e['n']} samples of h and of pi^(-1/2) GFF")
    print(f"  centre:  Var h {e['h_centre'].var():.4f}   Var GFF {e['g_centre'].var():.4f}")
    vals, cnt = np.unique(e["h_centre"], return_counts=True)
    print("  h(centre) values:", ", ".join(f"{int(v)}: {c}" for v, c in zip(vals, cnt)))
    vals, cnt = np.unique(e["h_max"], return_counts=True)
    print("  max |h| over the lattice:", ", ".join(f"{int(v)}: {c}" for v, c in zip(vals, cnt)),
          f"  (mean {e['h_max'].mean():.2f})   max |GFF|: mean {e['g_max'].mean():.2f}, "
          f"range {e['g_max'].min():.2f}-{e['g_max'].max():.2f}")
    print(f"  block averages ({e['block']}x{e['block']}, interior): std h {e['h_blocks'].std():.4f}  GFF "
          f"{e['g_blocks'].std():.4f} | excess kurtosis h {kurtosis(e['h_blocks']):+.3f}  GFF {kurtosis(e['g_blocks']):+.3f}")


def ensemble_figure(e):
    """Distributions over many samples: h(centre), block averages, and the maximum over the lattice."""
    import matplotlib.pyplot as plt
    blue, orange, ink2 = "#2a78d6", "#eb6834", "#52514e"
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    ax = axes[0]
    vals, cnt = np.unique(e["h_centre"], return_counts=True)
    ax.bar(vals, cnt / e["n"], width=0.6, color=blue, label="h(centre), integer")
    s = e["g_centre"].std()
    xs = np.linspace(-4.5, 4.5, 400)
    ax.plot(xs, np.exp(-xs ** 2 / (2 * s * s)) / np.sqrt(2 * np.pi * s * s), color=orange, lw=2,
            label="π^(-1/2) GFF(centre), density")
    ax.set_yscale("log")
    ax.set_ylim(0.3 / e["n"], 1.5)
    ax.set_xlabel("value at the centre")
    ax.set_title("pointwise value")
    ax.legend(frameon=False, fontsize=9)
    ax = axes[1]
    lim = 1.05 * max(np.abs(e["h_blocks"]).max(), np.abs(e["g_blocks"]).max())
    bins = np.linspace(-lim, lim, 61)
    ax.hist(e["h_blocks"], bins=bins, density=True, histtype="step", lw=2, color=blue, label="h")
    ax.hist(e["g_blocks"], bins=bins, density=True, histtype="step", lw=2, color=orange, label="π^(-1/2) GFF")
    ax.set_yscale("log")
    ax.set_xlabel(f"average over a {e['block']}x{e['block']} block")
    ax.set_title("block averages (interior blocks)")
    ax.legend(frameon=False, fontsize=9)
    ax = axes[2]
    hi = max(e["h_max"].max(), e["g_max"].max()) + 0.5
    ax.hist(e["h_max"], bins=np.arange(-0.25, hi + 0.5, 0.5), density=True, color=blue, alpha=0.85, label="max |h|")
    ax.hist(e["g_max"], bins=40, density=True, histtype="step", lw=2, color=orange, label="max |π^(-1/2) GFF|")
    ax.set_xticks(np.arange(0, int(hi) + 1))
    ax.set_xlabel("maximum over the lattice")
    ax.set_title("maximum")
    ax.legend(frameon=False, fontsize=9)
    fig.suptitle(f"L = {e['L']}, {e['n']} samples each", color=ink2)
    fig.tight_layout()
    return fig


def variance_map_figure(res, L):
    """Var h at every face from the main run (= E[N_odd]) against Var of pi^(-1/2) GFF (exact)."""
    import matplotlib.pyplot as plt
    vh = np.asarray(res[L]["field"])
    vg = gff_variance_map(L)
    diff = vh - vg
    c = L // 2 - 1
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    vmax = max(vh.max(), vg.max())
    im = axes[0].imshow(vh, cmap="Blues", vmin=0, vmax=vmax)
    axes[0].set_title(f"Var h at every face, L = {L} ({res[L]['samples']} samples)")
    fig.colorbar(im, ax=axes[0], shrink=0.8)
    im = axes[1].imshow(diff, cmap="Blues", vmin=min(0, np.percentile(diff, 0.5)), vmax=np.percentile(diff, 99.5))
    axes[1].set_title("Var h − Var π^(-1/2) GFF")
    fig.colorbar(im, ax=axes[1], shrink=0.8)
    for ax in axes[:2]:
        ax.axis("off")
    ax = axes[2]
    xs = np.arange(L - 1) - c
    ax.plot(xs, vh[c], color="#2a78d6", lw=1.5, label="Var h")
    ax.plot(xs, vg[c], color="#eb6834", lw=1.5, label="Var π^(-1/2) GFF")
    ax.plot(xs, diff[c], color="#52514e", lw=1.5, label="difference")
    ax.set_xlabel("position along the centre row")
    ax.legend(frameon=False)
    ax.set_title("centre row")
    fig.tight_layout()
    return fig
