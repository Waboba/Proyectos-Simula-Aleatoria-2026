"""Figures for the presentation, drawn at the size they have on a 16:9 beamer slide.

Every figure is saved in the folder  Graficos/  of the project (next to the notebooks), created on first use,
so the project folder does not fill up with PNGs.  Each function takes only a file NAME, e.g. "Varianza.png".

    from custom_packages import presentation_figures as pf

    # monitor_distribution.ipynb
    pf.figura_funciones_test([0, 1], "Funciones_test_1.png")
    pf.figura_comparacion(1024, [0, 1], "Comparacion_Funciones_test_1.png")
    pf.figura_varianza(res, "Varianza_h_y_GFF.png")                 # res = drc_sim.summarize()
    pf.figura_convergencia(1024, "Convergencia_al_GFF_1024.png")
    pf.figura_gaussianidad("No_gaussianidad_h.png")

    # monitor_geometry.ipynb
    pf.figura_H(H, "Funcion_Altura.png")                            # H = drc_height.height_H(...)
    pf.figura_clusters(L, omega1, omega2, "Clusters_1024.png", titulo="...")

    # ising_double_current_sampler_V3.ipynb  (cases come from run_case / run_double_case of that notebook)
    pf.figura_corriente(case, "Corriente_Aleatoria_UnicoSample.png", titulo="sin fuentes")
    pf.figura_corrientes(results, ["sin fuentes", "2 fuentes", "8 fuentes"], "Corriente_Aleatoria_V2.png")

Why the slide size: a figure drawn 12-15 inches wide and shrunk by LaTeX to ~5.5 inches keeps only a third of its
font size.  Drawn at 5.5 inches, 10 pt in matplotlib is 10 pt on the slide.  In LaTeX use
\\includegraphics[width=\\textwidth]{...} (full width) or [width=\\linewidth] inside a column, without caption.

Every function has a `titulo` argument: the default is the title used so far, titulo=None draws no title.
The style (font sizes, no top/right spines) is applied only while each figure is drawn; it does not change
the other plots of the notebook.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import NullFormatter
from scipy.stats import norm

from . import test_functions as tf
from . import drc_height as dh
from . import drc_clusters as dcl

# ----------------------------------------------------------------------------------------------------------
# output folder, size and style
# ----------------------------------------------------------------------------------------------------------
GRAFICOS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Graficos")

ANCHO = 5.5                       # inches: text width of a 16:9 beamer slide
ALTO_MAX = 2.75                   # inches: a taller figure no longer fits under the frame title
DPI = 300

AZUL, NARANJA, VERDE, MORADO = "#2a78d6", "#eb6834", "#2a9d6f", "#8a5cc2"
COLS = [AZUL, NARANJA, VERDE, MORADO]
NEGRO, GRIS = "#0b0b0b", "#52514e"
ETIQ = ["$f_1$", "$f_2$", "$f_3$", "$f_4$"]

ESTILO = {"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "xtick.labelsize": 9,
          "ytick.labelsize": 9, "legend.fontsize": 9, "axes.spines.top": False, "axes.spines.right": False}

# currents (same colours as the sampler notebook)
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
COL_CARA = {+1: "#2a78d6", -1: "#eb6834"}
FUENTE_LAMINA = 10                # pt: titles and legends of the current figures
ALPHA_CARAS = 0.45                # face colours (stronger than in the notebook: projectors wash them out)
COLOR_PAR = "#138a4b"             # even non-zero edges: solid green (tested against both face colours)


def ruta(nombre):
    """Full path of a figure in Graficos/ (the folder is created if needed)."""
    os.makedirs(GRAFICOS, exist_ok=True)
    return os.path.join(GRAFICOS, nombre)


def _guardar(fig, nombre, mostrar=True):
    path = ruta(nombre)
    fig.savefig(path, dpi=DPI, facecolor="white")
    print("guardado:", path)
    if mostrar:
        plt.show()
    else:
        plt.close(fig)


def _estilo(f):
    """Decorator: draws the figure inside plt.rc_context(ESTILO), so the style stays local to it."""
    def g(*args, **kwargs):
        with plt.rc_context(ESTILO):
            return f(*args, **kwargs)
    g.__name__, g.__doc__ = f.__name__, f.__doc__
    return g


# ----------------------------------------------------------------------------------------------------------
# test functions and h(f) against the GFF            (monitor_distribution)
# ----------------------------------------------------------------------------------------------------------
@_estilo
def figura_funciones_test(idx, nombre, titulo="Funciones test", ancho=ANCHO, alto=2.6, n=160, mostrar=True):
    """3D surfaces of the test functions idx (e.g. [0, 1] or [2, 3])."""
    funcs = list(tf.default_test_functions().values())
    t = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(t, t)
    fig = plt.figure(figsize=(ancho, alto))
    for k, i in enumerate(idx):
        Z = funcs[i](X, Y)
        ax = fig.add_subplot(1, len(idx), k + 1, projection="3d")
        ax.plot_surface(X[::2, ::2], Y[::2, ::2], Z[::2, ::2], cmap="RdBu_r", vmin=-1, vmax=1,
                        linewidth=0, antialiased=True)
        ax.set_zlim(-1, 1)
        ax.set_box_aspect((1, 1, 0.65))
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_zticks([-1, 0, 1])
        ax.tick_params(labelsize=9, colors=GRIS, pad=0)
        ax.view_init(elev=28, azim=-60)
        ax.set_title(ETIQ[i], fontsize=14, color=NEGRO, pad=0)
    if titulo:
        fig.suptitle(titulo, fontsize=12)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.02, top=0.80 if titulo else 0.90, wspace=0.0)
    _guardar(fig, nombre, mostrar)


@_estilo
def figura_comparacion(L, idx, nombre, name="main", funcs=None, titulo="default", ancho=ANCHO, alto=2.65,
                       mostrar=True):
    """For each function in idx: histogram of h(f) against the exact GFF law (top) and quantile-quantile plot
    (bottom).  Values are divided by the GFF standard deviation, so both axes count standard deviations."""
    funcs = funcs or tf.default_test_functions()
    if titulo == "default":
        titulo = f"$h(f)$ contra el GFF ($L = {L}$)"
    names, vals = tf.load(L, name)
    vals = [v for v in vals if len(v) >= 50]
    C = tf.gff_covariance(L, funcs)
    fig, axes = plt.subplots(2, len(idx), figsize=(ancho, alto), squeeze=False)
    x = np.linspace(-5, 5, 400)
    for k, i in enumerate(idx):
        z = np.concatenate([v[:, i] for v in vals]) / np.sqrt(C[i, i])
        ax = axes[0, k]
        bins = np.linspace(-5, 5, 51)
        ax.hist(z, bins=bins, density=True, color=AZUL, alpha=0.85, label=f"$h(f)$ simulado ($L={L}$)")
        ax.plot(x, norm.pdf(x), color=NEGRO, lw=1.5, label="GFF$/\\sqrt{\\pi}$ (exacto)")
        ax.set_yscale("log")
        ax.set_ylim(0.5 / (len(z) * (bins[1] - bins[0])), 1)
        ax.set_xlim(-5, 5)
        ax.set_title(ETIQ[i], fontsize=13, color=NEGRO, pad=3)
        ax = axes[1, k]
        q = np.linspace(0.0005, 0.9995, 1500)
        ax.plot(norm.ppf(q), np.quantile(z, q), color=AZUL, lw=2.2)
        ax.plot([-4, 4], [-4, 4], color=GRIS, lw=1, ls="--")
        ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
        ax.set_xticks([-4, 0, 4]); ax.set_yticks([-4, 0, 4])
        ax.set_xlabel("cuantiles GFF")
        if k == 0:
            ax.set_ylabel("cuantiles $h(f)$")
    h, l = axes[0, 0].get_legend_handles_labels()
    if titulo:
        fig.suptitle(titulo, fontsize=12)
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 0.925 if titulo else 1.0))
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.18, top=0.70 if titulo else 0.79, hspace=0.65, wspace=0.25)
    _guardar(fig, nombre, mostrar)


@_estilo
def figura_varianza(res, nombre, titulo="Varianza de $h$ en el centro de la grilla", alto=2.6, mostrar=True):
    """Var h(centre) against L (log scale) and the line of slope 1/(2 pi^2) through the largest L.
    res = drc_sim.summarize()"""
    Ls = np.array(sorted(res))
    v = np.array([res[L]["var"] for L in Ls])
    e = np.array([res[L]["var_se"] for L in Ls])
    x = np.log(Ls)
    b = 1 / (2 * np.pi ** 2)
    fig, ax = plt.subplots(figsize=(ANCHO, alto))
    ax.plot(x, v[-1] + b * (x - x[-1]), color=NEGRO, lw=1.2, label="pendiente $1/(2\\pi^2)$")
    ax.errorbar(x, v, e, fmt="o", color=AZUL, ms=6, capsize=3, label="simulación")
    ax.set_xticks(x)
    ax.set_xticklabels(Ls)
    ax.set_xlabel("L (escala logarítmica)")
    ax.set_ylabel("Var $h$(centro)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="upper left")
    if titulo:
        fig.suptitle(titulo, fontsize=12)
    fig.tight_layout()
    _guardar(fig, nombre, mostrar)


def _leyenda_abajo(fig, ax, titulo):
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False)
    if titulo:
        fig.suptitle(titulo, fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=1.5)


@_estilo
def figura_convergencia(L, nombre, name="main", funcs=None, titulo="default", alto=2.6, mostrar=True):
    """Variance ratio and excess kurtosis of h(f) as samples are added (bands = +- 1 standard error)."""
    if titulo == "default":
        titulo = f"Convergencia de $h(f)$ al GFF ($L = {L}$)"
    r = tf.running_estimates(L, name, funcs)
    fig, ax = plt.subplots(1, 2, figsize=(ANCHO, alto))
    for i in range(len(r["names"])):
        for a, key in ((ax[0], "ratio"), (ax[1], "kurt")):
            y, s = r[key][:, i], r[key + "_se"][:, i]
            a.plot(r["N"], y, color=COLS[i], lw=1.6, label=ETIQ[i])
            a.fill_between(r["N"], y - s, y + s, color=COLS[i], alpha=0.15, lw=0)
    ax[0].axhline(1, color=NEGRO, lw=1, ls="--")
    ax[1].axhline(0, color=NEGRO, lw=1, ls="--")
    ax[0].set_title("Var $h(f)$ / Var GFF$(f)$")
    ax[1].set_title("curtosis de $h(f)$")
    for a in ax:
        a.set_xscale("log")
        a.xaxis.set_minor_formatter(NullFormatter())
        a.set_xlabel("muestras usadas")
        a.grid(alpha=0.25)
    _leyenda_abajo(fig, ax[0], titulo)
    _guardar(fig, nombre, mostrar)


@_estilo
def figura_gaussianidad(nombre, name="main", funcs=None, titulo="Razón de varianzas y curtosis de $h(f)$ según $L$",
                        alto=2.6, mostrar=True):
    """Variance ratio and excess kurtosis of h(f) against L (all samples; bars = +- 1 standard error)."""
    rows = tf.trend(name, funcs)
    Ls = sorted(rows)
    xs = np.log2(Ls)
    fig, ax = plt.subplots(1, 2, figsize=(ANCHO, alto))
    k = len(rows[Ls[0]]["names"])
    for i in range(k):
        dx = (i - (k - 1) / 2) * 0.05
        for a, key in ((ax[0], "ratio"), (ax[1], "kurt")):
            y = [rows[L][key][i] for L in Ls]
            s = [rows[L][key + "_se"][i] for L in Ls]
            a.errorbar(xs + dx, y, s, fmt="o-", color=COLS[i], lw=1.2, ms=5, capsize=2, label=ETIQ[i])
    ax[0].axhline(1, color=NEGRO, lw=1, ls="--")
    ax[1].axhline(0, color=NEGRO, lw=1, ls="--")
    ax[0].set_title("Var $h(f)$ / Var GFF$(f)$")
    ax[1].set_title("curtosis de $h(f)$")
    for a in ax:
        a.set_xticks(xs)
        a.set_xticklabels(Ls)
        a.set_xlabel("L")
        a.grid(alpha=0.25)
    _leyenda_abajo(fig, ax[0], titulo)
    _guardar(fig, nombre, mostrar)


# ----------------------------------------------------------------------------------------------------------
# height function and clusters            (monitor_geometry)
# ----------------------------------------------------------------------------------------------------------
@_estilo
def figura_H(Hres, nombre, window=None, titulo=None, alto=2.3, mostrar=True):
    """H on faces, H on vertices and a zoom (faces = squares, vertices = discs).
    Hres = drc_height.height_H(...) (the dict passed to plot_H); window = (row0, col0, side)."""
    Hf, Hv = Hres["H_faces"], Hres["H_vertices"]
    L = Hv.shape[0]
    lev = dh._levels(Hres)
    cmap = plt.get_cmap("RdBu_r", len(lev) - 1)
    nrm = BoundaryNorm(lev, cmap.N)
    if window is None:
        s = min(16, L - 1)
        window = (L // 2 - s // 2, L // 2 - s // 2, s)
    r0, c0, s = window
    extra = 0.25 if titulo else 0.0
    alto = alto + extra
    fig, axes = plt.subplots(1, 3, figsize=(ANCHO, alto))
    for ax, arr, t, off in ((axes[0], Hf, "H en caras (= h)", 0.5), (axes[1], Hv, "H en vértices", 0.0)):
        n = arr.shape[0]
        ax.imshow(arr, cmap=cmap, norm=nrm, interpolation="nearest",
                  extent=(off - 0.5, off + n - 0.5, off + n - 0.5, off - 0.5))
        ax.add_patch(plt.Rectangle((c0 - 0.5 + off, r0 - 0.5 + off), s, s, fill=False, ec="k", lw=1.2))
        ax.set_title(t)
    ax = axes[2]
    ax.pcolormesh(np.arange(c0, c0 + s + 1), np.arange(r0, r0 + s + 1), Hf[r0:r0 + s, c0:c0 + s],
                  cmap=cmap, norm=nrm, edgecolors="none")
    rr, cc = np.mgrid[r0:r0 + s + 1, c0:c0 + s + 1]
    sel = (rr < L) & (cc < L)
    sc = ax.scatter(cc[sel], rr[sel], c=Hv[rr[sel], cc[sel]], cmap=cmap, norm=nrm, s=16,
                    edgecolors="k", linewidths=0.4, zorder=3)
    ax.set_xlim(c0 - 0.5, c0 + s + 0.5)
    ax.set_ylim(r0 + s + 0.5, r0 - 0.5)
    ax.set_title("zoom")
    for a in axes:
        a.set_aspect("equal")
        a.set_xticks([])
        a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(True)
    if titulo:
        fig.suptitle(titulo, fontsize=12)
    top = 0.88 * (alto - extra) / alto
    fig.subplots_adjust(left=0.01, right=0.90, top=top, bottom=0.02, wspace=0.06)
    cax = fig.add_axes([0.92, 0.12 * (alto - extra) / alto, 0.018, 0.68 * (alto - extra) / alto])
    step = max(1, int(np.ceil((len(lev) - 1) / 5)))          # about 5 ticks on the colour bar
    cb = fig.colorbar(sc, cax=cax, ticks=(lev[:-1] + 0.25)[::step])
    cb.ax.tick_params(labelsize=9)
    cb.set_label("H", fontsize=10)
    _guardar(fig, nombre, mostrar)


@_estilo
def figura_clusters(L, omega1, omega2, nombre, mode="border", titulo=None, lado=2.6, mostrar=True):
    """Clusters (by default those touching the border).  A long title can be split with "\\n"."""
    img, info = dcl.cluster_image(L, omega1, omega2, mode)
    lineas = titulo.count("\n") + 1 if titulo else 0
    extra = 0.08 + 0.2 * lineas if titulo else 0             # room above for the title
    alto = lado + extra
    fig, ax = plt.subplots(figsize=(lado, alto))
    ax.imshow(img, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color(GRIS)
    if titulo:
        ax.set_title(titulo, fontsize=11)
    fig.subplots_adjust(left=0.01, right=0.99, top=1 - (extra / alto if titulo else 0.01), bottom=0.01)
    _guardar(fig, nombre, mostrar)
    print(f"{len(info)} clústeres tocan el borde")


# ----------------------------------------------------------------------------------------------------------
# single and double current on a small grid            (ising_double_current_sampler_V3)
# a "case" is the dict returned by run_case / run_double_case of that notebook:
# keys R, C, A, edges, edge_faces, n_outer, tau, odd, in2
# ----------------------------------------------------------------------------------------------------------
def draw_lamina(ax, case, lw_impar=2.0, lw_par=1.6, lw_cero=0.4, punto=1.5, fuente=28):
    """Like draw() of the notebook, with line widths for a small figure and without +/- on the faces."""
    R, C, A = case["R"], case["C"], case["A"]
    edges, ef, n_outer = case["edges"], case["edge_faces"], case["n_outer"]
    tau, odd, in2 = case["tau"], case["odd"], case["in2"]
    inner = tau[n_outer:].reshape(R - 1, C - 1)
    ax.set_facecolor(SURFACE)
    for a in range(R - 1):
        for b in range(C - 1):
            ax.add_patch(Rectangle((b, -(a + 1)), 1, 1, facecolor=COL_CARA[inner[a, b]], alpha=ALPHA_CARAS, lw=0))
    lo, hi = 0.15, 0.45                                       # outer strips = border sectors
    for e in range(len(edges)):
        f1, f2 = ef[e]
        sec = f1 if f1 < n_outer else (f2 if f2 < n_outer else None)
        if sec is None:
            continue
        (i1, j1), (i2, j2) = divmod(int(edges[e, 0]), C), divmod(int(edges[e, 1]), C)
        if i1 == i2 == 0:
            rect = (min(j1, j2), lo, 1, hi - lo)
        elif i1 == i2 == R - 1:
            rect = (min(j1, j2), -(R - 1) - hi, 1, hi - lo)
        elif j1 == j2 == 0:
            rect = (-hi, -max(i1, i2), hi - lo, 1)
        else:
            rect = (C - 1 + lo, -max(i1, i2), hi - lo, 1)
        ax.add_patch(Rectangle(rect[:2], rect[2], rect[3], facecolor=COL_CARA[tau[sec]], alpha=0.9, lw=0))
    for e in range(len(edges)):
        u, v = divmod(int(edges[e, 0]), C), divmod(int(edges[e, 1]), C)
        xs, ys = [u[1], v[1]], [-u[0], -v[0]]
        if odd[e]:
            ax.plot(xs, ys, color=INK, lw=lw_impar, solid_capstyle="round", zorder=4)
        elif in2[e]:
            ax.plot(xs, ys, color=COLOR_PAR, lw=lw_par, solid_capstyle="butt", zorder=3)
        else:
            ax.plot(xs, ys, color="#c9c8c3", lw=lw_cero, zorder=2)
    ii, jj = np.meshgrid(range(R), range(C), indexing="ij")
    ax.scatter(jj.ravel(), -ii.ravel(), s=punto, color=INK2, zorder=5, linewidths=0)
    if A:
        ax.scatter([v[1] for v in A], [-v[0] for v in A], s=fuente, facecolor=SURFACE,
                   edgecolor=INK, linewidth=1.5, zorder=6)
    ax.set_aspect("equal")
    ax.set_xlim(-0.6, C - 0.4)
    ax.set_ylim(-(R - 1) - 0.6, 0.6)
    ax.axis("off")


def leyenda_lamina(con_fuentes=True):
    h = [Line2D([0], [0], color=INK, lw=2.0, label="impar"),
         Line2D([0], [0], color=COLOR_PAR, lw=2.0, label="par > 0")]
    if con_fuentes:
        h.append(Line2D([0], [0], marker="o", color="none", markerfacecolor=SURFACE, markeredgecolor=INK,
                        markeredgewidth=1.5, markersize=6, label="fuente"))
    return h


def figura_corriente(case, nombre, titulo=None, ancho=2.55, mostrar=True):
    """One sample (right column of a slide).  In LaTeX: \\includegraphics[width=\\linewidth]{...}"""
    alto = ancho + (0.25 if titulo else 0)                   # square: fits the column without overflowing
    fig, ax = plt.subplots(figsize=(ancho, alto))
    fig.patch.set_facecolor("white")
    draw_lamina(ax, case)
    if titulo:
        ax.set_title(titulo, fontsize=FUENTE_LAMINA + 1, color=INK, pad=4)
    fig.legend(handles=leyenda_lamina(bool(case["A"])), loc="lower center", ncol=3, frameon=False,
               fontsize=FUENTE_LAMINA, handlelength=1.6, columnspacing=1.0, handletextpad=0.4)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99 - (0.25 if titulo else 0) / alto, bottom=0.30 / alto)
    _guardar(fig, nombre, mostrar)


def figura_corrientes(results, titulos, nombre, ancho=ANCHO, alto=2.55, mostrar=True):
    """Several samples side by side (full slide, no caption).  In LaTeX: \\includegraphics[width=\\textwidth]{...}
    results = dict label -> case (the `results` / `dres` of the notebook); titulos = one short title per case."""
    fig, axes = plt.subplots(1, len(results), figsize=(ancho, alto))
    fig.patch.set_facecolor("white")
    for ax, cs, t in zip(np.atleast_1d(axes), results.values(), titulos):
        draw_lamina(ax, cs, lw_impar=1.6, lw_par=1.2, lw_cero=0.3, punto=1.0, fuente=18)
        ax.set_title(t, fontsize=FUENTE_LAMINA + 1, color=INK, pad=3)
    con_fuentes = any(cs["A"] for cs in results.values())
    fig.legend(handles=leyenda_lamina(con_fuentes), loc="lower center", ncol=3, frameon=False,
               fontsize=FUENTE_LAMINA, handlelength=1.8, columnspacing=1.5)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.12, wspace=0.08)
    _guardar(fig, nombre, mostrar)
