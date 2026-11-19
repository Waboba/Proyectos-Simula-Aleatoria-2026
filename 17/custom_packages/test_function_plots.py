"""Presentation figure of the test functions used in test_functions.py.

    from custom_packages import test_function_plots as tfp
    tfp.plot_test_functions(save="test_functions.png")          # the four default functions
    tfp.describe()                                              # definitions as text, for a slide

It only draws the functions; nothing here touches the simulation or its files.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401  (registers the 3D projection)

from . import test_functions as tf

INK, INK2 = "#0b0b0b", "#52514e"

# one-line description of every default function (used under the maps and by describe())
PARAMS = {
    "bump centre r=0.25": ("f1", "bump, centre (0.5, 0.5), radius 0.25", "large, centred"),
    "bump centre r=0.10": ("f2", "bump, centre (0.5, 0.5), radius 0.10", "small, centred (same centre as f1)"),
    "bump (0.3,0.3) r=0.15": ("f3", "bump, centre (0.3, 0.3), radius 0.15", "off-centre"),
    "dipole (left - right)": ("f4", "bump(0.35, 0.5; 0.15) - bump(0.65, 0.5; 0.15)", "antisymmetric: left minus right"),
}

FORMULA = (r"bump$(x_0,y_0;r)(x,y)=\exp\!\left(1-\dfrac{1}{1-\rho^{2}/r^{2}}\right)$ for "
           r"$\rho=\sqrt{(x-x_0)^2+(y-y_0)^2}<r$, and $0$ otherwise  (maximum 1 at the centre, "
           r"smooth, supported in the disc $\rho\leq r$)")


def describe(funcs=None):
    """Prints the definitions (also returns them as a string)."""
    funcs = funcs or tf.default_test_functions()
    lines = ["Test functions on D = (0,1)^2, all continuous with compact support inside D:",
             "  bump(x0, y0; r)(x, y) = exp(1 - 1 / (1 - rho^2 / r^2)) if rho < r, 0 otherwise,  rho = |(x, y) - (x0, y0)|",
             ""]
    for nm in funcs:
        tag, what, why = PARAMS.get(nm, ("", nm, ""))
        lines.append(f"  {tag:3s} {nm:24s} {what}" + (f"   [{why}]" if why else ""))
    text = "\n".join(lines)
    print(text)
    return text


def plot_test_functions(funcs=None, n=240, save=None, dpi=200):
    """2D maps (top) and 3D surfaces (bottom) of the test functions, one column per function.
    Colours: red positive, blue negative, same scale [-1, 1] for all.  save=<file.png> writes the figure."""
    funcs = funcs or tf.default_test_functions()
    k = len(funcs)
    t = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(t, t)                       # x to the right, y upwards
    fig = plt.figure(figsize=(3.9 * k, 8.2))
    fig.subplots_adjust(left=0.05, right=0.91, top=0.80, bottom=0.03, wspace=0.38, hspace=0.45)
    for j, (nm, f) in enumerate(funcs.items()):
        Z = f(X, Y)
        tag, what, why = PARAMS.get(nm, ("", "", ""))
        ax = fig.add_subplot(2, k, j + 1)
        im = ax.imshow(Z, extent=[0, 1, 0, 1], origin="lower", cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_title(f"{tag}  {nm}".strip(), fontsize=11, color=INK, pad=8)
        ax.set_xlabel("x", color=INK2)
        ax.set_ylabel("y", color=INK2)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        if what:
            ax.text(0.5, -0.26, what, transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color=INK2)
        ax3 = fig.add_subplot(2, k, k + j + 1, projection="3d")
        s = max(1, n // 80)
        ax3.plot_surface(X[::s, ::s], Y[::s, ::s], Z[::s, ::s], cmap="RdBu_r", vmin=-1, vmax=1, linewidth=0,
                         antialiased=True, rstride=1, cstride=1)
        ax3.set_zlim(-1, 1)
        ax3.set_box_aspect((1, 1, 0.7))
        ax3.set_xticks([0, 0.5, 1])
        ax3.set_yticks([0, 0.5, 1])
        ax3.set_zticks([-1, 0, 1])
        ax3.view_init(elev=28, azim=-60)
        ax3.tick_params(labelsize=7, colors=INK2)
    cax = fig.add_axes([0.935, 0.52, 0.012, 0.28])
    fig.colorbar(im, cax=cax, label="value of f")
    fig.suptitle("Test functions f on D = (0,1)²: continuous, compact support in D", fontsize=14, color=INK, y=0.965)
    fig.text(0.5, 0.905, FORMULA, ha="center", fontsize=10, color=INK2)
    if save:
        fig.savefig(save, dpi=dpi, bbox_inches="tight", facecolor="white")
    return fig
