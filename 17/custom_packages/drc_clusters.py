"""
Drawing the clusters (connected components of the support of the double current) of one sample.

  cluster_image(L, omega1, omega2, mode="border")
      mode="border": clusters that touch the border of the domain, each in its own colour, plus the outer
                     boundary of each of them drawn in black; every other cluster in light grey.
      mode="all"   : every cluster in its own colour (the 300k small ones are recycled colours).
  draw_clusters(...)  matplotlib figure with the image, the number/sizes of border clusters in the title.
  frame(...)          three-panel RGB frame (h | clusters | face spins) used by the video test.
"""
import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from . import drc_sim as ds


def labels(L, omega1, omega2):
    g = ds.grid(L)
    sup = omega1 | omega2
    adj = coo_matrix((np.ones(sup.sum(), dtype=np.int8), (g["u"][sup], g["v"][sup])), shape=(L * L, L * L))
    n, lab = connected_components(adj, directed=False)
    return n, lab.reshape(L, L)


_PAL = np.array([[31, 119, 180], [255, 127, 14], [44, 160, 44], [214, 39, 40], [148, 103, 189], [140, 86, 75],
                 [227, 119, 194], [188, 189, 34], [23, 190, 207], [0, 0, 0]], dtype=np.uint8)[:9]


def border_clusters(lab):
    b = np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    return b


def cluster_image(L, omega1, omega2, mode="border", min_size=2, scale=1, lab=None):
    """RGB uint8 image (L*scale, L*scale).  Returns image and the list of (label, size) of border clusters."""
    if lab is None:
        _, lab = labels(L, omega1, omega2)
    size = np.bincount(lab.ravel())
    img = np.full((L, L, 3), 255, dtype=np.uint8)
    rng = np.random.default_rng(1)
    bord = border_clusters(lab)
    bord = bord[size[bord] >= min_size]
    if mode == "all":
        col = rng.integers(40, 230, size=(len(size), 3)).astype(np.uint8)
        big = size[lab] >= min_size
        img[big] = col[lab[big]]
    else:
        img[size[lab] >= min_size] = (225, 225, 225)
    order = bord[np.argsort(-size[bord])]
    objs = ndimage.find_objects(lab + 1)                      # bounding boxes, index = label
    for k, c in enumerate(order):
        sl = objs[c]
        m = lab[sl] == c
        img[sl][m] = _PAL[k % 8] if mode == "border" else img[sl][m]
        filled = ndimage.binary_fill_holes(np.pad(m, 1))[1:-1, 1:-1]
        outer = filled & ~ndimage.binary_erosion(np.pad(filled, 1))[1:-1, 1:-1]
        outer &= m | ~m                                       # boundary of the filled shape
        sub = img[sl]
        sub[outer & m] = (0, 0, 0)
    if scale > 1:
        img = np.kron(img, np.ones((scale, scale, 1), dtype=np.uint8))
    return img, [(int(c), int(size[c])) for c in order]


def draw_clusters(L, omega1, omega2, mode="border", path=None, window=None, title=None):
    import matplotlib.pyplot as plt
    img, info = cluster_image(L, omega1, omega2, mode)
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    ax.imshow(img, interpolation="nearest")
    if window:
        r0, c0, s = window
        ax.set_xlim(c0, c0 + s); ax.set_ylim(r0 + s, r0)
    ax.set_xticks([]); ax.set_yticks([])
    if mode == "all":
        n = int(np.sum(np.bincount(labels(L, omega1, omega2)[1].ravel()) >= 2))
        default = f"L={L}: all clusters with 2 or more vertices ({n} of them), one colour each"
    else:
        default = (f"L={L}: {len(info)} clusters touching the border "
                   f"(sizes {', '.join(str(s) for _, s in info[:6])}{'...' if len(info) > 6 else ''})")
    ax.set_title(title or default)
    if path:
        fig.savefig(path, dpi=130); plt.close(fig)
    return fig, info


def frame(L, omega1, omega2, h, tau_dag_face=None, scale=1, mode="border"):
    """Three side-by-side panels as one RGB uint8 array: h | clusters | face spins (tau_dag=(-1)^h)."""
    import matplotlib
    cm = matplotlib.colormaps["RdBu_r"]
    hh = np.clip(h, -3, 3)
    p1 = (cm((hh + 3.5) / 7.0)[..., :3] * 255).astype(np.uint8)
    p3 = np.where((h % 2 == 0)[..., None], np.uint8(245), np.uint8(40)) * np.ones((1, 1, 3), np.uint8)
    p2, _ = cluster_image(L, omega1, omega2, mode)
    p2 = p2[:L - 1, :L - 1]
    out = np.concatenate([p1, np.full((L - 1, 4, 3), 120, np.uint8), p2, np.full((L - 1, 4, 3), 120, np.uint8), p3], axis=1)
    if scale > 1:
        out = np.kron(out, np.ones((scale, scale, 1), dtype=np.uint8))
    return out

def _font(px):
    """DejaVu Sans (ships with matplotlib, so it exists on every computer with matplotlib)."""
    import os
    import matplotlib
    from PIL import ImageFont
    return ImageFont.truetype(os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf"), px)


def _ajustar(draw, texto, ancho, px):
    """Largest font (<= px) for which `texto` fits in `ancho` pixels."""
    while px > 8:
        f = _font(px)
        if draw.textlength(texto, font=f) <= ancho:
            return f
        px -= 1
    return _font(8)


def titulo_frame(fr, titulo=None, paneles=None):
    """White band on top of an RGB frame: a centred title and, optionally, one label per panel."""
    from PIL import Image, ImageDraw
    if not titulo and not paneles:
        return fr
    H, W = fr.shape[:2]
    px = max(14, W // 55)                                    # title size relative to the frame width
    filas = (1 if titulo else 0) + (1 if paneles else 0)
    banda = int(filas * 1.5 * px + 0.5 * px)
    img = Image.new("RGB", (W, H + banda), "white")
    img.paste(Image.fromarray(fr), (0, banda))
    d = ImageDraw.Draw(img)
    y = int(0.35 * px)
    if titulo:
        d.text((W / 2, y), titulo, font=_ajustar(d, titulo, W - 20, px), fill=(11, 11, 11), anchor="ma")
        y += int(1.5 * px)
    if paneles:
        ancho = W / len(paneles)
        for i, t in enumerate(paneles):
            d.text(((i + 0.5) * ancho, y), t, font=_ajustar(d, t, ancho - 10, int(0.8 * px)),
                   fill=(82, 81, 78), anchor="ma")
    return np.asarray(img)

def _ffmpeg():
    """Path of the ffmpeg program: the one on the PATH, else the one shipped with the package imageio-ffmpeg.
    If neither is found, the error says which Python the notebook is using, so the package can be installed there."""
    import shutil
    import sys
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
    except ImportError:
        raise RuntimeError(
            "ffmpeg not found: the package imageio-ffmpeg is not installed in the Python that runs this notebook:\n"
            f"    {sys.executable}\n"
            "Install it from a notebook cell (this installs into that same Python), then restart the kernel:\n"
            "    %pip install imageio-ffmpeg") from None
    try:
        return imageio_ffmpeg.get_ffmpeg_exe()
    except RuntimeError as err:
        raise RuntimeError(f"imageio-ffmpeg is installed but its ffmpeg binary is missing ({err}).\n"
                           "Reinstall it from a notebook cell, then restart the kernel:\n"
                           "    %pip install --force-reinstall imageio-ffmpeg") from None


def _codec(exe):
    """H.264 if this ffmpeg has it (libx264), else MPEG-4 part 2 (plays everywhere, files about twice as big)."""
    import subprocess
    try:
        out = subprocess.run([exe, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=30).stdout
    except Exception:
        out = ""
    if "libx264" in out:
        return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p"]
    return ["-c:v", "mpeg4", "-q:v", "4", "-pix_fmt", "yuv420p"]


def make_video(path, L=256, n_frames=300, fps=10, scale=2, seed=0, burn_in=200, beta_ratio=None, mode="border",
               sweeps_per_frame=1, titulo=None, paneles=None):
    """Stream frames straight into ffmpeg (nothing is stored on disk but the mp4).
    beta_ratio: None -> critical;  a list/array of length n_frames of beta/beta_c (e.g. np.linspace(0.8, 1.2, n))
    gives a 'changes of phase' movie (the sampler is the same, only J changes from frame to frame)."""
    import subprocess
    import tempfile
    exe = _ffmpeg()
    codec = _codec(exe)
    g = ds.grid(L)
    rng = np.random.default_rng(np.random.SeedSequence([9200, L, seed]))
    t = [(rng.integers(0, 2, size=g["n_faces"], dtype=np.int8) * 2 - 1).astype(np.int8) for _ in range(2)]
    for s in t:
        s[0] = 1
    p = None

    def step(J):
        x, y = np.tanh(J), 1.0 / np.cosh(J)
        t[0] = ds.sw_step(t[0], g, x, rng)
        t[1] = ds.sw_step(t[1], g, x, rng)
        return x, y

    for _ in range(burn_in):
        step(ds.J_CRIT * (1.0 if beta_ratio is None else beta_ratio[0]))
    for k in range(n_frames):
        J = ds.J_CRIT * (1.0 if beta_ratio is None else float(beta_ratio[k]))
        for _ in range(sweeps_per_frame):
            x, y = step(J)
        a1, b1 = ds.current_omega(t[0], g, y, rng)
        a2, b2 = ds.current_omega(t[1], g, y, rng)
        omega1 = a1 ^ a2
        support = a1 | b1 | a2 | b2
        h, _, _ = ds.nesting(omega1, support, g, rng)
        fr = frame(L, omega1, support & ~omega1, h, scale=scale, mode=mode)
        if titulo or paneles:
            beta = 1.0 if beta_ratio is None else float(beta_ratio[k])
            try:
                texto = titulo.format(beta=beta, k=k + 1, n=n_frames) if titulo else None
            except (KeyError, IndexError, ValueError):
                texto = titulo                                   # braces that are not {beta}, {k} or {n}
            fr = titulo_frame(fr, texto, paneles)
        if p is None:
            Hh, Ww = fr.shape[0] - fr.shape[0] % 2, fr.shape[1] - fr.shape[1] % 2
            errfile = tempfile.TemporaryFile()
            p = subprocess.Popen([exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                  "-s", f"{Ww}x{Hh}", "-r", str(fps), "-i", "-"] + codec + [path],
                                 stdin=subprocess.PIPE, stderr=errfile)
        try:
            p.stdin.write(np.ascontiguousarray(fr[:Hh, :Ww]).tobytes())
        except (BrokenPipeError, OSError):
            p.wait()
            errfile.seek(0)
            raise RuntimeError(f"ffmpeg stopped at frame {k} ({exe}).  Its message:\n"
                               + errfile.read().decode(errors="replace")) from None
    p.stdin.close()
    p.wait()
    if p.returncode != 0:
        errfile.seek(0)
        raise RuntimeError(f"ffmpeg finished with code {p.returncode}:\n" + errfile.read().decode(errors="replace"))
    return path
