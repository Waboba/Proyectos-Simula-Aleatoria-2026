#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gráficos de las ligas de agentes de póker (Q-learning tabular).

Uso (desde la raíz del proyecto, junto a entrenar_liga.py y la carpeta Clases/):

    python graficos_liga.py resultados                       # todos los experimentos de la grilla
    python graficos_liga.py resultados/grid_w_0_25_nag_10_manos_m3   # un solo experimento
    python graficos_liga.py resultados --salida graficos --formato pdf --dpi 150
    python graficos_liga.py resultados --agentes 1,mejor,ultimo      # agentes para los heatmaps de política
    python graficos_liga.py resultados --sin-politica                # solo métricas (no carga los .npz)

Qué genera (carpeta --salida, por defecto "graficos/"):

  <experimento>/cadenas/       largo de cadena vs iteración, vs manos jugadas (episodios acumulados)
                               y vs umbral de winrate (curva fina recalculada desde enfrentamientos.csv)
  <experimento>/entrenamiento/ episodios y tiempo por agente, winrates, bb/mano, curvas de aprendizaje
  <experimento>/matrices/      heatmaps de las matrices de enfrentamiento y fuerza global por agente
  <experimento>/politica/      heatmaps de acciones por mano / fase / presión, matriz 13x13 preflop,
                               evolución del estilo de juego a lo largo de la liga
  _comparativa/                comparaciones entre experimentos (w, n_agentes, max_manos) + resumen_experimentos.csv

Nota sobre w (peso_fichas): en entorno.py la recompensa mixta es (1-w)*victoria + w*(fichas/fichas_iniciales),
es decir w=0 -> solo victoria (+1/-1) y w=1 -> solo fichas. (El texto de ayuda de entrenar_liga.py dice lo
contrario; los gráficos usan la definición del código.)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, str(Path(__file__).resolve().parent))

# --------------------------------------------------------------------------
# Dependencias del proyecto (con degradación elegante)
# --------------------------------------------------------------------------
try:
    from Clases.cadenas import adyacencia, cadena_estricta_mas_larga, matriz_probabilidad
    HAY_CADENAS = True
except Exception as _e:  # pragma: no cover
    HAY_CADENAS = False
    print(f"[aviso] No pude importar Clases.cadenas ({_e}); se omite la curva fina de cadenas.")

try:
    from Clases.abstraccion import (N_ESTADOS, N_FASES, N_FUERZA, N_INICIATIVA, N_POSICION,
                                    N_PRESION, N_PROYECTO, N_SPR, fuerza_preflop)
    HAY_ABSTRACCION = True
except Exception as _e:  # pragma: no cover
    HAY_ABSTRACCION = False
    N_FASES, N_FUERZA, N_PROYECTO, N_PRESION, N_POSICION, N_SPR, N_INICIATIVA = 4, 14, 3, 4, 2, 3, 3
    N_ESTADOS = N_FASES * N_FUERZA * N_PROYECTO * N_PRESION * N_POSICION * N_SPR * N_INICIATIVA
    print(f"[aviso] No pude importar Clases.abstraccion ({_e}); se omite la matriz 13x13 preflop.")

FMT = "png"
DPI = 130

ACCIONES = ("Retirarse", "Igualar/Pasar", "Subir ½ pozo", "Subir pozo", "All-in")
ACC_CORTO = ("F", "C", "R½", "R", "AI")
COLORES_ACC = ["#c0392b", "#95a5a6", "#82e0aa", "#1e8449", "#6c3483"]

ETQ_PRE = [f"Chen {i}" + (" (basura)" if i == 0 else " (premium)" if i == 11 else "") for i in range(12)]
ETQ_POST = ["Carta alta <J", "Carta alta ≥J", "Par de mesa", "Par medio/bajo", "Top pair k. débil",
            "Top pair k. bueno/Overpair", "Doble par", "Trío (trips)", "Set (oculto)", "Escalera",
            "Color", "Full house", "Póker", "Esc. de color"]
ETQ_PRESION = ["Sin apuesta", "≤40% pozo", "≤100% pozo", ">100% pozo"]
ETQ_INI = ("bote limpio", "bote subido", "rival apuesta")
NOMBRES_FASE = ("PREFLOP", "FLOP", "TURN", "RIVER")

UMBRALES_RES = (0.6, 0.7, 0.8, 0.9)  # los que guarda cadenas.csv
UMBRALES_FINOS = {
    "racha": np.round(np.arange(0.50, 0.951, 0.025), 3),
    "mano": np.round(np.arange(0.50, 0.801, 0.02), 3),
    "partida": np.round(np.arange(0.50, 0.801, 0.02), 3),
}
ETQ_W = "peso fichas w  (0 = solo victoria, 1 = solo fichas)"

DIMS = (N_FASES, N_FUERZA, N_PROYECTO, N_PRESION, N_POSICION, N_SPR, N_INICIATIVA)
FASE, FUERZA, PROY, PRES, POS, SPR, INI = np.unravel_index(np.arange(N_ESTADOS), DIMS)


# --------------------------------------------------------------------------
# Utilidades generales
# --------------------------------------------------------------------------
def guardar(fig, carpeta: Path, nombre: str) -> None:
    carpeta.mkdir(parents=True, exist_ok=True)
    fig.savefig(carpeta / f"{nombre}.{FMT}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def seguro(fn, *args, **kw) -> None:
    """Ejecuta un gráfico; si falla, avisa y sigue con el resto."""
    try:
        fn(*args, **kw)
    except Exception as e:  # noqa: BLE001
        print(f"   [aviso] {fn.__name__} falló: {e}")
        if kw.pop("_debug", False):
            traceback.print_exc()


def leer(ruta: Path) -> pd.DataFrame:
    return pd.read_csv(ruta) if ruta.exists() else pd.DataFrame()


def _color_texto(cm, v: float) -> str:
    r, g, b, _ = cm(min(max(v, 0.0), 1.0))
    return "white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.5 else "black"


def heat(ax, M, xl, yl, titulo, cmap="viridis", vmin=0.0, vmax=1.0, fmt="{:.0%}", fs=7, cbar=True):
    """Heatmap anotado; las celdas NaN salen en gris."""
    M = np.asarray(M, dtype=float)
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("#dddddd")
    im = ax.imshow(np.ma.masked_invalid(M), cmap=cm, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(xl)))
    ax.set_xticklabels(xl, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(yl)))
    ax.set_yticklabels(yl, fontsize=8)
    rango = (vmax - vmin) or 1.0
    for i, j in np.ndindex(M.shape):
        if np.isfinite(M[i, j]):
            ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=fs,
                    color=_color_texto(cm, (M[i, j] - vmin) / rango))
    if cbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    ax.set_title(titulo, fontsize=10)
    return im


# --------------------------------------------------------------------------
# Un experimento (una carpeta de liga)
# --------------------------------------------------------------------------
class Exp:
    def __init__(self, ruta: Path) -> None:
        self.ruta = ruta
        self.nombre = ruta.name
        man = json.loads((ruta / "manifest.json").read_text(encoding="utf-8"))
        self.cfg = man["config"]
        self.n_hechos = max([int(k) for k in man["agentes"]] + [0])
        self.n_nodos = self.n_hechos + 1
        self.w = float(self.cfg.get("peso_fichas", 0.5))
        self.max_manos = int(self.cfg.get("max_manos_partida") or 0)
        self.objetivo = float(self.cfg.get("objetivo", 0.6))
        self.criterio = self.cfg.get("criterio_victoria", "racha")
        self.k = int(self.cfg.get("k_racha", 3))
        self.it = leer(ruta / "iteraciones.csv")
        self.cad = leer(ruta / "cadenas.csv")
        self.enf = leer(ruta / "enfrentamientos.csv")
        self.fina: pd.DataFrame | None = None
        self.metricas_ultimo: dict = {}
        self.metricas_agentes: pd.DataFrame | None = None
        self._fz = None
        if not self.it.empty:
            self.it = self.it.sort_values("iteracion")

    @property
    def etiqueta(self) -> str:
        return f"w={self.w:g} · n={self.n_hechos} · max_manos={self.max_manos or '∞'}"

    def matrices(self):
        n = self.n_nodos
        G = np.zeros((n, n), dtype=np.int64)
        Gp = np.zeros((n, n), dtype=np.int64)
        for r in self.enf.itertuples(index=False):
            a, b = int(r.a), int(r.b)
            G[a, b], G[b, a] = r.ganadas_a, r.ganadas_b
            Gp[a, b], Gp[b, a] = r.partidas_ganadas_a, r.partidas_ganadas_b
        return G, Gp

    def episodios_acumulados(self) -> pd.Series:
        return self.it.set_index("iteracion")["episodios"].cumsum()

    def ruta_agente(self, m: int) -> Path:
        return self.ruta / "agentes" / f"agente_{m:02d}.npz"

    def fuerza_agentes(self) -> pd.DataFrame:
        """Fuerza de cada agente (1..n) según la matriz del criterio de la liga.
        `fuerza` = prob. media de ganarle a los OTROS agentes (si hay solo 1 agente, a todos)."""
        if getattr(self, "_fz", None) is not None:
            return self._fz
        nombre = {"racha": "matriz_racha", "mano": "matriz_mano", "partida": "matriz_partida"}.get(
            self.criterio, "matriz_racha")
        r = _cargar_matriz(self, nombre)
        if r is None:
            self._fz = pd.DataFrame()
            return self._fz
        M = r[0]
        filas = []
        for i in range(1, len(M)):
            vs_ag = float(np.nanmean(M[i, 1:]))
            fila = dict(agente=i, fuerza_vs_agentes=vs_ag, fuerza_vs_uniforme=float(M[i, 0]),
                        fuerza=vs_ag if np.isfinite(vs_ag) else float(np.nanmean(M[i])))
            if not self.enf.empty:
                r0 = self.enf[(self.enf.a == i) & (self.enf.b == 0)]
                r1 = self.enf[(self.enf.a == i) & (self.enf.b == i - 1)]
                fila["bb_vs_uniforme"] = float(r0.bb_por_mano_a.iloc[0]) if len(r0) else np.nan
                fila["bb_vs_anterior"] = float(r1.bb_por_mano_a.iloc[0]) if (i > 1 and len(r1)) else np.nan
            filas.append(fila)
        self._fz = pd.DataFrame(filas)
        return self._fz


# --------------------------------------------------------------------------
# Cadenas
# --------------------------------------------------------------------------
def calcular_curva_fina(e: Exp) -> pd.DataFrame | None:
    """Largo de la cadena estricta más larga para muchos umbrales (recalculado desde los conteos)."""
    if not HAY_CADENAS or e.enf.empty:
        return None
    G, Gp = e.matrices()
    base, crit = (Gp, "mano") if e.criterio == "partida" else (G, "racha" if e.criterio == "racha" else "mano")
    filas = []
    for cons in (0, 1):
        P = matriz_probabilidad(base, crit, e.k, bool(cons))
        for u in UMBRALES_FINOS[e.criterio]:
            cad = cadena_estricta_mas_larga(adyacencia(P, float(u)))
            filas.append({"conservador": cons, "umbral": float(u), "largo": len(cad),
                          "cadena": ">".join(str(x) for x in cad)})
    return pd.DataFrame(filas)


def curva_umbral(e: Exp, cons: int):
    """(umbrales, largos) de la iteración final; usa la curva fina si existe, si no cadenas.csv."""
    if e.fina is not None:
        d = e.fina[e.fina.conservador == cons].sort_values("umbral")
        return d.umbral.values, d.largo.values
    if e.cad.empty:
        return np.array([]), np.array([])
    ult = e.cad[(e.cad.iteracion == e.cad.iteracion.max()) & (e.cad.conservador == cons)].sort_values("umbral")
    return ult.umbral.values, ult.largo_nodos.values


def p_cadena_vs_iteracion(e: Exp, out: Path) -> None:
    if e.cad.empty:
        return
    umbrales = sorted(e.cad.umbral.unique())
    colores = plt.cm.viridis(np.linspace(0, 0.9, len(umbrales)))
    ep = e.episodios_acumulados()
    for x_tipo in ("iteracion", "episodios"):
        fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.3), sharey=True)
        for ax, cons, tit in zip(axs, (0, 1), ("Puntual", "Conservador (cota inferior Wilson 95%)")):
            for c, u in zip(colores, umbrales):
                d = e.cad[(e.cad.conservador == cons) & np.isclose(e.cad.umbral, u)].sort_values("iteracion")
                x = d.iteracion if x_tipo == "iteracion" else d.iteracion.map(ep) / 1e6
                ax.plot(x, d.largo_nodos, marker="o", ms=4, color=c, label=f"umbral {u:.2f}")
            ax.set_title(tit)
            ax.set_xlabel("Iteración (agentes entrenados)" if x_tipo == "iteracion"
                          else "Manos de entrenamiento acumuladas (millones)")
            ax.grid(alpha=0.3)
        axs[0].set_ylabel("Largo de la cadena estricta (nodos, incl. Uniforme)")
        axs[1].legend(fontsize=8, title="Winrate pedido")
        fig.suptitle(f"Largo de cadena — {e.etiqueta} · criterio {e.criterio}", y=1.02)
        guardar(fig, out / "cadenas", f"cadena_vs_{x_tipo}")


def p_cadena_vs_umbral_fino(e: Exp, out: Path) -> None:
    if not HAY_CADENAS or e.enf.empty:
        return
    G, Gp = e.matrices()
    fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, crit in zip(axs, ("racha", "mano", "partida")):
        base, c = (Gp, "mano") if crit == "partida" else (G, "racha" if crit == "racha" else "mano")
        for cons, ls, lab in ((0, "-", "puntual"), (1, "--", "conservador")):
            P = matriz_probabilidad(base, c, e.k, bool(cons))
            us = UMBRALES_FINOS[crit]
            ls_ = [len(cadena_estricta_mas_larga(adyacencia(P, float(u)))) for u in us]
            ax.plot(us, ls_, ls, marker="o", ms=3.5, label=lab)
        ax.set_title(f"criterio: {crit}" + (f" (k={e.k})" if crit == "racha" else ""))
        ax.set_xlabel("Winrate (probabilidad) exigido a cada par de la cadena")
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("Largo de cadena (nodos)")
    axs[0].legend()
    fig.suptitle(f"Cadena más larga vs. winrate pedido — {e.etiqueta}", y=1.03)
    guardar(fig, out / "cadenas", "cadena_vs_umbral_fino")


def p_transitividad(e: Exp, out: Path) -> None:
    if e.cad.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    umbrales = sorted(e.cad.umbral.unique())
    for c, u in zip(plt.cm.viridis(np.linspace(0, 0.9, len(umbrales))), umbrales):
        d = e.cad[(e.cad.conservador == 0) & np.isclose(e.cad.umbral, u)].sort_values("iteracion")
        ax.plot(d.iteracion, d.frac_transitiva, marker="o", ms=4, color=c, label=f"umbral {u:.2f}")
    ax.set_xlabel("Iteración")
    ax.set_ylabel("Fracción de caminos a→b→c con a→c")
    ax.set_title(f"Transitividad de la relación 'le gana' — {e.etiqueta}")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    guardar(fig, out / "cadenas", "transitividad")


# --------------------------------------------------------------------------
# Entrenamiento y enfrentamientos
# --------------------------------------------------------------------------
def p_entrenamiento(e: Exp, out: Path) -> None:
    it = e.it
    if it.empty:
        return
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.2))
    cols = ["#2e8b57" if a == 1 else "#c0392b" for a in it.alcanzo_objetivo]
    axs[0].bar(it.iteracion, it.episodios / 1e3, color=cols)
    axs[0].set_ylabel("Manos de entrenamiento (miles)")
    axs[0].set_title("Manos hasta congelar\n(verde = alcanzó objetivo, rojo = tope de episodios)")
    axs[1].bar(it.iteracion, it.tiempo_s / 60, color="#4a6fa5")
    axs[1].set_ylabel("Minutos")
    axs[1].set_title("Tiempo de entrenamiento por agente")
    cum = it.episodios.cumsum() / 1e6
    axs[2].plot(it.iteracion, cum, marker="o", color="#8e44ad")
    axs[2].set_ylabel("Millones de manos")
    axs[2].set_title("Manos acumuladas")
    for ax in axs:
        ax.set_xlabel("Agente")
        ax.grid(alpha=0.3)
    fig.suptitle(e.etiqueta, y=1.03)
    guardar(fig, out / "entrenamiento", "episodios_y_tiempo")


def p_winrates(e: Exp, out: Path) -> None:
    it = e.it
    if it.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4.8))
    series = (("winrate_vs_anterior", "Agente m vs m-1 (al congelar)", "#c0392b"),
              ("winrate_vs_uniforme", "Agente m vs Uniforme (al congelar)", "#2980b9"),
              ("winrate_mezcla_parada", "Mezcla de rivales (parada)", "#27ae60"),
              ("winrate_min_parada", "Peor rival (parada)", "#8e44ad"))
    for col, lab, c in series:
        if col in it:
            ax.plot(it.iteracion, it[col], marker="o", ms=4, label=lab, color=c)
    ax.axhline(e.objetivo, ls="--", color="gray", label=f"objetivo {e.objetivo:.2f}")
    ax.axhline(0.5, ls=":", color="k", alpha=0.5)
    ax.set_xlabel("Agente")
    ax.set_ylabel(f"Valor del criterio ({e.criterio}" + (f", k={e.k})" if e.criterio == "racha" else ")"))
    ax.set_title(f"Winrates por agente — {e.etiqueta}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    guardar(fig, out / "entrenamiento", "winrates_por_agente")


def p_bb(e: Exp, out: Path) -> None:
    enf = e.enf
    if enf.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ancho = 0.38
    for desp, b_fun, lab, c in ((-ancho / 2, lambda m: m - 1, "vs agente anterior", "#c0392b"),
                                (ancho / 2, lambda m: 0, "vs Uniforme", "#2980b9")):
        xs, ys, es = [], [], []
        for m in range(1, e.n_hechos + 1):
            if m == 1 and lab.startswith("vs agente"):
                continue
            r = enf[(enf.a == m) & (enf.b == b_fun(m))]
            if len(r):
                xs.append(m + desp)
                ys.append(float(r.bb_por_mano_a.iloc[0]))
                es.append(float(r.bb_ic95.iloc[0]))
        ax.bar(xs, ys, width=ancho, yerr=es, capsize=2, label=lab, color=c, alpha=0.85)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Agente m")
    ax.set_ylabel("Ganancia media (bb por mano) ± IC95")
    ax.set_title(f"Ganancia en ciegas grandes por mano — {e.etiqueta}")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    guardar(fig, out / "entrenamiento", "bb_por_mano")


def p_curvas_entrenamiento(e: Exp, out: Path) -> None:
    n = e.n_hechos
    if n == 0:
        return
    cols = min(5, n)
    filas = int(np.ceil(n / cols))
    fig, axs = plt.subplots(filas, cols, figsize=(3.6 * cols, 2.9 * filas), squeeze=False, sharey=True)
    for m in range(1, n + 1):
        ax = axs[(m - 1) // cols][(m - 1) % cols]
        f = e.ruta / "entrenamiento" / f"agente_{m:02d}" / "metricas.csv"
        if f.exists():
            d = pd.read_csv(f)
            for j in range(m):
                c = f"valor_vs_{j}"
                if c in d:
                    ax.plot(d.episodio / 1e3, d[c], lw=1, alpha=0.8,
                            color="black" if j == 0 else plt.cm.tab20(j % 20))
            if "valor_mezcla" in d:
                ax.plot(d.episodio / 1e3, d.valor_mezcla, lw=2, color="crimson")
        ax.axhline(e.objetivo, ls="--", color="gray")
        ax.set_title(f"Agente {m}", fontsize=9)
        ax.grid(alpha=0.3)
        if m > (filas - 1) * cols:
            ax.set_xlabel("Manos (miles)", fontsize=8)
    for k in range(n, filas * cols):
        axs[k // cols][k % cols].axis("off")
    fig.suptitle(f"Curvas de aprendizaje (negro = vs Uniforme, rojo = mezcla, gris = objetivo) — {e.etiqueta}",
                 y=1.01)
    guardar(fig, out / "entrenamiento", "curvas_aprendizaje")


def _cargar_matriz(e: Exp, nombre: str):
    f = e.ruta / f"{nombre}.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f, index_col=0)
    M = df.values.astype(float)
    np.fill_diagonal(M, np.nan)
    return M, list(df.columns), list(df.index)


def p_matrices(e: Exp, out: Path) -> None:
    desc = {"matriz_mano": "por mano", "matriz_partida": f"por partida (max {e.max_manos or '∞'} manos)",
            "matriz_racha": f"por racha (k={e.k})"}
    for nombre, d in desc.items():
        r = _cargar_matriz(e, nombre)
        if r is None:
            continue
        M, xl, yl = r
        fig, ax = plt.subplots(figsize=(0.62 * len(xl) + 3, 0.55 * len(yl) + 2.5))
        heat(ax, M, xl, yl, f"P(fila le gana a columna) — {d}\n{e.etiqueta}", cmap="RdBu",
             vmin=0.25, vmax=0.75, fmt="{:.2f}", fs=7)
        ax.set_xlabel("Rival")
        ax.set_ylabel("Agente")
        guardar(fig, out / "matrices", nombre)


def p_fuerza_global(e: Exp, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    hay = False
    for nombre, lab, c in (("matriz_mano", "por mano", "#2980b9"), ("matriz_partida", "por partida", "#c0392b"),
                           ("matriz_racha", f"por racha (k={e.k})", "#27ae60")):
        r = _cargar_matriz(e, nombre)
        if r is None:
            continue
        M = r[0]
        ax.plot(range(len(M)), np.nanmean(M, axis=1), marker="o", label=lab, color=c)
        hay = True
    if not hay:
        plt.close(fig)
        return
    ax.axhline(0.5, ls=":", color="k", alpha=0.5)
    ax.set_xticks(range(e.n_nodos))
    ax.set_xticklabels(["Unif"] + [f"A{i}" for i in range(1, e.n_nodos)])
    ax.set_ylabel("Probabilidad media de ganarle a los demás")
    ax.set_title(f"Fuerza global de cada nodo — {e.etiqueta}")
    ax.grid(alpha=0.3)
    ax.legend()
    guardar(fig, out / "matrices", "fuerza_global")


# --------------------------------------------------------------------------
# Política: heatmaps de acciones
# --------------------------------------------------------------------------
class Politica:
    """Política greedy de una tabla Q: sólo se consideran acciones realmente visitadas
    (así se evita elegir una acción ilegal que nunca se probó)."""

    def __init__(self, ruta: Path) -> None:
        with np.load(ruta, allow_pickle=False) as d:
            Q = d["Q"].astype(np.float64)
            V = d["visitas"].astype(np.float64)
        if Q.shape[0] != N_ESTADOS:
            raise ValueError(f"{ruta.name}: la tabla Q tiene {Q.shape[0]} estados y la abstracción actual "
                             f"tiene {N_ESTADOS} (liga entrenada con otra versión de abstraccion.py)")
        visit = V > 0
        Qm = np.where(visit, Q, -np.inf)
        hay = visit.any(axis=1)
        self.a = np.where(hay, Qm.argmax(axis=1), -1)
        self.w = V.sum(axis=1)
        self.qmax = np.where(hay, Qm.max(axis=1), np.nan)


def shares(p: Politica, mask: np.ndarray) -> np.ndarray:
    """Fracción (ponderada por visitas) de cada acción greedy dentro de los estados de `mask`."""
    m = mask & (p.a >= 0)
    tot = p.w[m].sum()
    if tot <= 0:
        return np.full(5, np.nan)
    return np.array([p.w[m & (p.a == k)].sum() for k in range(5)]) / tot


def tabla_acciones_por_fuerza(p: Politica, fase: int, n_f: int) -> np.ndarray:
    return np.array([shares(p, (FASE == fase) & (FUERZA == f)) for f in range(n_f)])


def etiquetas_fuerza(fase: int):
    return ETQ_PRE if fase == 0 else ETQ_POST


def p_acciones_por_fuerza(p: Politica, nombre: str, out: Path) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    for fase, ax in enumerate(axs.ravel()):
        n_f = 12 if fase == 0 else N_FUERZA
        M = tabla_acciones_por_fuerza(p, fase, n_f)
        heat(ax, M, ACCIONES, etiquetas_fuerza(fase), NOMBRES_FASE[fase], cmap="viridis", vmin=0, vmax=1)
    fig.suptitle(f"{nombre}: % de estados (ponderado por visitas) donde la acción greedy es cada una, "
                 "según la fuerza de la mano", y=0.94)
    guardar(fig, out / "politica", f"{nombre}_acciones_por_fuerza")


def p_accion_modal(p: Politica, nombre: str, out: Path) -> None:
    """Acción más frecuente por (fuerza, presión) en cada fase; se muestra la acción y su %."""
    fig, axs = plt.subplots(1, 4, figsize=(19, 7))
    cmap = ListedColormap(COLORES_ACC).copy()
    cmap.set_bad("#dddddd")
    for fase, ax in enumerate(axs):
        n_f = 12 if fase == 0 else N_FUERZA
        A = np.full((n_f, N_PRESION), np.nan)
        S = np.full((n_f, N_PRESION), np.nan)
        for f in range(n_f):
            for pr in range(N_PRESION):
                s = shares(p, (FASE == fase) & (FUERZA == f) & (PRES == pr))
                if np.isfinite(s).all():
                    A[f, pr], S[f, pr] = s.argmax(), s.max()
        ax.imshow(np.ma.masked_invalid(A), cmap=cmap, vmin=-0.5, vmax=4.5, aspect="auto")
        for i, j in np.ndindex(A.shape):
            if np.isfinite(A[i, j]):
                ax.text(j, i, f"{ACC_CORTO[int(A[i, j])]}\n{S[i, j]:.0%}", ha="center", va="center", fontsize=7)
        ax.set_xticks(range(N_PRESION))
        ax.set_xticklabels(ETQ_PRESION, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n_f))
        ax.set_yticklabels(etiquetas_fuerza(fase), fontsize=8)
        ax.set_title(NOMBRES_FASE[fase])
    axs[0].legend(handles=[Patch(color=c, label=a) for c, a in zip(COLORES_ACC, ACCIONES)],
                  loc="upper center", bbox_to_anchor=(2.4, -0.18), ncol=5, fontsize=9)
    fig.suptitle(f"{nombre}: acción dominante según fuerza de la mano y presión del rival (celda = acción y % de estados)",
                 y=1.0)
    guardar(fig, out / "politica", f"{nombre}_accion_modal")


GRID_RANGOS = list(range(14, 1, -1))
GRID_TXT = {14: "A", 13: "K", 12: "Q", 11: "J", 10: "T"}
CONTEXTOS_PREFLOP = (
    ("SB (dealer) debe igualar la ciega", dict(pos=1, pres=2, spr=2, ini=2)),
    ("BB con opción (rival igualó)", dict(pos=0, pres=0, spr=2, ini=0)),
    ("BB enfrenta una subida", dict(pos=0, pres=2, spr=2, ini=2)),
)


def grilla_preflop(p: Politica, ctx: dict) -> np.ndarray:
    M = np.full((13, 13), np.nan)
    for i, r1 in enumerate(GRID_RANGOS):
        for j, r2 in enumerate(GRID_RANGOS):
            hi, lo = max(r1, r2), min(r1, r2)
            c1 = (hi - 2) * 4
            c2 = (lo - 2) * 4 + (0 if i < j else 1)  # i<j: suited · i>j: offsuit · i==j: par
            g = fuerza_preflop(c1, c2)
            s = int(np.ravel_multi_index((0, g, 0, ctx["pres"], ctx["pos"], ctx["spr"], ctx["ini"]), DIMS))
            if p.a[s] >= 0:
                M[i, j] = p.a[s]
    return M


def p_manos_preflop(p: Politica, nombre: str, out: Path) -> None:
    if not HAY_ABSTRACCION:
        return
    fig, axs = plt.subplots(1, 3, figsize=(19, 6.6))
    cmap = ListedColormap(COLORES_ACC).copy()
    cmap.set_bad("#dddddd")
    etq = [GRID_TXT.get(r, str(r)) for r in GRID_RANGOS]
    for ax, (tit, ctx) in zip(axs, CONTEXTOS_PREFLOP):
        M = grilla_preflop(p, ctx)
        ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=-0.5, vmax=4.5)
        for i, j in np.ndindex(M.shape):
            if np.isfinite(M[i, j]):
                suf = "" if i == j else ("s" if i < j else "o")
                ax.text(j, i, f"{etq[min(i, j)]}{etq[max(i, j)]}{suf}\n{ACC_CORTO[int(M[i, j])]}",
                        ha="center", va="center", fontsize=6.5)
        ax.set_xticks(range(13))
        ax.set_xticklabels(etq)
        ax.set_yticks(range(13))
        ax.set_yticklabels(etq)
        ax.set_title(tit)
    axs[1].legend(handles=[Patch(color=c, label=a) for c, a in zip(COLORES_ACC, ACCIONES)],
                  loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=5)
    fig.suptitle(f"{nombre}: acción preflop por mano inicial (arriba de la diagonal = suited 's', abajo = offsuit 'o'). "
                 "Manos con igual puntaje de Chen comparten celda de decisión.", y=1.0)
    guardar(fig, out / "politica", f"{nombre}_manos_preflop")


def p_agresion_contexto(p: Politica, nombre: str, out: Path) -> None:
    post = FASE > 0
    fold = np.array([shares(p, post & (FUERZA == f) & (PRES == pr))[0] for f in range(N_FUERZA)
                     for pr in range(1, N_PRESION)]).reshape(N_FUERZA, N_PRESION - 1)
    combos = [(pos, ini) for pos in (0, 1) for ini in range(3)]
    agr = np.array([[shares(p, post & (FUERZA == f) & (POS == pos) & (INI == ini))[2:].sum()
                     for pos, ini in combos] for f in range(N_FUERZA)])
    fig, axs = plt.subplots(1, 2, figsize=(14, 6.5))
    heat(axs[0], fold, ETQ_PRESION[1:], ETQ_POST, "% de retiros al enfrentar apuesta (postflop)", cmap="Reds")
    heat(axs[1], agr, [f"{'Dealer' if pos else 'No-dealer'}\n{ETQ_INI[ini]}" for pos, ini in combos], ETQ_POST,
         "% de subidas (½ pozo, pozo o all-in) (postflop)", cmap="Greens")
    fig.suptitle(f"{nombre}: cuándo se retira y cuándo agrede", y=1.02)
    guardar(fig, out / "politica", f"{nombre}_retiro_y_agresion")


def p_valor_q(p: Politica, nombre: str, out: Path) -> None:
    M = np.full((N_FUERZA, N_FASES), np.nan)
    for fase in range(N_FASES):
        for f in range(N_FUERZA):
            m = (FASE == fase) & (FUERZA == f) & (p.a >= 0)
            if p.w[m].sum() > 0:
                M[f, fase] = np.average(p.qmax[m], weights=p.w[m])
    lim = np.nanmax(np.abs(M)) if np.isfinite(M).any() else 1.0
    fig, ax = plt.subplots(figsize=(7, 7))
    heat(ax, M, NOMBRES_FASE, ETQ_POST, f"{nombre}: valor Q máximo medio (esperanza de recompensa)",
         cmap="RdYlGn", vmin=-lim, vmax=lim, fmt="{:+.3f}")
    guardar(fig, out / "politica", f"{nombre}_valor_q")


def graficos_agente(e: Exp, m: int, out: Path) -> None:
    ruta = e.ruta_agente(m)
    if not ruta.exists():
        return
    p = Politica(ruta)
    nombre = f"agente_{m:02d}"
    for fn in (p_acciones_por_fuerza, p_accion_modal, p_manos_preflop, p_agresion_contexto, p_valor_q):
        seguro(fn, p, nombre, out)


# --------------------------------------------------------------------------
# Política: evolución a lo largo de la liga
# --------------------------------------------------------------------------
BUCKETS_POST = (("débil", (FASE > 0) & (FUERZA <= 1)), ("media", (FASE > 0) & (FUERZA >= 2) & (FUERZA <= 5)),
                ("fuerte", (FASE > 0) & (FUERZA >= 6)))
BUCKETS_PRE = (("débil", (FASE == 0) & (FUERZA <= 3)), ("media", (FASE == 0) & (FUERZA >= 4) & (FUERZA <= 7)),
               ("fuerte", (FASE == 0) & (FUERZA >= 8)))


def metricas_politica(p: Politica) -> dict:
    d = {}
    for nom, mask in (("global", np.ones(N_ESTADOS, bool)), ("pre", FASE == 0), ("post", FASE > 0)):
        s = shares(p, mask)
        for k in range(5):
            d[f"{nom}_{k}"] = s[k]
        d[f"agresion_{nom}"] = s[2:].sum()
    for nom, buckets in (("pre", BUCKETS_PRE), ("post", BUCKETS_POST)):
        for et, mask in buckets:
            d[f"raise_{nom}_{et}"] = shares(p, mask)[2:].sum()
            d[f"fold_{nom}_{et}"] = shares(p, mask & (PRES > 0))[0]
    d["farol_river"] = shares(p, (FASE == 3) & (PROY == 2) & (FUERZA <= 1))[2:].sum()
    for k in range(5):  # cantidad de estados visitados donde la acción greedy es k
        d[f"n_{k}"] = int(((p.a == k)).sum())
    return d


def p_evolucion_politica(e: Exp, out: Path) -> None:
    filas, prev = [], None
    for m in range(1, e.n_hechos + 1):
        if not e.ruta_agente(m).exists():
            continue
        p = Politica(e.ruta_agente(m))
        d = metricas_politica(p)
        d["agente"] = m
        if prev is not None:
            ok = (prev.a >= 0) & (p.a >= 0)
            d["cambio_politica"] = p.w[ok & (p.a != prev.a)].sum() / max(p.w[ok].sum(), 1e-12)
        else:
            d["cambio_politica"] = np.nan
        filas.append(d)
        prev = p
        if m == e.n_hechos:
            e.metricas_ultimo = d
    if not filas:
        return
    df = pd.DataFrame(filas)
    e.metricas_agentes = df
    (out / "politica").mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "politica" / "metricas_politica_por_agente.csv", index=False)

    # 1) reparto de acciones
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.5), sharey=True)
    for ax, nom, tit in zip(axs, ("global", "pre", "post"), ("Global", "Preflop", "Postflop")):
        base = np.zeros(len(df))
        for k in range(5):
            v = df[f"{nom}_{k}"].fillna(0).values
            ax.bar(df.agente, v, bottom=base, color=COLORES_ACC[k], label=ACCIONES[k])
            base += v
        ax.set_title(tit)
        ax.set_xlabel("Agente")
        ax.grid(alpha=0.3, axis="y")
    axs[0].set_ylabel("Fracción de estados (ponderada por visitas)")
    axs[2].legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle(f"Reparto de acciones greedy por agente — {e.etiqueta}", y=1.03)
    guardar(fig, out / "politica", "evolucion_reparto_acciones")

    # 2) agresividad y retiros por fuerza
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    cb = {"débil": "#c0392b", "media": "#f39c12", "fuerte": "#27ae60"}
    for (r, c), (pref, nom, tit) in zip(((0, 0), (0, 1), (1, 0), (1, 1)),
                                         (("raise", "pre", "% subidas — preflop"),
                                          ("raise", "post", "% subidas — postflop"),
                                          ("fold", "pre", "% retiros ante apuesta — preflop"),
                                          ("fold", "post", "% retiros ante apuesta — postflop"))):
        ax = axs[r][c]
        for et in ("débil", "media", "fuerte"):
            ax.plot(df.agente, df[f"{pref}_{nom}_{et}"], marker="o", ms=4, color=cb[et], label=f"mano {et}")
        ax.set_title(tit)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
    axs[0][0].legend(fontsize=8)
    for ax in axs[1]:
        ax.set_xlabel("Agente")
    fig.suptitle(f"Estilo de juego por fuerza de mano — {e.etiqueta}", y=1.0)
    guardar(fig, out / "politica", "evolucion_agresividad_por_fuerza")

    # 3) faroles y cambio de política
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    axs[0].plot(df.agente, df.farol_river, marker="o", color="#8e44ad")
    axs[0].set_title("Faroleo en river: % subidas con proyecto fallido (carta alta)")
    axs[1].plot(df.agente, df.cambio_politica, marker="o", color="#2c3e50")
    axs[1].set_title("% de estados donde cambió la acción vs. agente anterior")
    for ax in axs:
        ax.set_xlabel("Agente")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
    fig.suptitle(e.etiqueta, y=1.03)
    guardar(fig, out / "politica", "evolucion_farol_y_cambio")


def elegir_agentes(e: Exp, spec: str) -> list[int]:
    res = []
    for tok in [t.strip() for t in spec.split(",") if t.strip()]:
        if tok == "ultimo":
            res.append(e.n_hechos)
        elif tok == "mejor":
            fz = e.fuerza_agentes()
            if not fz.empty:
                res.append(int(fz.sort_values("fuerza", ascending=False).iloc[0].agente))
        elif tok.isdigit() and 1 <= int(tok) <= e.n_hechos:
            res.append(int(tok))
    vistos = []
    for m in res:
        if m not in vistos:
            vistos.append(m)
    return vistos


# --------------------------------------------------------------------------
# Comparativa entre experimentos
# --------------------------------------------------------------------------
def construir_resumen(exps: list[Exp]) -> pd.DataFrame:
    filas = []
    for e in exps:
        it = e.it
        f = dict(exp=e.nombre, w=e.w, n_ag=e.n_hechos, max_manos=e.max_manos, objetivo=e.objetivo,
                 criterio=e.criterio)
        if not it.empty:
            f["episodios_total"] = it.episodios.sum()
            f["frac_alcanzo"] = it.alcanzo_objetivo.mean()
            f["tiempo_h"] = it.tiempo_s.sum() / 3600
            f["wr_vs_anterior"] = it.winrate_vs_anterior.mean()
            f["wr_vs_uniforme"] = it.winrate_vs_uniforme.mean()
            f["wr_min_parada"] = it.winrate_min_parada.mean()
            f["bb_vs_anterior"] = it.bb_vs_anterior.mean()
            f["transitividad"] = it.frac_transitiva.mean()
        if not e.cad.empty:
            last = e.cad[e.cad.iteracion == e.cad.iteracion.max()]
            for cons, suf in ((0, "p"), (1, "c")):
                for u in UMBRALES_RES:
                    r = last[(last.conservador == cons) & np.isclose(last.umbral, u)]
                    f[f"largo_{suf}_{int(round(u * 100))}"] = r.largo_nodos.iloc[0] if len(r) else np.nan
        for k, v in e.metricas_ultimo.items():
            f[f"ult_{k}"] = v
        filas.append(f)
    return pd.DataFrame(filas)


def fig_facetas(df, x, filas, linea, col, titulo, nombre, out, ylim=None, etq_x=None):
    """Grilla de subplots: una fila por métrica, una columna por valor de `col`, una línea por valor de `linea`."""
    cols = sorted(df[col].unique())
    lineas = sorted(df[linea].unique())
    fig, axs = plt.subplots(len(filas), len(cols), figsize=(4.3 * len(cols) + 1.2, 3.2 * len(filas) + 0.8),
                            squeeze=False, sharex=True)
    colores = plt.cm.plasma(np.linspace(0.05, 0.85, len(lineas)))
    marcas = "osD^vP*X"
    for r, (etq, y) in enumerate(filas):
        if y not in df:
            continue
        for c, cv in enumerate(cols):
            ax = axs[r][c]
            sub = df[df[col] == cv]
            for i, (cl, lv) in enumerate(zip(colores, lineas)):
                d = sub[sub[linea] == lv].groupby(x)[y].mean().sort_index()
                if d.empty:
                    continue
                ax.plot(d.index, d.values, marker=marcas[i % len(marcas)], color=cl, alpha=0.85,
                        label=f"{linea}={lv:g}" if linea != "max_manos" else f"max_manos={'∞' if lv == 0 else int(lv)}")
            if r == 0:
                ax.set_title(f"{col}={cv:g}")
            if c == 0:
                ax.set_ylabel(etq, fontsize=9)
            if r == len(filas) - 1:
                ax.set_xlabel(etq_x or x)
            if ylim:
                ax.set_ylim(*ylim)
            ax.grid(alpha=0.3)
    h, l = {}, {}
    for ax in axs.ravel():
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            l[ll] = hh
    if l:
        fig.legend(l.values(), l.keys(), loc="center right", bbox_to_anchor=(1.0, 0.5), fontsize=9)
    fig.suptitle(titulo, y=1.0)
    fig.tight_layout(rect=(0, 0, 0.92, 0.97))
    guardar(fig, out, nombre)


def c_largo_vs_w(df, out):
    for suf, tit in (("p", "puntual"), ("c", "conservador")):
        filas = [(f"umbral {u:.1f}\nlargo de cadena", f"largo_{suf}_{int(round(u * 100))}") for u in UMBRALES_RES]
        fig_facetas(df, "w", filas, "max_manos", "n_ag",
                    f"Largo de la cadena estricta vs. w ({tit}) — cada fila = winrate pedido", f"largo_cadena_vs_w_{tit}",
                    out, etq_x=ETQ_W)


def c_heatmap_largo(df, out):
    ns = sorted(df.n_ag.unique())
    for u in UMBRALES_RES:
        fig, axs = plt.subplots(2, len(ns), figsize=(4.6 * len(ns), 7.2), squeeze=False)
        for r, (suf, tit) in enumerate((("p", "puntual"), ("c", "conservador"))):
            col = f"largo_{suf}_{int(round(u * 100))}"
            if col not in df:
                continue
            vmax = max(2.0, float(np.nanmax(df[col])))
            for c, n in enumerate(ns):
                pv = df[df.n_ag == n].pivot_table(index="max_manos", columns="w", values=col, aggfunc="mean")
                yl = ["∞" if v == 0 else f"{int(v)}" for v in pv.index]
                heat(axs[r][c], pv.values, [f"{v:g}" for v in pv.columns], yl, f"{tit} · n_agentes={n}",
                     cmap="YlGnBu", vmin=1, vmax=vmax, fmt="{:.1f}", fs=9)
                axs[r][c].set_xlabel("w (0 = victoria, 1 = fichas)")
                axs[r][c].set_ylabel("max_manos por partida")
        fig.suptitle(f"Largo de cadena (umbral {u:.1f}) según w y max_manos", y=1.0)
        fig.tight_layout()
        guardar(fig, out, f"heatmap_largo_cadena_umbral{int(round(u * 100))}")


def c_curvas_umbral(exps, out):
    ms = sorted({e.max_manos for e in exps})
    ns = sorted({e.n_hechos for e in exps})
    ws = sorted({e.w for e in exps})
    colores = dict(zip(ws, plt.cm.plasma(np.linspace(0.05, 0.85, len(ws)))))
    for cons, tit in ((0, "puntual"), (1, "conservador")):
        fig, axs = plt.subplots(len(ms), len(ns), figsize=(4.4 * len(ns) + 1.2, 3.2 * len(ms) + 0.8),
                                squeeze=False, sharex=True)
        for e in exps:
            us, ls = curva_umbral(e, cons)
            if len(us) == 0:
                continue
            ax = axs[ms.index(e.max_manos)][ns.index(e.n_hechos)]
            ax.plot(us, ls, marker="o", ms=3, color=colores[e.w], alpha=0.85, label=f"w={e.w:g}")
        for r, m in enumerate(ms):
            axs[r][0].set_ylabel(f"max_manos={'∞' if m == 0 else m}\nlargo de cadena")
        for c, n in enumerate(ns):
            axs[0][c].set_title(f"n_agentes={n}")
            axs[-1][c].set_xlabel("Winrate pedido (umbral)")
        for ax in axs.ravel():
            ax.grid(alpha=0.3)
        h, l = {}, {}
        for ax in axs.ravel():
            for hh, ll in zip(*ax.get_legend_handles_labels()):
                l[ll] = hh
        fig.legend(l.values(), l.keys(), loc="center right", bbox_to_anchor=(1.0, 0.5))
        fig.suptitle(f"Largo de cadena vs. winrate pedido ({tit}), una curva por w", y=1.0)
        fig.tight_layout(rect=(0, 0, 0.93, 0.97))
        guardar(fig, out, f"largo_cadena_vs_umbral_{tit}")


def c_scatter(df, out):
    col = "largo_p_60"
    if col not in df or "episodios_total" not in df:
        return
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    rng = np.random.default_rng(0)
    marcas = dict(zip(sorted(df.n_ag.unique()), "osD^v"))
    for ax, y, tit in ((axs[0], col, "largo de cadena (umbral 0.6)"), (axs[1], "wr_vs_anterior", "winrate vs agente anterior")):
        if y not in df:
            continue
        for n, g in df.groupby("n_ag"):
            im = ax.scatter(g.episodios_total / 1e6, g[y] + (rng.normal(0, 0.03, len(g)) if y == col else 0),
                            c=g.w, cmap="plasma", vmin=0, vmax=1, marker=marcas[n], s=60, edgecolor="k",
                            label=f"n_agentes={n}")
        ax.set_xlabel("Manos de entrenamiento totales (millones)")
        ax.set_ylabel(tit)
        ax.grid(alpha=0.3)
    plt.colorbar(im, ax=axs, label="w")
    axs[0].legend(fontsize=8)
    guardar(fig, out, "scatter_manos_vs_resultados")


def c_estilo(df, out):
    filas = [("% subidas postflop", "ult_agresion_post"), ("% subidas preflop", "ult_agresion_pre"),
             ("% retiros ante apuesta\n(mano débil, postflop)", "ult_fold_post_débil"),
             ("% subidas con mano fuerte\n(postflop)", "ult_raise_post_fuerte"),
             ("Faroleo en river", "ult_farol_river")]
    filas = [f for f in filas if f[1] in df]
    if not filas:
        return
    fig_facetas(df, "w", filas, "max_manos", "n_ag", "Estilo de juego del último agente vs. w", "estilo_ultimo_agente_vs_w",
                out, etq_x=ETQ_W, ylim=(-0.02, 1.02))


def comparativa(exps: list[Exp], out: Path) -> None:
    df = construir_resumen(exps)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "resumen_experimentos.csv", index=False)
    print(f"   resumen_experimentos.csv ({len(df)} experimentos)")
    c_largo_vs_w(df, out)
    c_heatmap_largo(df, out)
    c_curvas_umbral(exps, out)
    c_scatter(df, out)
    c_estilo(df, out)
    fig_facetas(df, "w", [("Manos de entrenamiento\ntotales", "episodios_total"),
                          ("Fracción de agentes que\nalcanzó el objetivo", "frac_alcanzo"),
                          ("Tiempo total (horas)", "tiempo_h")], "max_manos", "n_ag",
                "Costo de entrenamiento vs. w", "entrenamiento_vs_w", out, etq_x=ETQ_W)
    fig_facetas(df, "w", [("Winrate vs agente\nanterior", "wr_vs_anterior"), ("Winrate vs Uniforme", "wr_vs_uniforme"),
                          ("Peor rival al parar", "wr_min_parada"), ("bb/mano vs anterior", "bb_vs_anterior"),
                          ("Transitividad", "transitividad")], "max_manos", "n_ag",
                "Calidad de los agentes vs. w (promedio sobre la liga)", "calidad_vs_w", out, etq_x=ETQ_W)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
ACC_LARGAS = ("Fold (Retirarse)", "Call / Check (Igualar/Pasar)", "Raise ½ pozo (Subir ½ pozo)",
              "Raise pozo (Subir pozo)", "All-in")
UMBRAL_MIN_TABLA = 0.55


def _nombre_nodo(i: int) -> str:
    return "Uniforme" if i == 0 else f"Agente {i}"


def _idx_nombre(s: str) -> int:
    s = s.strip()
    return 0 if s == "Uniforme" else int(s.split()[-1])


def cadenas_exp(e: Exp) -> list[tuple[int, float, list[int]]]:
    """(conservador, umbral, cadena como lista de nodos) de la liga: curva fina si existe, si no cadenas.csv."""
    res = []
    if e.fina is not None and "cadena" in e.fina:
        for r in e.fina.itertuples(index=False):
            res.append((int(r.conservador), float(r.umbral), [int(x) for x in str(r.cadena).split(">") if x != ""]))
    elif not e.cad.empty:
        ult = e.cad[e.cad.iteracion == e.cad.iteracion.max()]
        for r in ult.itertuples(index=False):
            res.append((int(r.conservador), float(r.umbral), [_idx_nombre(x) for x in str(r.cadena).split(">")]))
    return res


def construir_tablas(exps: list[Exp], umbral_min: float = UMBRAL_MIN_TABLA):
    """(tabla de cadenas con su mejor agente, ranking de todos los agentes)."""
    filas, ranking = [], []
    for e in exps:
        fz = e.fuerza_agentes()
        if fz.empty:
            continue
        fzi = fz.set_index("agente")
        mai = e.metricas_agentes.set_index("agente") if e.metricas_agentes is not None else None
        mejor_liga = int(fz.sort_values("fuerza", ascending=False).iloc[0].agente)
        cadenas = cadenas_exp(e)

        def estrategia(m: int) -> dict:
            if mai is None or m not in mai.index:
                return {}
            sh = [float(mai.loc[m, f"global_{k}"]) for k in range(5)]
            if not np.isfinite(sh).all():
                return {}
            return {"accion_dominante": ACCIONES[int(np.argmax(sh))], "pct_retiro": sh[0], "pct_igualar": sh[1],
                    "pct_subidas": sum(sh[2:]), "pct_allin": sh[4]}

        def metricas(m: int) -> dict:
            f = fzi.loc[m]
            d = {"fuerza": f["fuerza"], "fuerza_vs_agentes": f["fuerza_vs_agentes"],
                 "fuerza_vs_uniforme": f["fuerza_vs_uniforme"]}
            for c in ("bb_vs_uniforme", "bb_vs_anterior"):
                if c in f:
                    d[c] = f[c]
            d.update(estrategia(m))
            return d

        en_cadena = set()
        for cons, u, c in cadenas:
            ags = [x for x in c if x != 0]
            if cons == 0 and u >= umbral_min and len(ags) >= 2:
                en_cadena.update(ags)

        base_exp = dict(experimento=e.nombre, w=e.w, n_ag=e.n_hechos, max_manos=e.max_manos or "inf",
                        criterio=e.criterio)
        for cons, modo in ((0, "puntual"), (1, "conservador")):
            cand = [(u, c) for cc, u, c in cadenas
                    if cc == cons and u >= umbral_min and len([x for x in c if x != 0]) >= 2]
            fila = dict(base_exp, modo=modo)
            if cand:
                umax = max(u for u, _ in cand)
                c = max((c for u, c in cand if np.isclose(u, umax)), key=len)
                ags = [x for x in c if x != 0]
                top = ags[0]
                fila.update(umbral=umax, largo_agentes=len(ags), cadena=" > ".join(_nombre_nodo(x) for x in c),
                            criterio_mejor="cabeza de cadena", mejor_agente=top,
                            agente_mas_fuerte_liga=mejor_liga, **metricas(top),
                            archivo=f"{e.nombre}/agentes/agente_{top:02d}.npz")
            else:
                fila.update(umbral=np.nan, largo_agentes=0, cadena="(sin cadena de >=2 agentes)",
                            criterio_mejor="mayor fuerza (sin cadena)", mejor_agente=mejor_liga,
                            agente_mas_fuerte_liga=mejor_liga, **metricas(mejor_liga),
                            archivo=f"{e.nombre}/agentes/agente_{mejor_liga:02d}.npz")
            filas.append(fila)

        for pos, r in enumerate(fz.sort_values("fuerza", ascending=False).itertuples(index=False), 1):
            m = int(r.agente)
            ranking.append(dict(base_exp, agente=m, rank_en_liga=pos, en_alguna_cadena=m in en_cadena, **metricas(m)))
    return pd.DataFrame(filas), pd.DataFrame(ranking)


def a_markdown(df: pd.DataFrame) -> str:
    def f(v):
        return f"{v:.3f}" if isinstance(v, float) else str(v)
    cols = list(df.columns)
    lineas = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lineas += ["| " + " | ".join(f(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lineas)


def escribir_tablas(exps: list[Exp], out: Path, umbral_min: float = UMBRAL_MIN_TABLA):
    tc, tr = construir_tablas(exps, umbral_min)
    if tc.empty:
        return tc, tr
    out.mkdir(parents=True, exist_ok=True)
    tc.to_csv(out / "tabla_mejores_agentes.csv", index=False)
    (out / "tabla_mejores_agentes.md").write_text(a_markdown(tc.round(4)), encoding="utf-8")
    tr.to_csv(out / "ranking_agentes.csv", index=False)
    return tc, tr


def imprimir_resumen(tc: pd.DataFrame, umbral_min: float) -> None:
    if tc.empty:
        return
    con = tc[tc.criterio_mejor == "cabeza de cadena"]
    print("\n=== Mejores agentes ===")
    if con.empty:
        print(f"No hay cadenas de >=2 agentes con umbral >= {umbral_min}. Agente más fuerte de cada liga:")
        sub = tc[tc.modo == "puntual"].sort_values("fuerza", ascending=False).head(25)
    else:
        print(f"{len(con)} casos (experimento x modo) con cadena de >=2 agentes; los más exigentes:")
        sub = con.sort_values(["umbral", "largo_agentes", "fuerza"], ascending=False).head(25)
    cols = [c for c in ("experimento", "modo", "umbral", "largo_agentes", "cadena", "mejor_agente", "fuerza",
                        "accion_dominante") if c in sub]
    print(sub[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))


# --------------------------------------------------------------------------
# Estrategia de los agentes que más ganan (estilo "preferencia de acciones")
# --------------------------------------------------------------------------
def _barras_estilo(ax, valores, titulo, xlabel, fmt="{:.0f}"):
    valores = np.asarray(valores, dtype=float)
    orden = np.argsort(-np.nan_to_num(valores))
    colores = plt.cm.Blues(np.linspace(0.85, 0.2, len(orden)))
    ax.barh([ACC_LARGAS[k] for k in orden], [valores[k] for k in orden], color=colores)
    ax.invert_yaxis()
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#d0d0d0")
    for i, k in enumerate(orden):
        ax.text(valores[k], i, " " + fmt.format(valores[k]), va="center", fontsize=9)
    ax.set_xlim(0, np.nanmax(valores) * 1.15 if np.nanmax(valores) > 0 else 1)
    ax.set_title(titulo, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Decisión estratégica (acción óptima)")


def p_estrategia_mejor_agente(e: Exp, out: Path) -> None:
    fz = e.fuerza_agentes()
    if fz.empty or not e.ruta_agente(1).exists():
        return
    m = int(fz.sort_values("fuerza", ascending=False).iloc[0].agente)
    p = Politica(e.ruta_agente(m))
    cuentas = np.bincount(p.a[p.a >= 0], minlength=5)
    fig, axs = plt.subplots(2, 2, figsize=(16, 9.5))
    _barras_estilo(axs[0][0], cuentas, "Estados donde prefiere cada acción", "Cantidad de estados visitados")
    _barras_estilo(axs[0][1], 100 * shares(p, np.ones(N_ESTADOS, bool)),
                   "Lo mismo, ponderado por visitas (lo que más juega)", "% de las decisiones", "{:.1f}%")
    _barras_estilo(axs[1][0], 100 * shares(p, FASE == 0), "Solo PREFLOP (ponderado)", "% de las decisiones", "{:.1f}%")
    _barras_estilo(axs[1][1], 100 * shares(p, FASE > 0), "Solo POSTFLOP (ponderado)", "% de las decisiones", "{:.1f}%")
    fila = fz[fz.agente == m].iloc[0]
    fig.suptitle(f"Estrategia global del agente más fuerte: Agente {m} — {e.etiqueta}\n"
                 f"(prob. media de ganarle a los otros agentes: {fila.fuerza:.3f})", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    guardar(fig, out / "politica", "estrategia_mejor_agente")


def datos_agentes(exps: list[Exp]):
    frames = []
    for e in exps:
        fz = e.fuerza_agentes()
        if e.metricas_agentes is None or fz.empty:
            continue
        m = e.metricas_agentes.merge(fz, on="agente")
        m.insert(0, "exp", e.nombre)
        m["w"], m["n_ag"], m["max_manos"] = e.w, e.n_hechos, e.max_manos
        frames.append(m)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df["pct"] = df.groupby("exp")["fuerza"].rank(pct=True)
    df["fuerza_rel"] = df["fuerza"] - df.groupby("exp")["fuerza"].transform("mean")
    df["grupo"] = np.select([df.pct >= 0.75, df.pct <= 0.25],
                            ["Top 25% de cada liga", "Bottom 25% de cada liga"], "Medio")
    return df


def c_estrategia_ganadores(exps: list[Exp], out: Path) -> None:
    df = datos_agentes(exps)
    if df is None or len(df) < 4:
        return
    colores = {"Top 25% de cada liga": "#08519c", "Medio": "#9ecae1", "Bottom 25% de cada liga": "#cb181d"}

    # 1) reparto de acciones: los que más ganan vs los que menos
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.2), sharey=True)
    alto = 0.27
    for ax, (nom, tit) in zip(axs, (("global", "Global"), ("pre", "Preflop"), ("post", "Postflop"))):
        for i, (g, c) in enumerate(colores.items()):
            sub = df[df.grupo == g]
            ax.barh(np.arange(5) + (i - 1) * alto, [100 * sub[f"{nom}_{k}"].mean() for k in range(5)],
                    height=alto, color=c, label=f"{g} (n={len(sub)})")
        ax.set_title(tit, fontweight="bold")
        ax.set_xlabel("% de decisiones (ponderado por visitas)")
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#d0d0d0")
    axs[0].set_yticks(range(5))
    axs[0].set_yticklabels(ACC_LARGAS)
    axs[0].set_ylim(4.6, -0.6)
    axs[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("¿Qué acciones prefieren los agentes que más ganan? (agentes agrupados por fuerza dentro de su liga)",
                 fontweight="bold", y=1.02)
    guardar(fig, out, "estrategia_ganadores_vs_perdedores")

    # 2) agresividad / retiros vs fuerza
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, col, tit in ((axs[0], "agresion_global", "% de subidas"), (axs[1], "global_0", "% de retiros")):
        sc = ax.scatter(df.fuerza_rel, 100 * df[col], c=df.w, cmap="plasma", vmin=0, vmax=1, edgecolor="k", s=45)
        ok = df[["fuerza_rel", col]].dropna()
        if len(ok) > 2 and ok.fuerza_rel.std() > 0:
            a, b = np.polyfit(ok.fuerza_rel, 100 * ok[col], 1)
            xs = np.linspace(ok.fuerza_rel.min(), ok.fuerza_rel.max(), 20)
            ax.plot(xs, a * xs + b, "k--", lw=1)
            try:
                rho = ok.corr(method="spearman").iloc[0, 1]
                ax.text(0.03, 0.95, f"Spearman = {rho:+.2f}", transform=ax.transAxes, va="top")
            except Exception:  # noqa: BLE001
                pass
        ax.set_xlabel("Fuerza relativa (vs. promedio de su liga)")
        ax.set_ylabel(tit + " (global, ponderado)")
        ax.grid(alpha=0.3)
    plt.colorbar(sc, ax=axs, label="w")
    fig.suptitle("Estilo de juego vs. fuerza (un punto = un agente)", fontweight="bold")
    guardar(fig, out, "agresion_vs_fuerza")

    # 3) el mejor agente de cada experimento
    mejores = df.sort_values("fuerza", ascending=False).groupby("exp").head(1).sort_values("fuerza", ascending=True)
    fig, ax = plt.subplots(figsize=(11, 0.38 * len(mejores) + 2.2))
    izq = np.zeros(len(mejores))
    for k in range(5):
        v = 100 * mejores[f"global_{k}"].fillna(0).values
        ax.barh(range(len(mejores)), v, left=izq, color=COLORES_ACC[k], label=ACCIONES[k])
        izq += v
    ax.set_yticks(range(len(mejores)))
    ax.set_yticklabels([f"{r.exp.replace('grid_', '')} · A{int(r.agente)}" for r in mejores.itertuples()], fontsize=8)
    ax.set_xlabel("% de decisiones (ponderado por visitas)")
    ax.set_title("Reparto de acciones del agente más fuerte de cada experimento (ordenado por fuerza)", fontweight="bold")
    ax.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.07))
    guardar(fig, out, "estrategia_mejor_agente_por_experimento")


def main() -> None:
    global FMT, DPI
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("carpeta", help="carpeta con una liga (manifest.json) o con varias (resultados/)")
    ap.add_argument("--salida", default="graficos")
    ap.add_argument("--formato", default="png", choices=("png", "pdf", "svg"))
    ap.add_argument("--dpi", type=int, default=130)
    ap.add_argument("--agentes", default="1,mejor,ultimo",
                    help="agentes para heatmaps detallados: números, 'mejor' (mayor fuerza media) y/o 'ultimo'")
    ap.add_argument("--sin-politica", action="store_true", help="no cargar los .npz")
    ap.add_argument("--sin-cadena-fina", action="store_true", help="no recalcular cadenas para umbrales finos")
    ap.add_argument("--sin-comparativa", action="store_true")
    ap.add_argument("--umbral-min-tabla", type=float, default=UMBRAL_MIN_TABLA,
                    help="umbral mínimo para considerar que una cadena de agentes es relevante (tabla de mejores)")
    ap.add_argument("--comparar-todo", action="store_true",
                    help="incluir en la comparativa carpetas que no empiezan con grid_ (liga, prueba, ...)")
    ap.add_argument("--solo-comparativa", action="store_true", help="omitir los gráficos por experimento")
    args = ap.parse_args()
    FMT, DPI = args.formato, args.dpi

    raiz = Path(args.carpeta)
    if (raiz / "manifest.json").exists():
        rutas = [raiz]
    else:
        rutas = sorted(p for p in raiz.iterdir() if p.is_dir() and (p / "manifest.json").exists())
    if not rutas:
        sys.exit(f"No encontré ligas (manifest.json) en {raiz}")
    salida = Path(args.salida)
    print(f"[*] {len(rutas)} experimento(s) → {salida}/")

    exps = []
    for i, r in enumerate(rutas, 1):
        print(f"[{i}/{len(rutas)}] {r.name}")
        e = Exp(r)
        out = salida / e.nombre
        if not args.sin_cadena_fina:
            e.fina = seguro_val(calcular_curva_fina, e)
        if not args.solo_comparativa:
            for fn in (p_cadena_vs_iteracion, p_cadena_vs_umbral_fino, p_transitividad, p_entrenamiento,
                       p_winrates, p_bb, p_curvas_entrenamiento, p_matrices, p_fuerza_global):
                if fn is p_cadena_vs_umbral_fino and args.sin_cadena_fina:
                    continue
                seguro(fn, e, out)
        politica_ok = not args.sin_politica and e.ruta_agente(1).exists()
        if politica_ok:
            try:
                Politica(e.ruta_agente(1))
            except ValueError as err:
                print(f"   [aviso] se omiten los gráficos de política: {err}")
                politica_ok = False
        if politica_ok:
            seguro(p_evolucion_politica, e, out)
            if not args.solo_comparativa:
                for m in elegir_agentes(e, args.agentes):
                    graficos_agente(e, m, out)
                seguro(p_estrategia_mejor_agente, e, out)
        if not args.solo_comparativa:
            escribir_tablas([e], out, args.umbral_min_tabla)
        exps.append(e)

    # tablas de mejores agentes (todas las ligas) en la raíz de la salida
    tc, _ = escribir_tablas(exps, salida, args.umbral_min_tabla)
    imprimir_resumen(tc, args.umbral_min_tabla)

    # la comparativa solo mezcla experimentos de la grilla (grid_*), salvo --comparar-todo
    comp = exps if args.comparar_todo else [e for e in exps if e.nombre.startswith("grid_")]
    if len(comp) > 1 and not args.sin_comparativa:
        print(f"[*] Comparativa entre {len(comp)} experimentos")
        comparativa(comp, salida / "_comparativa")
        seguro(c_estrategia_ganadores, comp, salida / "_comparativa")
    print("[✓] Listo.")


def seguro_val(fn, *args):
    try:
        return fn(*args)
    except Exception as e:  # noqa: BLE001
        print(f"   [aviso] {fn.__name__} falló: {e}")
        return None


if __name__ == "__main__":
    main()