#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figuras para la sección de Resultados (complementa a graficos_liga.py).

Debe quedar junto a graficos_liga.py. Uso (desde la raíz del proyecto):

    # un experimento representativo (el de cadena más larga a umbral 0.6, o el que indiques con --exp)
    python graficos_resultados.py resultados
    python graficos_resultados.py resultados --exp grid_w_0_25_nag_10_manos_m3 --umbrales 0.6,0.7,0.8,0.9

    # TODOS los experimentos (w, n_agentes y max_manos variables) + comparaciones entre ellos
    python graficos_resultados.py resultados --todos --formato pdf --dpi 200

Salida (carpeta --salida, por defecto graficos/resultados/). Cada gráfico es una figura independiente:

  por experimento (en la raíz de --salida, o en <salida>/<experimento>/ con --todos):
    1a_winrate_por_mano, 1b_bb_por_mano        aprendizaje y mejora generacional (vs Uniforme y vs anterior)
    2a_matriz_enfrentamiento                   P(fila le gana a columna)
    2b_grafo_dominancia_uXX                    grafo de dominancia y cadena más larga, un archivo por umbral
    5a_estrategia_mejor_agente                 reparto global / preflop / postflop de acciones del agente más fuerte
    5b_acciones_por_fuerza_preflop, 5c_..._postflop   acción preferida según fuerza de la mano
    5d1..5d3_preflop_*                         matriz 13x13 preflop del agente más fuerte (3 contextos)

  comparaciones entre experimentos (raíz de --salida):
    3a_largo_cadena_u60, 3b_largo_cadena_u90, 3c_winrate_vs_anterior, 3d_estilo_ultimo_agente   (efecto de w)
    4_comparacion_grafos_uXX_manos_M           (solo con --todos) mismos grafos en una grilla w x n_agentes,
                                               una figura por umbral y por valor de max_manos
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, Patch

import graficos_liga as gl
from graficos_liga import Exp, heat, guardar, seguro, _cargar_matriz

UMBRALES_DEFECTO = (0.6, 0.7, 0.8, 0.9)


def wilson(k: float, n: float, z: float = 1.96):
    """Proporción e intervalo de Wilson al 95%."""
    if n <= 0:
        return np.nan, np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, c - h, c + h


# --------------------------------------------------------------------------
# Figura 1: ¿aprenden? ¿mejoran generacionalmente?
# --------------------------------------------------------------------------
def f_validacion(e: Exp, out: Path) -> None:
    enf = e.enf
    if enf.empty:
        return
    series = (("vs Uniforme (aleatorio)", lambda m: 0, "#2980b9", -0.1),
              ("vs agente anterior", lambda m: m - 1, "#c0392b", 0.1))
    datos = []
    for lab, rival, color, desp in series:
        xs, p, err_lo, err_hi, bb, bb_e = [], [], [], [], [], []
        for m in range(1, e.n_hechos + 1):
            if lab.startswith("vs agente") and m == 1:
                continue
            r = enf[(enf.a == m) & (enf.b == rival(m))]
            if r.empty:
                continue
            r = r.iloc[0]
            k = float(r.ganadas_a)
            n = float(r.ganadas_a + r.ganadas_b)  # manos con ganador (se excluyen empates)
            pp, lo, hi = wilson(k, n)
            xs.append(m + desp)
            p.append(pp)
            err_lo.append(pp - lo)
            err_hi.append(hi - pp)
            bb.append(float(r.bb_por_mano_a))
            bb_e.append(float(r.bb_ic95))
        if xs:
            datos.append((lab, color, xs, p, [err_lo, err_hi], bb, bb_e))
    if not datos:
        return

    for nombre, idx, yerr_idx, ref, ylab, tit in (
            ("1a_winrate_por_mano", 3, 4, 0.5, "P(ganar la mano) entre manos con ganador\n(IC de Wilson 95%)",
             "Winrate por mano"),
            ("1b_bb_por_mano", 5, 6, 0.0, "Ganancia media (bb por mano) ± IC95", "Ganancia en ciegas grandes")):
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        for d in datos:
            ax.errorbar(d[2], d[idx], yerr=d[yerr_idx], fmt="o-", ms=4, lw=1, capsize=2, color=d[1], label=d[0])
        ax.axhline(ref, ls=":", color="k", alpha=0.6)
        ax.set_ylabel(ylab)
        ax.set_xlabel("Agente m (orden de entrenamiento)")
        ax.set_xticks(range(1, e.n_hechos + 1))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
        ax.set_title(f"{tit} — {e.etiqueta}", fontsize=10)
        guardar(fig, out, nombre)


# --------------------------------------------------------------------------
# Figura 2: matriz de enfrentamiento + grafo de dominancia
# --------------------------------------------------------------------------
def _datos_dominancia(e: Exp, u: float):
    """(matriz, etiquetas columnas, etiquetas filas, cadena estricta más larga a umbral u)."""
    nombre = {"racha": "matriz_racha", "mano": "matriz_mano", "partida": "matriz_partida"}.get(
        e.criterio, "matriz_racha")
    r = _cargar_matriz(e, nombre)
    if r is None:
        return None
    M, xl, yl = r
    cadena: list[int] = []
    if gl.HAY_CADENAS and not e.enf.empty:
        G, Gp = e.matrices()
        base, crit = (Gp, "mano") if e.criterio == "partida" else (G, "racha" if e.criterio == "racha" else "mano")
        P = gl.matriz_probabilidad(base, crit, e.k, False)
        cadena = [int(x) for x in gl.cadena_estricta_mas_larga(gl.adyacencia(P, u))]
    return M, xl, yl, cadena


def _dibujar_grafo(ax, M: np.ndarray, cadena: list[int], u: float, grande: bool = True) -> None:
    n = len(M)
    ang = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, n, endpoint=False)
    pos = np.c_[np.cos(ang), np.sin(ang)]
    aristas_cadena = set(zip(cadena[:-1], cadena[1:]))
    ms_c, ms_n, shr = (14, 8, 13) if grande else (8, 5, 7)
    for i in range(n):
        for j in range(n):
            if i == j or not np.isfinite(M[i, j]) or M[i, j] < u:
                continue
            en_cadena = (i, j) in aristas_cadena
            ax.add_patch(FancyArrowPatch(pos[i], pos[j], arrowstyle="-|>", mutation_scale=ms_c if en_cadena else ms_n,
                                         shrinkA=shr, shrinkB=shr, connectionstyle="arc3,rad=0.12",
                                         color="#c0392b" if en_cadena else "#aab7b8",
                                         lw=(2.4 if grande else 1.6) if en_cadena else (0.7 if grande else 0.5),
                                         alpha=1.0 if en_cadena else 0.7, zorder=3 if en_cadena else 1))
    en = set(cadena)
    for i in range(n):
        ax.scatter(*pos[i], s=520 if grande else 150, color="#fadbd8" if i in en else "#eaf2f8", edgecolor="k",
                   zorder=4)
        ax.text(*pos[i], "U" if i == 0 else str(i), ha="center", va="center", fontsize=10 if grande else 6,
                zorder=5, fontweight="bold")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")


def _txt_cadena(cadena: list[int]) -> str:
    return " > ".join("U" if x == 0 else str(x) for x in cadena) if cadena else "(no disponible)"


def f_dominancia(e: Exp, out: Path, umbral: float | None = None, matriz: bool = True) -> None:
    u = float(umbral if umbral is not None else e.objetivo)
    d = _datos_dominancia(e, u)
    if d is None:
        return
    M, xl, yl, cadena = d
    if matriz:
        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        heat(ax, M, xl, yl, f"P(fila le gana a columna) — criterio {e.criterio}\n{e.etiqueta}", cmap="RdBu",
             vmin=0.25, vmax=0.75, fmt="{:.2f}", fs=8)
        ax.set_xlabel("Rival")
        ax.set_ylabel("Agente")
        guardar(fig, out, "2a_matriz_enfrentamiento")

    fig, ax = plt.subplots(figsize=(7, 7))
    _dibujar_grafo(ax, M, cadena, u, grande=True)
    ax.set_title(f"Grafo de dominancia — {e.etiqueta}\nArista i→j: i le gana a j con P ≥ {u:.2f}\n"
                 f"Cadena estricta más larga (rojo): {_txt_cadena(cadena)}", fontsize=10)
    guardar(fig, out, f"2b_grafo_dominancia_u{int(round(u * 100))}")


def f_comparacion_grafos(exps: list[Exp], out: Path, umbrales: list[float]) -> None:
    """Grilla w (filas) x n_agentes (columnas) de grafos; una figura por umbral y por valor de max_manos."""
    mms = sorted({e.max_manos for e in exps})
    ws = sorted({e.w for e in exps})
    ns = sorted({e.n_hechos for e in exps})
    for u in umbrales:
        for mm in mms:
            fig, axs = plt.subplots(len(ws), len(ns), figsize=(3.6 * len(ns) + 0.6, 3.6 * len(ws) + 1.0),
                                    squeeze=False)
            hay = False
            for r_, w in enumerate(ws):
                for c_, n in enumerate(ns):
                    ax = axs[r_][c_]
                    ax.axis("off")
                    e = next((x for x in exps if x.w == w and x.n_hechos == n and x.max_manos == mm), None)
                    d = _datos_dominancia(e, u) if e is not None else None
                    if d is None:
                        continue
                    _dibujar_grafo(ax, d[0], d[3], u, grande=False)
                    ax.set_title(f"w={w:g} · n={n}\nlargo cadena: {len(d[3])}", fontsize=9)
                    hay = True
            if not hay:
                plt.close(fig)
                continue
            fig.suptitle(f"Grafos de dominancia (P ≥ {u:.2f}) — max_manos={'∞' if mm == 0 else mm} · "
                         "rojo = cadena estricta más larga", y=1.0)
            fig.tight_layout()
            guardar(fig, out, f"4_comparacion_grafos_u{int(round(u * 100))}_manos_{'inf' if mm == 0 else mm}")


# --------------------------------------------------------------------------
# Figura 3: efecto de w (y de la definición de victoria) sobre cadenas y estilo
# --------------------------------------------------------------------------
def f_efecto_w(exps: list[Exp], out: Path) -> None:
    df = gl.construir_resumen(exps)
    if df.empty or df.w.nunique() < 2:
        return
    paneles = (
        ("3a_largo_cadena_u60", "Largo de cadena, winrate ≥ 0.6", "nodos (incl. Uniforme)",
         (("largo_p_60", "puntual", "#2980b9", "-"), ("largo_c_60", "conservador", "#2980b9", "--")), None, 1),
        ("3b_largo_cadena_u90", "Largo de cadena, winrate ≥ 0.9", "nodos (incl. Uniforme)",
         (("largo_p_90", "puntual", "#c0392b", "-"), ("largo_c_90", "conservador", "#c0392b", "--")), None, 1),
        ("3c_winrate_vs_anterior", "Winrate del agente m vs m−1", "criterio de la liga",
         (("wr_vs_anterior", "promedio de la liga", "#27ae60", "-"),), None, None),
        ("3d_estilo_ultimo_agente", "Estilo del último agente", "fracción de decisiones",
         (("ult_agresion_post", "% subidas postflop", "#8e44ad", "-"),
          ("ult_agresion_pre", "% subidas preflop", "#f39c12", "-"),
          ("ult_fold_post_débil", "% retiros con mano débil", "#7f8c8d", "--")), (-0.02, 1.02), None),
    )
    for nombre, tit, yl, cols, ylim, ymin in paneles:
        fig, ax = plt.subplots(figsize=(7, 4.6))
        for col, lab, color, ls in cols:
            if col not in df or df[col].notna().sum() == 0:
                continue
            g = df.groupby("w")[col].agg(["mean", "min", "max"]).dropna()
            ax.plot(g.index, g["mean"], ls, marker="o", ms=5, color=color, label=lab)
            ax.fill_between(g.index, g["min"], g["max"], color=color, alpha=0.12)
        ax.set_title(f"{tit}\n(línea = promedio de la grilla, banda = mín–máx)", fontsize=10)
        ax.set_xlabel("w  (0 = solo victoria, 1 = solo fichas)")
        ax.set_ylabel(yl)
        ax.grid(alpha=0.3)
        if ylim:
            ax.set_ylim(*ylim)
        if ymin is not None:
            ax.set_ylim(bottom=ymin)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
        guardar(fig, out, nombre)


# --------------------------------------------------------------------------
# Figura 5: estrategia del agente más fuerte
# --------------------------------------------------------------------------
def f_estrategia(e: Exp, out: Path) -> None:
    fz = e.fuerza_agentes()
    if fz.empty or not e.ruta_agente(1).exists():
        return
    fila = fz.sort_values("fuerza", ascending=False).iloc[0]
    m = int(fila.agente)
    if not e.ruta_agente(m).exists():
        return
    p = gl.Politica(e.ruta_agente(m))
    sub = f"Agente {m} (fuerza {fila.fuerza:.3f}) — {e.etiqueta}"

    # 5a) reparto de acciones: global / preflop / postflop
    mascaras = (("Global", np.ones(gl.N_ESTADOS, bool), "#34495e"), ("Preflop", gl.FASE == 0, "#2980b9"),
                ("Postflop", gl.FASE > 0, "#e67e22"))
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ancho = 0.26
    for i, (nom, mask, c) in enumerate(mascaras):
        ax.bar(np.arange(5) + (i - 1) * ancho, 100 * gl.shares(p, mask), width=ancho, color=c, label=nom)
    ax.set_xticks(range(5))
    ax.set_xticklabels(gl.ACCIONES)
    ax.set_ylabel("% de decisiones (ponderado por visitas)")
    ax.set_title(f"Estrategia del agente más fuerte de la liga\n{sub}", fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    guardar(fig, out, "5a_estrategia_mejor_agente")

    # 5b / 5c) acción preferida según fuerza de la mano
    for nombre, mask, n_f, etq, tit in (
            ("5b_acciones_por_fuerza_preflop", gl.FASE == 0, 12, gl.ETQ_PRE, "Preflop"),
            ("5c_acciones_por_fuerza_postflop", gl.FASE > 0, gl.N_FUERZA, gl.ETQ_POST, "Postflop")):
        M = np.array([gl.shares(p, mask & (gl.FUERZA == f)) for f in range(n_f)])
        fig, ax = plt.subplots(figsize=(7.2, 0.45 * n_f + 2.4))
        heat(ax, M, gl.ACCIONES, etq, f"{tit}: acción preferida según la fuerza de la mano\n{sub}",
             cmap="viridis", vmin=0, vmax=1)
        guardar(fig, out, nombre)

    # 5d) matriz 13x13 preflop (requiere Clases.abstraccion)
    if gl.HAY_ABSTRACCION:
        cmap = ListedColormap(gl.COLORES_ACC).copy()
        cmap.set_bad("#dddddd")
        etq = [gl.GRID_TXT.get(r, str(r)) for r in gl.GRID_RANGOS]
        sufijos = ("sb", "bb_opcion", "bb_vs_subida")
        for k, (tit, ctx) in enumerate(gl.CONTEXTOS_PREFLOP, 1):
            M = gl.grilla_preflop(p, ctx)
            fig, ax = plt.subplots(figsize=(7.5, 7.8))
            ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=-0.5, vmax=4.5)
            for i, j in np.ndindex(M.shape):
                if np.isfinite(M[i, j]):
                    suf = "" if i == j else ("s" if i < j else "o")
                    ax.text(j, i, f"{etq[min(i, j)]}{etq[max(i, j)]}{suf}\n{gl.ACC_CORTO[int(M[i, j])]}",
                            ha="center", va="center", fontsize=6.5)
            ax.set_xticks(range(13))
            ax.set_xticklabels(etq)
            ax.set_yticks(range(13))
            ax.set_yticklabels(etq)
            ax.set_title(f"Preflop — {tit}\n{sub}\n(sobre la diagonal = suited 's', bajo = offsuit 'o')", fontsize=9)
            ax.legend(handles=[Patch(color=c, label=a) for c, a in zip(gl.COLORES_ACC, gl.ACCIONES)],
                      loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=3, fontsize=8)
            guardar(fig, out, f"5d{k}_preflop_{sufijos[k - 1]}")


# --------------------------------------------------------------------------
def correr_experimento(e: Exp, out: Path, umbrales: list, estrategia: bool = True) -> None:
    seguro(f_validacion, e, out)
    for k, u in enumerate(umbrales):
        seguro(f_dominancia, e, out, u, k == 0)  # la matriz solo se dibuja una vez
    if estrategia:
        seguro(f_estrategia, e, out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("carpeta", help="carpeta con varias ligas (resultados/) o con una sola (manifest.json)")
    ap.add_argument("--exp", default=None, help="experimento representativo (por defecto: el de cadena más larga "
                                                "a umbral 0.6)")
    ap.add_argument("--todos", action="store_true",
                    help="generar las figuras de TODOS los experimentos (en <salida>/<experimento>/) y las "
                         "comparaciones entre ellos")
    ap.add_argument("--umbral", type=float, default=None, help="un umbral para el grafo de dominancia")
    ap.add_argument("--umbrales", default="", help="varios umbrales separados por coma (ej. 0.6,0.7,0.8,0.9); "
                                                   "con --todos el defecto es 0.6,0.7,0.8,0.9")
    ap.add_argument("--sin-estrategia", action="store_true", help="omitir los gráficos de estrategia (no carga .npz)")
    ap.add_argument("--salida", default="graficos/resultados")
    ap.add_argument("--formato", default="png", choices=("png", "pdf", "svg"))
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--comparar-todo", action="store_true", help="incluir carpetas que no empiezan con grid_")
    a = ap.parse_args()
    gl.FMT, gl.DPI = a.formato, a.dpi

    raiz = Path(a.carpeta)
    rutas = [raiz] if (raiz / "manifest.json").exists() else sorted(
        p for p in raiz.iterdir() if p.is_dir() and (p / "manifest.json").exists())
    if not rutas:
        sys.exit(f"No encontré ligas (manifest.json) en {raiz}")
    exps = []
    for r in rutas:
        e = Exp(r)
        if e.ruta_agente(e.n_hechos).exists():  # estilo del último agente (para la figura 3d)
            try:
                e.metricas_ultimo = gl.metricas_politica(gl.Politica(e.ruta_agente(e.n_hechos)))
            except Exception as err:  # noqa: BLE001
                print(f"   [aviso] {e.nombre}: sin métricas de política ({err})")
        exps.append(e)

    grid = exps if a.comparar_todo else [e for e in exps if e.nombre.startswith("grid_")] or exps
    out = Path(a.salida)

    umbrales = [float(x) for x in a.umbrales.split(",") if x.strip()]
    if not umbrales:
        umbrales = [a.umbral] if a.umbral is not None else (list(UMBRALES_DEFECTO) if a.todos else [None])
    estrategia = not a.sin_estrategia

    if a.todos:
        print(f"[*] {len(grid)} experimentos → {out}/<experimento>/")
        for i, e in enumerate(grid, 1):
            print(f"[{i}/{len(grid)}] {e.nombre}")
            correr_experimento(e, out / e.nombre, umbrales, estrategia)
        if len(grid) > 1:
            seguro(f_comparacion_grafos, grid, out, [u for u in umbrales if u is not None])
    else:
        if a.exp:
            rep = next((e for e in exps if e.nombre == a.exp), None)
            if rep is None:
                sys.exit(f"No existe el experimento {a.exp}. Disponibles: {', '.join(e.nombre for e in exps)}")
        else:
            res = gl.construir_resumen(grid)
            nom = (res.sort_values(["largo_p_60", "wr_vs_anterior"], ascending=False).exp.iloc[0]
                   if "largo_p_60" in res else res.exp.iloc[0])
            rep = next(e for e in grid if e.nombre == nom)
        print(f"[*] Experimento representativo: {rep.nombre}")
        correr_experimento(rep, out, umbrales, estrategia)

    if len(grid) > 1:
        seguro(f_efecto_w, grid, out)
    print(f"[✓] Figuras en {out}/")


if __name__ == "__main__":
    main()