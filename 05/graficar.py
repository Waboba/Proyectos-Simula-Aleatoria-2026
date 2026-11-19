"""
Gráficos a partir de los resultados de la liga (requiere matplotlib).

    python graficar.py resultados/liga/entrenamiento/agente_01                     # evolución de un agente
    python graficar.py resultados/liga/entrenamiento/agente_01 resultados/liga/entrenamiento/agente_02
    python graficar.py --iteraciones resultados/liga/iteraciones.csv              # resumen global de la liga
    python graficar.py --matriz resultados/liga/matriz_racha.csv                  # mapa de calor de la liga
"""
import argparse
import csv
import os

import matplotlib
import numpy as np


def leer_csv(ruta: str) -> dict[str, np.ndarray]:
    with open(ruta, newline="", encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    if not filas:
        raise ValueError(f"El archivo {ruta} está vacío.")
    resultado = {}
    for k in filas[0]:
        try:
            resultado[k] = np.array(
                [float(fila[k]) if fila[k] != "" else np.nan for fila in filas],
                dtype=np.float64,
            )
        except ValueError:
            pass  # ignora columnas de texto como 'hash_q'
    return resultado


def _nombre_nodo(j: int) -> str:
    return "Uniforme" if j == 0 else f"Agente {j}"


def graficar_corridas(carpetas: list[str], rival: int | None, salida: str, mostrar: bool, objetivo: float) -> None:
    """Grafica metricas.csv generado por liga.py dentro de entrenamiento/agente_MM/."""
    import matplotlib.pyplot as plt

    fig, ejes = plt.subplots(2, 2, figsize=(13, 8))
    varios = len(carpetas) > 1

    for carpeta in carpetas:
        datos = leer_csv(os.path.join(carpeta, "metricas.csv"))
        nombre = os.path.basename(os.path.normpath(carpeta))
        x = datos["episodio"]

        # 1) Valor del criterio contra rivales individuales (valor_vs_j + ic95_vs_j)
        indices_rivales = sorted(
            int(k[len("valor_vs_"):]) for k in datos if k.startswith("valor_vs_")
        )
        if rival is not None:
            indices_rivales = [j for j in indices_rivales if j == rival]

        for j in indices_rivales:
            y = datos[f"valor_vs_{j}"]
            ic = datos[f"ic95_vs_{j}"]
            etiqueta = f"{nombre} vs {_nombre_nodo(j)}" if varios else f"vs {_nombre_nodo(j)}"
            linea, = ejes[0, 0].plot(x, y, marker="o", ms=3, label=etiqueta)
            ejes[0, 0].fill_between(x, y - ic, y + ic, alpha=0.15, color=linea.get_color())

        # 2) Valor mezcla (promedio / global contra todo el pool)
        ejes[0, 1].plot(x, datos["valor_mezcla"], marker="o", ms=3, label=nombre)

        # 3) Recompensa media durante el entrenamiento
        ejes[1, 0].plot(x, datos["recompensa_entreno_media"], marker="o", ms=3, label=nombre)

        # 4) Epsilon y estados visitados
        l_eps, = ejes[1, 1].plot(x, datos["epsilon"], label=f"{nombre}: epsilon")
        ax2 = ejes[1, 1].twinx()
        ax2.plot(x, datos["estados_visitados"], linestyle="--", color=l_eps.get_color(), alpha=0.7)
        ax2.set_ylabel("estados visitados (línea discontinua)")

    ejes[0, 0].axhline(objetivo, color="red", linestyle=":", label=f"objetivo ({objetivo:.0%})")
    ejes[0, 0].axhline(0.5, color="gray", linestyle=":")
    ejes[0, 0].set_title("Rendimiento evaluación vs cada rival (criterio de liga)")

    ejes[0, 1].axhline(objetivo, color="red", linestyle=":", label=f"objetivo ({objetivo:.0%})")
    ejes[0, 1].axhline(0.5, color="gray", linestyle=":")
    ejes[0, 1].set_title("Rendimiento mezcla contra el pool")

    ejes[1, 0].axhline(0.0, color="gray", linestyle=":")
    ejes[1, 0].set_title("Recompensa media en entrenamiento (por bloque)")

    ejes[1, 1].set_title("Exploración (epsilon) y cobertura de tabla Q")
    ejes[1, 1].set_ylabel("epsilon")

    for ax in (ejes[0, 0], ejes[0, 1], ejes[1, 0]):
        ax.legend()
    for ax in ejes.flat:
        ax.set_xlabel("episodios (manos de entrenamiento)")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(salida)), exist_ok=True)
    fig.savefig(salida, dpi=150)
    print(f"Figura guardada en {salida}")
    if mostrar:
        plt.show()


def graficar_iteraciones(ruta_csv: str, salida: str, mostrar: bool, objetivo: float) -> None:
    """Grafica el resumen por agente de iteraciones.csv generado por liga.py."""
    import matplotlib.pyplot as plt

    datos = leer_csv(ruta_csv)
    x = datos["iteracion"]

    fig, ejes = plt.subplots(2, 2, figsize=(13, 8))

    # 1) Win rates al congelar
    ejes[0, 0].plot(x, datos["winrate_vs_uniforme"], marker="o", label="vs Uniforme")
    ejes[0, 0].plot(x, datos["winrate_vs_anterior"], marker="s", label="vs Agente anterior")
    ejes[0, 0].plot(x, datos["winrate_min_parada"], marker="^", linestyle="--", label="mínimo en parada")
    ejes[0, 0].axhline(objetivo, color="red", linestyle=":", label=f"objetivo ({objetivo:.0%})")
    ejes[0, 0].set_title("Rendimiento al congelar (criterio de victoria)")
    ejes[0, 0].set_ylabel("score / win rate")
    ejes[0, 0].legend()

    # 2) Largo de cadena estricta
    ejes[0, 1].plot(x, datos["largo_cadena"], marker="o", label="puntual")
    ejes[0, 1].plot(x, datos["largo_cadena_conservador"], marker="s", label="conservador (IC95)")
    ejes[0, 1].plot(x, x, linestyle=":", color="gray", label="máximo teórico (n agentes)")
    ejes[0, 1].set_title("Largo de cadena estricta más larga")
    ejes[0, 1].set_ylabel("cantidad de agentes")
    ejes[0, 1].legend()

    # 3) Episodios necesarios para converger
    alcanzo = datos["alcanzo_objetivo"] == 1
    colores = ["#2ca02c" if ok else "#d62728" for ok in alcanzo]
    ejes[1, 0].bar(x, datos["episodios"], color=colores)
    ejes[1, 0].set_title("Manos de entrenamiento por agente (verde = alcanzó objetivo)")
    ejes[1, 0].set_ylabel("episodios")

    # 4) Transitividad y bb/mano vs anterior
    l1, = ejes[1, 1].plot(x, datos["frac_transitiva"], marker="o", color="tab:purple", label="fracción transitiva")
    ejes[1, 1].set_ylim(-0.05, 1.05)
    ejes[1, 1].set_ylabel("fracción transitiva", color="tab:purple")
    ax2 = ejes[1, 1].twinx()
    l2, = ax2.plot(x, datos["bb_vs_anterior"], marker="d", color="tab:orange", linestyle="--", label="bb/mano vs anterior")
    ax2.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax2.set_ylabel("bb/mano vs anterior", color="tab:orange")
    ejes[1, 1].set_title("Transitividad de la liga y ventaja en bb/mano")
    ejes[1, 1].legend([l1, l2], [l1.get_label(), l2.get_label()], loc="lower right")

    for ax in ejes.flat:
        ax.set_xlabel("agente (iteración)")
        ax.set_xticks(x)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(salida)), exist_ok=True)
    fig.savefig(salida, dpi=150)
    print(f"Figura guardada en {salida}")
    if mostrar:
        plt.show()


def graficar_matriz(ruta_csv: str, salida: str, mostrar: bool) -> None:
    import matplotlib.pyplot as plt

    with open(ruta_csv, newline="", encoding="utf-8") as f:
        filas = list(csv.reader(f))
    etiquetas = filas[0][1:]
    W = np.array([[float(v) for v in fila[1:]] for fila in filas[1:]])
    fig, ax = plt.subplots(figsize=(1.2 * len(etiquetas) + 3, 1.0 * len(etiquetas) + 2))
    im = ax.imshow(W, vmin=0, vmax=1, cmap="RdYlGn")
    ax.set_xticks(range(len(etiquetas)), etiquetas, rotation=45, ha="right")
    ax.set_yticks(range(len(etiquetas)), etiquetas)
    for i in range(len(etiquetas)):
        for j in range(len(etiquetas)):
            ax.text(j, i, f"{W[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_xlabel("rival (columna)")
    ax.set_ylabel("agente (fila): score contra la columna")
    fig.colorbar(im)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(salida)), exist_ok=True)
    fig.savefig(salida, dpi=150)
    print(f"Figura guardada en {salida}")
    if mostrar:
        plt.show()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("carpetas", nargs="*", help="carpetas entrenamiento/agente_MM con metricas.csv")
    ap.add_argument("--rival", type=int, default=None, help="índice de rival específico a graficar (0=Uniforme, 1=Agente 1, ...)")
    ap.add_argument("--iteraciones", default=None, help="ruta a iteraciones.csv de la liga")
    ap.add_argument("--matriz", default=None, help="ruta a matriz_mano.csv / matriz_racha.csv / matriz_partida.csv")
    ap.add_argument("--salida", default=None)
    ap.add_argument("--objetivo", type=float, default=0.60, help="línea de referencia del objetivo")
    ap.add_argument("--mostrar", action="store_true", help="abrir la ventana del gráfico")
    args = ap.parse_args()

    if not args.mostrar:
        matplotlib.use("Agg")

    if args.matriz:
        salida = args.salida or os.path.splitext(args.matriz)[0] + ".png"
        graficar_matriz(args.matriz, salida, args.mostrar)
    elif args.iteraciones:
        salida = args.salida or os.path.splitext(args.iteraciones)[0] + ".png"
        graficar_iteraciones(args.iteraciones, salida, args.mostrar, args.objetivo)
    elif args.carpetas:
        salida = args.salida or os.path.join(args.carpetas[0], "curvas.png")
        graficar_corridas(args.carpetas, args.rival, salida, args.mostrar, args.objetivo)
    else:
        ap.error("indica al menos una carpeta de entrenamiento, --iteraciones o --matriz")


if __name__ == "__main__":
    main()