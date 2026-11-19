"""
Corre (o reanuda) una liga de agentes con congelamiento sucesivo.

    python entrenar_liga.py resultados/liga                                  # 10 agentes, criterio "racha" (3 seguidas)
    python entrenar_liga.py resultados/liga --n-agentes 20 --semilla 1
    python entrenar_liga.py resultados/liga --criterio partida --objetivo 0.6
    python entrenar_liga.py resultados/liga --n-agentes 30 --reanudar        # agrega agentes sin tocar los anteriores
    python entrenar_liga.py resultados/prueba --rapida                       # prueba de humo (minutos)

Después de entrenar, corrobora los resultados con:  python verificar_liga.py <carpeta>
"""
import argparse
import csv
import os

from Clases.liga import ConfigLiga, correr_liga


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("carpeta")
    ap.add_argument("--n-agentes", type=int, default=None)
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--criterio", choices=("racha", "mano", "partida"), default="racha",
                    help="definición de 'ganar' (parada y cadenas)")
    ap.add_argument("--k", type=int, default=3, help="largo de la racha (criterio racha)")
    ap.add_argument("--objetivo", type=float, default=0.60)
    ap.add_argument("--max-manos-partida", type=int, default=None, 
                help="manos máximas por partida; si nadie quiebra, gana quien tenga más fichas (criterio partida)")
    ap.add_argument("--umbral", type=float, default=0.90, help="umbral de la cadena estricta principal")
    ap.add_argument("--parada", choices=("cada_rival", "mezcla"), default="cada_rival")
    ap.add_argument("--max-episodios", type=int, default=None)
    ap.add_argument("--manos-por-par", type=int, default=None, help="manos fijas por par (en vez de repartir el total)")
    ap.add_argument("--partidas-por-par", type=int, default=None)
    ap.add_argument("--rapida", action="store_true", help="configuración reducida de prueba")
    ap.add_argument("--reanudar", action="store_true")
    ap.add_argument("--detener-si-no-alcanza", action="store_true")
    ap.add_argument("--recompensa", choices=("auto", "fichas", "victoria", "mixta"), default="auto",
                help="tipo de función de recompensa")
    ap.add_argument("--peso-fichas", type=float, default=0.5, 
                help="peso w para la recompensa mixta (0 = solo fichas, 1 = solo victoria)")
    args = ap.parse_args()

    kw = dict(semilla=args.semilla, criterio_victoria=args.criterio, k_racha=args.k, objetivo=args.objetivo,
              umbral_cadena=args.umbral, criterio_parada=args.parada, manos_por_par=args.manos_por_par,
              partidas_por_par=args.partidas_por_par, detener_si_no_alcanza=args.detener_si_no_alcanza, recompensa=args.recompensa,peso_fichas=args.peso_fichas)
    if args.n_agentes is not None:
        kw["n_agentes"] = args.n_agentes
    if args.max_episodios is not None:
        kw["max_episodios"] = args.max_episodios
    if args.max_manos_partida is not None:
        kw["max_manos_partida"] = args.max_manos_partida
    cfg = ConfigLiga.prueba_rapida(**kw) if args.rapida else ConfigLiga(**kw)

    manifest = correr_liga(cfg, args.carpeta, reanudar=args.reanudar)

    # resumen: cadena más larga de la última iteración, por umbral y modo
    ultima = max(int(k) for k in manifest["agentes"])
    print(f"\n=== Resumen (iteración {ultima}) ===")
    with open(os.path.join(args.carpeta, "cadenas.csv"), newline="", encoding="utf-8") as f:
        for fila in csv.DictReader(f):
            if int(fila["iteracion"]) == ultima:
                modo = "conservador" if fila["conservador"] == "1" else "puntual    "
                print(f"  umbral {float(fila['umbral']):.1f} | {modo} | {fila['largo_nodos']:>2} nodos | "
                      f"transitiva {float(fila['frac_transitiva']):.2f} | {fila['cadena']}")
    print(f"\nArchivos en {args.carpeta}. Siguiente paso: python verificar_liga.py {args.carpeta}")


if __name__ == "__main__":
    main()
