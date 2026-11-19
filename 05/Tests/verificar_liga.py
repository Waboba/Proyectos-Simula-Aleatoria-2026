"""
Corroboración independiente de una liga ya entrenada.

    python verificar_liga.py resultados/liga                         # integridad + re-medición con cartas nuevas
    python verificar_liga.py resultados/liga --partidas 4000 --manos 40000
    python verificar_liga.py resultados/liga --rachas 3000            # simula directamente las carreras a k manos seguidas
    python verificar_liga.py resultados/liga --reentrenar 2           # reentrena el Agente 2 y compara su tabla Q

Qué comprueba:
  1. INTEGRIDAD: el hash SHA-256 de cada tabla Q guardada coincide con el manifest.
  2. RE-MEDICIÓN: vuelve a enfrentar cada par con semillas DISTINTAS a las del entrenamiento
     (cartas nuevas) y compara con los resultados guardados. Como el agente se congeló mirando
     un conjunto de evaluación, esta es la medición sin sesgo de selección.
  3. CADENAS: recalcula la cadena estricta más larga con la matriz re-medida y revisa TODOS los
     pares de la cadena (p, intervalo de confianza 95%, si superan el umbral).
  4. (--rachas N) estima por simulación directa la probabilidad de ganar la carrera a k manos
     seguidas y la contrasta con la fórmula (que supone manos independientes).
  5. (--reentrenar m) reentrena el agente m con la misma configuración y semillas y comprueba que
     obtiene exactamente la misma tabla Q (reproducibilidad).
"""
import argparse
import hashlib
import json
import os

import numpy as np

from Clases.agente_q import AgenteQ
from Clases.cadenas import (adyacencia, cadena_estricta_mas_larga, matriz_probabilidad,
                            verificar_cadena, wilson_ic)
from Clases.evaluacion import prob_racha, simular_rachas
from Clases.liga import (ConfigLiga, _env_kwargs, cargar_enfrentamientos, construir_pool,
                         enfrentar_nuevo_contra_previos, matriz_para_criterio, nombre_nodo,
                         reentrenar_y_comparar, ruta_agente, ROL_PAR_RIVALES, ROL_UNIFORME_EVAL, _semilla)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("carpeta")
    ap.add_argument("--manos", type=int, default=None, help="manos por par en la re-medición (por defecto, las de la liga)")
    ap.add_argument("--partidas", type=int, default=None, help="partidas por par en la re-medición")
    ap.add_argument("--etiqueta", type=int, default=1, help="cambia las semillas de la re-medición (distinta de 0)")
    ap.add_argument("--rachas", type=int, default=0, help="carreras a simular por cada par de la cadena")
    ap.add_argument("--umbral", type=float, default=None, help="umbral de cadena (por defecto el de la liga)")
    ap.add_argument("--reentrenar", type=int, default=0, help="número de agente a reentrenar y comparar")
    args = ap.parse_args()

    with open(os.path.join(args.carpeta, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    cfg = ConfigLiga.from_dict(manifest["config"])
    if args.umbral is not None:
        cfg.umbral_cadena = args.umbral
    n_hechos = max(int(k) for k in manifest["agentes"])
    n = n_hechos + 1
    print(f"Liga con {n_hechos} agentes + Uniforme | criterio de victoria: {cfg.criterio_victoria} | umbral {cfg.umbral_cadena}")

    # 1) integridad
    print("\n[1] Integridad de los agentes guardados")
    todo_ok = True
    for m in range(1, n_hechos + 1):
        ag = AgenteQ.cargar(ruta_agente(args.carpeta, m))
        ok = ag.hash_q() == manifest["agentes"][str(m)]["hash_q_sha256"]
        todo_ok &= ok
        print(f"   Agente {m}: {'OK' if ok else 'HASH DISTINTO'}  ({manifest['agentes'][str(m)]['episodios']:,} manos de entrenamiento, "
              f"objetivo {'alcanzado' if manifest['agentes'][str(m)]['alcanzo_objetivo'] else 'NO alcanzado'})")

    # 2) re-medición con cartas nuevas
    print(f"\n[2] Re-medición con semillas nuevas (etiqueta {args.etiqueta})")
    G0, Gp0, _, _ = cargar_enfrentamientos(args.carpeta, n)
    G1 = np.zeros_like(G0)
    Gp1 = np.zeros_like(Gp0)
    for m in range(1, n_hechos + 1):
        for f in enfrentar_nuevo_contra_previos(cfg, args.carpeta, m, args.manos, args.partidas, etiqueta=args.etiqueta):
            G1[m, f["b"]], G1[f["b"], m] = f["ganadas_a"], f["ganadas_b"]
            Gp1[m, f["b"]], Gp1[f["b"], m] = f["partidas_ganadas_a"], f["partidas_ganadas_b"]
    base0, crit = matriz_para_criterio(cfg, G0, Gp0)
    base1, _ = matriz_para_criterio(cfg, G1, Gp1)
    z_max, fuera = 0.0, 0
    for i in range(n):
        for j in range(i):
            n0, n1 = base0[i, j] + base0[j, i], base1[i, j] + base1[j, i]
            if n0 == 0 or n1 == 0:
                continue
            p0, p1 = base0[i, j] / n0, base1[i, j] / n1
            se = np.sqrt(max(p0 * (1 - p0) / n0 + p1 * (1 - p1) / n1, 1e-12))
            z = abs(p0 - p1) / se
            z_max = max(z_max, z)
            fuera += z > 3
    print(f"   Diferencia máxima entre la medición original y la nueva: {z_max:.2f} desviaciones estándar; "
          f"pares con |z|>3: {fuera}")
    if fuera:
        print("   (!) Hay pares cuya medición original no se sostiene: sospecha de sesgo de selección o muestra pequeña.")

    # 3) cadenas
    print("\n[3] Cadena estricta más larga con la matriz RE-MEDIDA")
    for conservador in (False, True):
        P = matriz_probabilidad(base1, crit, cfg.k_racha, conservador)
        ady = adyacencia(P, cfg.umbral_cadena)
        cadena = cadena_estricta_mas_larga(ady)
        etiqueta = "conservador (cota inferior IC95)" if conservador else "puntual"
        print(f"   Modo {etiqueta}: {' > '.join(nombre_nodo(x) for x in cadena)}  [{len(cadena)} nodos]")
    P = matriz_probabilidad(base1, crit, cfg.k_racha, False)
    cadena = cadena_estricta_mas_larga(adyacencia(P, cfg.umbral_cadena))
    if len(cadena) > 1:
        print("   Detalle de TODOS los pares de la cadena (el de arriba le gana al de abajo):")
        for fila in verificar_cadena(base1, cadena, crit, cfg.k_racha, cfg.umbral_cadena):
            print(f"     {nombre_nodo(fila['i'])} > {nombre_nodo(fila['j'])}: p={fila['p_mano']:.3f} "
                  f"IC95=[{fila['p_ic_inf']:.3f}, {fila['p_ic_sup']:.3f}] n={fila['manos']}  "
                  f"puntual {'✓' if fila['cumple_puntual'] else '✗'} conservador {'✓' if fila['cumple_conservador'] else '✗'}")
    else:
        print("   Ningún par supera el umbral: no hay cadenas (ni de largo 2).")

    # 4) rachas simuladas
    if args.rachas and len(cadena) > 1:
        print(f"\n[4] Carreras a {cfg.k_racha} manos seguidas: fórmula vs simulación directa ({args.rachas} carreras por par)")
        for a_, b_ in zip(cadena[:-1], cadena[1:]):
            pool, _ = construir_pool(cfg, args.carpeta, max(a_, b_) + 1, ROL_UNIFORME_EVAL, ROL_PAR_RIVALES)
            pol_a = pool[a_] if a_ < len(pool) else None
            rival = pool[b_]
            if pol_a is None:
                continue
            sim = simular_rachas(pol_a, rival, cfg.k_racha, args.rachas, _semilla(cfg, a_, b_, 7), **_env_kwargs(cfg))
            p_mano = G1[a_, b_] / max(G1[a_, b_] + G1[b_, a_], 1)
            print(f"   {nombre_nodo(a_)} vs {nombre_nodo(b_)}: simulada {sim['prob_racha_simulada']:.3f}±{sim['ic95']:.3f} "
                  f"| fórmula con p_mano={p_mano:.3f}: {prob_racha(p_mano, cfg.k_racha):.3f}")

    # 5) reproducibilidad
    if args.reentrenar:
        print(f"\n[5] Reentrenando el Agente {args.reentrenar} desde cero con la misma configuración y semillas...")
        h_nuevo, h_orig, igual = reentrenar_y_comparar(args.carpeta, args.reentrenar, imprimir=lambda *a: None)
        print(f"   hash original : {h_orig}\n   hash reentrenado: {h_nuevo}\n   -> {'IDÉNTICOS: el entrenamiento es reproducible' if igual else 'DISTINTOS (¿otra versión de numpy/Python?)'}")
    print("\nIntegridad:", "OK" if todo_ok else "FALLÓ")


if __name__ == "__main__":
    main()