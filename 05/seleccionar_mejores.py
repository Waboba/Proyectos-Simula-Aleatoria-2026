"""
Recorre todas las ligas entrenadas dentro de `resultados/`, identifica los mejores
agentes de cada liga y los copia en una carpeta destino manteniendo exactamente
el formato estándar (`agente_01.npz`, `agente_02.npz`, ...).

Uso:
    python seleccionar_mejores.py
    python seleccionar_mejores.py --destino resultados/mejores_agentes --limpiar
    python seleccionar_mejores.py --modo cadena --limpiar
    python seleccionar_mejores.py --modo top_k --k 2 --limpiar
"""
import argparse
import csv
import json
import os
import shutil
import numpy as np

from Clases.cadenas import adyacencia, cadena_estricta_mas_larga, matriz_probabilidad
from Clases.liga import (
    ConfigLiga,
    cargar_enfrentamientos,
    matriz_para_criterio,
    nombre_nodo,
    ruta_agente,
)


def evaluar_liga(carpeta_liga: str, modo: str, k: int, umbral: float | None, conservador: bool) -> list[dict]:
    manifest_path = os.path.join(carpeta_liga, "manifest.json")
    if not os.path.exists(manifest_path):
        return []

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    if not manifest.get("agentes"):
        return []

    cfg = ConfigLiga.from_dict(manifest["config"])
    u = umbral if umbral is not None else cfg.umbral_cadena
    n_hechos = max(int(idx) for idx in manifest["agentes"])
    n_nodos = n_hechos + 1

    G, Gp, _, _ = cargar_enfrentamientos(carpeta_liga, n_nodos)
    base, crit = matriz_para_criterio(cfg, G, Gp)
    P = matriz_probabilidad(base, crit, cfg.k_racha, conservador)

    wr_promedio = {}
    for m in range(1, n_hechos + 1):
        rivales = [j for j in range(n_nodos) if j != m]
        wr_promedio[m] = float(np.mean([P[m, j] for j in rivales])) if rivales else 0.0

    cadena = cadena_estricta_mas_larga(adyacencia(P, u))
    agentes_cadena = [nodo for nodo in cadena if nodo != 0]

    seleccionados: list[tuple[int, str]] = []
    if modo == "cima_cadena":
        if agentes_cadena:
            seleccionados.append((agentes_cadena[0], "cima_cadena_estricta"))
        else:
            mejor = max(wr_promedio, key=wr_promedio.get)
            seleccionados.append((mejor, "fallback_mejor_wr_promedio"))
    elif modo == "cadena":
        if agentes_cadena:
            for pos, m in enumerate(agentes_cadena, 1):
                seleccionados.append((m, f"cadena_pos_{pos}_de_{len(agentes_cadena)}"))
        else:
            mejor = max(wr_promedio, key=wr_promedio.get)
            seleccionados.append((mejor, "fallback_mejor_wr_promedio"))
    elif modo == "ultimo":
        seleccionados.append((n_hechos, "ultimo_agente_liga"))
    elif modo == "top_k":
        ordenados = sorted(wr_promedio.keys(), key=lambda m: wr_promedio[m], reverse=True)
        for rank, m in enumerate(ordenados[:k], 1):
            seleccionados.append((m, f"top_{rank}_wr_promedio"))

    resultados = []
    nombre_liga = os.path.basename(os.path.normpath(carpeta_liga))
    for m, motivo in seleccionados:
        src = ruta_agente(carpeta_liga, m)
        if not os.path.exists(src):
            continue
        info_ag = manifest["agentes"][str(m)]
        resultados.append({
            "liga": nombre_liga,
            "agente_num_origen": m,
            "motivo": motivo,
            "criterio": cfg.criterio_victoria,
            "umbral": u,
            "wr_promedio": round(wr_promedio[m], 4),
            "wr_vs_uniforme": round(float(P[m, 0]), 4),
            "info_manifest": info_ag,
            "ruta_origen": src,
        })
    return resultados


def main() -> None:
    ap = argparse.ArgumentParser(description="Copia los mejores agentes con formato estándar agente_01.npz, agente_02.npz, ...")
    ap.add_argument("--origen", default="resultados", help="Carpeta raíz donde están las ligas")
    ap.add_argument("--destino", default=os.path.join("resultados", "mejores_agentes"),
                    help="Carpeta donde se guardarán las copias")
    ap.add_argument("--modo", choices=("cima_cadena", "cadena", "top_k", "ultimo"), default="cima_cadena",
                    help="Criterio de selección")
    ap.add_argument("--k", type=int, default=1, help="Cantidad de agentes por liga si --modo top_k")
    ap.add_argument("--umbral", type=float, default=None, help="Umbral de dominancia")
    ap.add_argument("--conservador", action="store_true", help="Usar cota inferior de Wilson al 95%%")
    ap.add_argument("--limpiar", action="store_true", help="Vaciar la carpeta destino antes de copiar")
    args = ap.parse_args()

    if not os.path.isdir(args.origen):
        raise FileNotFoundError(f"No existe la carpeta de origen: {args.origen}")

    if args.limpiar and os.path.exists(args.destino):
        shutil.rmtree(args.destino)
    os.makedirs(args.destino, exist_ok=True)

    destino_abs = os.path.abspath(args.destino)
    candidatos = []

    for raiz, _, archivos in os.walk(args.origen):
        if os.path.abspath(raiz).startswith(destino_abs):
            continue
        if "manifest.json" in archivos:
            candidatos.extend(evaluar_liga(raiz, args.modo, args.k, args.umbral, args.conservador))

    if not candidatos:
        print(f"No se encontraron ligas válidas con manifest.json dentro de '{args.origen}'.")
        return

    # Ordenar de mayor a menor win rate promedio para que agente_01.npz sea el mejor de todos
    candidatos.sort(key=lambda x: x["wr_promedio"], reverse=True)

    agentes_manifest = {}
    filas_csv = []

    for idx_nuevo, item in enumerate(candidatos, start=1):
        nombre_archivo = f"agente_{idx_nuevo:02d}.npz"
        ruta_dest = os.path.join(args.destino, nombre_archivo)
        shutil.copy2(item["ruta_origen"], ruta_dest)

        info = dict(item["info_manifest"])
        info["origen_liga"] = item["liga"]
        info["origen_agente"] = item["agente_num_origen"]
        info["motivo_seleccion"] = item["motivo"]
        agentes_manifest[str(idx_nuevo)] = info

        filas_csv.append({
            "archivo": nombre_archivo,
            "liga_origen": item["liga"],
            "agente_origen": f"agente_{item['agente_num_origen']:02d}.npz",
            "motivo": item["motivo"],
            "criterio": item["criterio"],
            "wr_promedio": item["wr_promedio"],
            "wr_vs_uniforme": item["wr_vs_uniforme"],
            "ruta_origen": item["ruta_origen"],
        })

    # Guardar manifest.json compatible con la estructura de liga
    manifest_destino = {
        "descripcion": "Mejores agentes seleccionados de resultados/",
        "modo_seleccion": args.modo,
        "agentes": agentes_manifest,
    }
    with open(os.path.join(args.destino, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest_destino, f, indent=2, ensure_ascii=False)

    # Guardar índice CSV para saber de qué liga vino cada agente_XX.npz
    csv_path = os.path.join(args.destino, "indice_origen.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(filas_csv[0].keys()))
        writer.writeheader()
        writer.writerows(filas_csv)

    print(f"Se copiaron {len(filas_csv)} agentes en '{args.destino}' con formato estándar:")
    for fila in filas_csv:
        print(f"  - {fila['archivo']} <- {fila['liga_origen']}/{fila['agente_origen']} "
              f"({fila['motivo']}, WR prom: {fila['wr_promedio']:.3f})")


if __name__ == "__main__":
    main()
    