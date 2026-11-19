import subprocess
import sys
from itertools import product

def ejecutar_grilla():
    # --- GRIEGA DE EXPERIMENTOS A PROBAR ---
    # Puedes ajustar estos valores según lo que quieras explorar
    pesos_fichas = [0,0.25, 0.5, 0.75,1]
    n_agentes_lista = [5, 10, 15]         # Largo de la liga
    max_manos_lista = [3, 5, 10]    
    # Genera todas las combinaciones posibles entre las listas
    combinaciones = list(product(pesos_fichas, n_agentes_lista, max_manos_lista))
    
    total = len(combinaciones)
    print(f"[*] Se ejecutarán un total de {total} experimentos en la grilla.")

    for i, (w, n_ag, max_m) in enumerate(combinaciones, 1):
        # Creamos un nombre de carpeta descriptivo basado en los parámetros
        s_max_m = f"m{max_m}" if max_m is not None else "inf"
        nombre_carpeta = f"resultados/grid_w_{w}_nag_{n_ag}_manos_{s_max_m}".replace(".", "_")
        
        print(f"\n==================================================")
        print(f" [{i}/{total}] Experimento: peso={w} | n_agentes={n_ag} | max_manos={max_m}")
        print(f" Carpeta: {nombre_carpeta}")
        print(f"==================================================")
        
        # Armamos el comando base
        comando = [
            sys.executable, "entrenar_liga.py",
            nombre_carpeta,
            "--n-agentes", str(n_ag),
            "--recompensa", "mixta",
            "--peso-fichas", str(w),
            "--criterio", "racha"
        ]
        
        # Agregamos max-manos-partida solo si está definido
        if max_m is not None:
            comando.extend(["--max-manos-partida", str(max_m)])
            
        # Opcional: si quieres que las pruebas sean rápidas mientras depuras, 
        # puedes descomentar la siguiente línea:
        # comando.append("--rapida")

        # Ejecuta el proceso
        resultado = subprocess.run(comando)
        
        if resultado.returncode != 0:
            print(f"[AVISO] El experimento {i} finalizó con código de error {resultado.returncode}")

if __name__ == "__main__":
    ejecutar_grilla()