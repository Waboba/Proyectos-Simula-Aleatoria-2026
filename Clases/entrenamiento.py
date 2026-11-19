"""
Una mano de entrenamiento de Q-learning.

El bucle completo (rivales, criterio de parada, checkpoints, métricas) vive en liga.py;
aquí solo está la unidad básica: jugar UNA mano con el agente explorando (epsilon-greedy)
y actualizar su tabla Q.
"""
from .agente_q import AgenteQ
from .entorno import PokerHeadsUpEnv

#Función que permite al agente jugar una mano completa y recibir sus respectivas recompensas
def jugar_mano_entrenamiento(agente: AgenteQ, env: PokerHeadsUpEnv) -> tuple[float, int]:
    """Juega una mano (env.reset() + decisiones) actualizando `agente.Q` con Q-learning.
    Devuelve (recompensa de la mano, número de decisiones del agente).

    Si el rival se retira antes de que el agente decida (info["mano_terminada"]), no hay
    decisión que aprender, pero la mano igual cuenta (recompensa real, 0 decisiones)."""
    #Reseteamos la mano y le entregamos lo necesario al agente para que pueda jugar
    obs, info = env.reset()
    #Si la mano ya estaba terminada, devolvemos su respectiva recompensa y le avisamos que no tomó ninguna decisión
    if info["mano_terminada"]:
        return float(info["recompensa"]), 0
    #El agente toma su primera decisión en base al estado de la partida y la máscara de acciones permitidas
    estado, accion = agente.elegir(obs, info["mascara"], explorar=True)
    #Empezamos a contar las decisiones que tomó el agente
    decisiones = 0
    while True:
        #Avanzamos un paso
        obs2, r, terminado, _, info2 = env.step(accion)
        #Aumentamos el número de decisiones tomadas por el agente
        decisiones += 1
        #Si tras el paso, la mano terminó
        if terminado:
            #Actualizamos la Q-table dependiendo del estado la acción y la recompensa obtenida
            agente.actualizar(estado, accion, r, None, None, True)
            #Retorna la recompensa y cuantas decisiones tomó
            return float(r), decisiones
        #Si la mano no ha terminado, entonces tomamos otra decisión
        estado2, accion2 = agente.elegir(obs2, info2["mascara"], explorar=True)
        #En base a ese estado actualizamos las recompensas parciales de la Q-table en base al estado y la acción respectia
        agente.actualizar(estado, accion, r, estado2, info2["mascara"], False)
        #Iteramos denuevo
        estado, accion = estado2, accion2


# ----------------------------------------------------------------------------
# Bucle de entrenamiento simple con checkpoints y métricas
# ----------------------------------------------------------------------------
# (liga.py tiene su propio bucle, con criterio de parada y rivales congelados;
#  este es el bucle genérico: N manos contra el rival del entorno.)
import csv
import json
import os

import numpy as np

from .evaluacion import evaluar_partidas, evaluar_politica

#Definimos ahora una función que busque el mejor escore previo registrado en el csv para solo guardar mejoras.
def _mejor_score_previo(ruta_csv: str, nombres: list[str]) -> float:
    """Mejor score (promedio de winrate_partidas_* entre rivales) ya registrado en el CSV."""
    mejor = -float("inf")
    if os.path.exists(ruta_csv):
        with open(ruta_csv, newline="", encoding="utf-8") as f:
            for fila in csv.DictReader(f):
                try:
                    mejor = max(mejor, float(np.mean([float(fila[f"winrate_partidas_{n}"]) for n in nombres])))
                except (KeyError, ValueError):
                    pass
    return mejor

#Definimos la función que entrena
def entrenar_q(agente: AgenteQ, env: PokerHeadsUpEnv, n_episodios: int, directorio: str,
               rivales_eval: dict, eval_cada: int = 1000, eval_manos: int = 2000,
               eval_partidas: int = 100, checkpoint_cada: int = 5000,
               eps_final: float = 0.05, episodios_decaimiento: int | None = None,
               imprimir=print) -> None:
    """Entrena `agente` n_episodios manos contra el rival de `env`.

    Cada `eval_cada` manos evalúa (greedy) contra `rivales_eval` ({nombre: política}) y añade
    una fila a metricas.csv (episodio, winrate_partidas_<n>, bb_por_mano_<n>); guarda
    ultimo.npz y, si mejora el promedio de winrate por partida, mejor.npz. Cada
    `checkpoint_cada` manos guarda checkpoint_epNNNNNNNNN.npz. Reanudable: usa
    agente.episodios como contador global y añade filas al mismo CSV.

    Si `episodios_decaimiento` no es None, epsilon decae linealmente hasta `eps_final`."""
    #Creamos el directorio si es que no existe
    os.makedirs(directorio, exist_ok=True)
    #Vemos cuantos episodios (manos completas) ha jugado nuestro rival
    ep_inicio = agente.episodios
    #Generamos la lista de los rivales a los que se enfrentará el agente
    nombres = list(rivales_eval)
    #Y la ruta donde guardaremos las métricas
    ruta_csv = os.path.join(directorio, "metricas.csv")
    #Guardaremos el episodio, los nombres nuestro winrate y bb por mano contra ese agente particular
    columnas = ["episodio"] + [c for n in nombres for c in (f"winrate_partidas_{n}", f"bb_por_mano_{n}")]
    #Extraemos el factor de exploración/explotación
    eps0 = agente.epsilon

    #Extraemos de un json todos los hiperparametros de nuestro entrenamiento
    with open(os.path.join(directorio, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"ep_inicio": ep_inicio, "n_episodios": n_episodios, "alpha": agente.alpha,
                   "gamma": agente.gamma, "eval_cada": eval_cada, "eval_manos": eval_manos,
                   "eval_partidas": eval_partidas, "checkpoint_cada": checkpoint_cada}, f, indent=2)
    #Buscamos el mejor score previo
    mejor = _mejor_score_previo(ruta_csv, nombres)
    #Revisamos si tenemos progreso antes
    nuevo_csv = not os.path.exists(ruta_csv)
    #Abrimos la ruta
    with open(ruta_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columnas)
        #Si tenemos un nuevo csv escribimos el header
        if nuevo_csv:
            w.writeheader()
        #Si no, iteramos sobre los episodios
        for _ in range(n_episodios):
            #Si decaemos, entonces el epsilon baja hasta epsilon final
            if episodios_decaimiento:
                agente.epsilon = max(eps_final, eps0 - (eps0 - eps_final) * (agente.episodios - ep_inicio) / episodios_decaimiento)
            #Luego jugamos la mano, aumentamos los episodios 
            jugar_mano_entrenamiento(agente, env)
            agente.episodios += 1
            ep = agente.episodios
            #Si nuestras eval cada divide entonces cada esa cantidad de pasos evaluamos
            if ep % eval_cada == 0:
                #Creamos una versión greedy del agente, ie, solo juega por el mayor valor de la Q-table
                #Además, esta vista no afecta al rng del entrenamiento al usar la aleatoriedad basada en el episodio
                vista = agente.vista_greedy(np.random.default_rng(ep))
                fila, wr = {"episodio": ep}, []
                #Por cada rival a enfrentar evaluamos el rendimiento del agente en evaluaciones por partidas y en evaluaciones por politica
                for nombre, rival in rivales_eval.items():
                    #Evaluamos las partidas (hasta quedarse sin fichas) y las manos, midiendo winrate y bb_por_mano
                    p = evaluar_partidas(vista, rival, n_partidas=eval_partidas)
                    m = evaluar_politica(vista, rival, n_manos=eval_manos)
                    fila[f"winrate_partidas_{nombre}"] = p["winrate_partidas"]
                    fila[f"bb_por_mano_{nombre}"] = m["bb_por_mano"]
                    wr.append(p["winrate_partidas"])
                #Escribimos los datos
                w.writerow(fila)
                #Y forzamos a que se escriban en disco
                f.flush()
                imprimir(f"ep {ep}: " + " ".join(f"{k}={v:.3f}" for k, v in fila.items() if k != "episodio"))
                #El score se calcula como el promedio de los winrates entre todos los agentes
                score = float(np.mean(wr)) if wr else 0.0
                #Si nuestro score es mejor o no habíamos guardado nada, entonces guardamos el nuevo mejor y si no simplemente guardamos como el ultimo
                if score > mejor or not os.path.exists(os.path.join(directorio, "mejor.npz")):
                    mejor = max(mejor, score)
                    agente.guardar(os.path.join(directorio, "mejor.npz"))
                agente.guardar(os.path.join(directorio, "ultimo.npz"))
            #Si tenemos guardado checkpoint
            if ep % checkpoint_cada == 0:
                agente.guardar(os.path.join(directorio, f"checkpoint_ep{ep:09d}.npz"))
    #Guardamos el último.npz 
    agente.guardar(os.path.join(directorio, "ultimo.npz"))
    if not os.path.exists(os.path.join(directorio, "mejor.npz")):
        agente.guardar(os.path.join(directorio, "mejor.npz"))