"""
Evaluación de políticas (agentes).

  evaluar_politica(a, b, n_manos)       -> win rate y bb/mano de `a` contra `b` (manos sueltas)
  evaluar_partidas(a, b, n_partidas)    -> win rate por PARTIDA completa
  prob_racha(p, k)                      -> P(ganar la carrera "primero en ganar k manos seguidas")
  simular_rachas(a, b, k, n_carreras)   -> la misma probabilidad, pero SIMULADA directamente

Tres formas de medir "ganarle" a un rival:
  * por MANO: fracción de manos con ganancia positiva (retirarse = perder la mano).
  * por RACHA: carrera "el primero en ganar k manos seguidas" (k=3 por defecto). Se calcula
    con la fórmula prob_racha desde el win rate por mano (supone manos independientes, lo
    que aquí es cierto porque cada mano reinicia las fichas) y se puede comprobar con
    simular_rachas.
  * por PARTIDA: partidas completas con fichas que se arrastran hasta que alguien quiebra.

Convenciones:
  * Se juegan manos sueltas (fichas reiniciadas, dealer alternado), con una semilla fija:
    así comparar checkpoints distintos usa EXACTAMENTE las mismas cartas.
  * win rate = ganadas / (ganadas + perdidas); los empates exactos no cuentan.
  * bb/mano = ganancia promedio por mano en ciegas grandes.
"""
import numpy as np

from .entorno import PokerHeadsUpEnv
from .simulacion import Jugar, PoliticaAleatoria

#Definimos una función que entrega recompensas de las manos jugadas
def _jugar_mano_eval(politica, env: PokerHeadsUpEnv, obs, info, usa_obs: bool):
    """Termina la mano que `env.reset()` dejó abierta. Devuelve (recompensa, gano)."""
    #Si la mano ya estaba terminada, entonces recuperamos la información de que ganó y su recompensa
    if info["mano_terminada"]:
        return info["recompensa"], info["gano"]
    #Mientras no esté terminada, usa la politica para decidir una acción y luego en base a esa realiza el paso del entrenamiento
    #Si finalizó, recuperamos la recompensa y si no continuamos
    while True:
        accion = politica(obs if usa_obs else None, info["mascara"])
        obs, r, terminado, _, info = env.step(accion)
        if terminado:
            return r, info["gano"]


def evaluar_politica(politica, oponente, n_manos: int = 4000, semilla: int = 12345, **env_kwargs) -> dict:
    #Extraemos el entorno del poker
    env = PokerHeadsUpEnv(oponente=oponente, **env_kwargs)
    #Fijamos los resultados en 0
    ganadas = perdidas = empates = 0
    #Fijamos un vector de tamño número de manos para guardar las recompensas obtenidas
    recompensas = np.empty(n_manos, dtype=np.float64)
    #Buscamos dentro de politica de manera segura si es que usa observaciones para decidir
    usa_obs = getattr(politica, "usa_obs", True)
    #Recorremos las manos
    for i in range(n_manos):
        #Reseteamos la mano y obtenemos los elementos para que el agente juegue
        obs, info = env.reset(seed=semilla if i == 0 else None)
        #Si estamos en la primera mano y el rival sigue una politica aleatoria
        if i == 0 and isinstance(oponente, PoliticaAleatoria):
            #Inyectamos la aleatoriedad del enviroment a nuestro oponente para evaluaciones reproducibles
            oponente.rng = env.np_random 
        #Verificamos la recompensa y si es que ganó
        r, gano = _jugar_mano_eval(politica, env, obs, info, usa_obs)
        #Guardamos la recompensa de la mano i-ésima
        recompensas[i] = r
        #Luego guardamos el reusltado de la mano en su respectivo contador
        if gano is True:
            ganadas += 1
        elif gano is False:
            perdidas += 1
        else:
            empates += 1
    #Calculamos las partidas que se hizo alguna decisión
    decididas = max(ganadas + perdidas, 1)
    #Y obtenemos la proporción de partidas ganas sobre decididas, ie, winrate    
    p = ganadas / decididas
    #Retorna un diccionario con la data necesaria para evaluar el desempeño
    #Particularmente calculamos el margen de error marginal con intervalo de confianza de un 95%
    #Además obtenemos las bb por mano (no podemos usar fichas crudas pues no es informativo respecto a cuanto realmente vale)
    #Recordar que las recompensas en el enviroment ya son entregadas en el formato ganado/ciega grande
    return {
        "n_manos": n_manos,
        "ganadas": ganadas, "perdidas": perdidas, "empates": empates,
        "winrate": p,
        "winrate_ic95": 1.96 * float(np.sqrt(p * (1 - p) / decididas)),
        "bb_por_mano": float(recompensas.mean()),
        "bb_ic95": float(1.96 * recompensas.std(ddof=1) / np.sqrt(n_manos)) if n_manos > 1 else 0.0,
    }

#Evaluamos partidas completas en vez de solo manos
def evaluar_partidas(politica, oponente, n_partidas: int = 500, max_manos: int = 300,
                     semilla: int = 4242, **jugar_kwargs) -> dict:
    """Partidas completas (fichas acumuladas hasta que alguien quiebra, o
    `max_manos`; en ese caso gana quien tenga más fichas). Se alterna quién
    es 'jugador 1' (el que reparte primero) para que sea justo."""
    #Elegimos la semilla e inyectamos esa aleatoriedad al openente si decide aleatoriamente.
    rng = np.random.default_rng(semilla)
    if isinstance(oponente, PoliticaAleatoria):
        oponente.rng = rng
    #Guardamos el número de victorias y las manos
    victorias = 0
    manos_total = 0
    #Recorremos el número de partidas
    for i in range(n_partidas):
        #Alternamos quien es jugador 1 y jugador 2 para que vaya variando el dealer
        if i % 2 == 0:
            partida = Jugar(politica, oponente, nombre_j1="A", nombre_j2="B", max_manos=max_manos,
                            rng=rng, **jugar_kwargs)
        else:
            partida = Jugar(oponente, politica, nombre_j1="B", nombre_j2="A", max_manos=max_manos,
                            rng=rng, **jugar_kwargs)
        #Simulamos la partida
        res = partida.simular()
        #Le sumamos las partidas que se hayan jugado
        manos_total += res["manos_jugadas"]
        #Y quien fue el ganador 
        ganador = res["ganador"]
        #Si no hay ganador, se cortó por máximo de manos y gana quien tenga más fichas.
        if ganador is None:
            f = res["fichas_finales"]
            ganador = max(f, key=f.get) if f["A"] != f["B"] else None
        #Si ganó el agente entonces sumamos la victoria
        if ganador == "A":
            victorias += 1
    #Calculamos su winrate
    p = victorias / n_partidas
    #Y retornamos los estadisticos por número de partidas
    return {"n_partidas": n_partidas, "winrate_partidas": p,
            "winrate_partidas_ic95": 1.96 * float(np.sqrt(p * (1 - p) / n_partidas)),
            "manos_promedio": manos_total / n_partidas}

#Calculamos la probabilidad de racha para ello lo simulamos como cadena de Markov
def prob_racha(p: float, k: int = 3) -> float:
    """Probabilidad de que A gane una carrera 'el primero en ganar k manos
    SEGUIDAS', si A gana cada mano con probabilidad p (manos independientes).
    Se resuelve como cadena de Markov absorbente. Ej.: k=3 -> p=0.60 da 0.73; p=0.71 da ~0.90."""
    #Definimos el estado neutral
    claves = {"0": 0}
    #Definimos la cantidad de partidas en racha de A y B
    for i in range(1, k):
        claves[("A", i)] = len(claves)
    for i in range(1, k):
        claves[("B", i)] = len(claves)
    #Generamos la matriz de transición
    n = len(claves)
    M = np.zeros((n, n))
    b = np.zeros(n)
    #Recorremos los posibles estados para definir las probabilidades de transición
    for s, i in claves.items():
        #Si A gana la siguiente mano con probabilidad p
        a_sig = ("A", 1) if (s == "0" or s[0] == "B") else ("A", s[1] + 1)
        if a_sig[1] == k:
            b[i] += p #A ganó y se agrega a b
        else: #Revisamos la transición al siguiente estado de la racha
            M[i, claves[a_sig]] += p
        #Análogamente para B pero con 1-p
        b_sig = ("B", 1) if (s == "0" or s[0] == "A") else ("B", s[1] + 1)
        if b_sig[1] < k:
            M[i, claves[b_sig]] += 1 - p
    return float(np.linalg.solve(np.eye(n) - M, b)[0])


def simular_rachas(politica, oponente, k: int = 3, n_carreras: int = 1000, semilla: int = 2024,
                   **env_kwargs) -> dict:
    """Simula DIRECTAMENTE la carrera 'primero en ganar k manos seguidas' (sin fórmula) para
    contrastarla con prob_racha. Cada mano es una mano suelta (fichas reiniciadas, dealer
    alternado); los empates exactos se descartan (igual que al calcular el win rate)."""
    #Extraemos el ambiente de poker
    env = PokerHeadsUpEnv(oponente=oponente, **env_kwargs)
    #Verficamos si usa observacion
    usa_obs = getattr(politica, "usa_obs", True)
    #Inicializamos contadores
    ganadas_a = manos_total = 0
    #Estas carreras consisten en jugar una serie de manos de corrido hasta que alguien alcance la racha
    primera = True
    for _ in range(n_carreras):
        #Inicializamos las rachas en 0
        racha_a = racha_b = 0
        while True:
            #Seteamos los ambientes de cero
            obs, info = env.reset(seed=semilla if primera else None)
            #Si estamos en la primera y hay politica aleatoria inyectamos aleatoriedad
            if primera:
                primera = False
                if isinstance(oponente, PoliticaAleatoria):
                    oponente.rng = env.np_random
            #Jugamos la mano       
            _, gano = _jugar_mano_eval(politica, env, obs, info, usa_obs)
            manos_total += 1
            #Dependiendo de quien gana actualizamos sus rachas y reseteamos la otra.
            if gano is True:
                racha_a, racha_b = racha_a + 1, 0
            elif gano is False:
                racha_a, racha_b = 0, racha_b + 1
            else:
                continue
            #Si superamos la cota de rachas entonces ganó una partida bajo esta condición
            if racha_a >= k:
                ganadas_a += 1
                break
            if racha_b >= k:
                break
    p = ganadas_a / n_carreras
    #Retornamos el diccionario con los datos
    return {"k": k, "n_carreras": n_carreras, "prob_racha_simulada": p,
            "ic95": 1.96 * float(np.sqrt(p * (1 - p) / n_carreras)),
            "manos_por_carrera": manos_total / n_carreras}
