"""
Liga de agentes con congelamiento sucesivo.

Procedimiento (ver entrenar_liga.py para el uso):

  Agente 1 : empieza en cero; entrena contra el UNIFORME hasta alcanzar el objetivo
             (60%); se congela.
  Agente m : copia los pesos (tabla Q) del agente m-1 y entrena contra un rival que en
             CADA MANO se elige al azar entre {Uniforme, Agente 1, ..., Agente m-1}
             (todos congelados) hasta alcanzar el objetivo; se congela.
  Al congelar el agente m se mide, con un número FIJO de manos, su resultado contra cada
  rival anterior y se actualiza el análisis de cadenas.

Archivos en la carpeta de la liga:
  manifest.json              configuración, versiones, semillas, hash de cada agente
  agentes/agente_MM.npz      tabla Q congelada de cada agente
  entrenamiento/agente_MM/   metricas.csv + checkpoints del entrenamiento de ese agente
  enfrentamientos.csv        resultado de cada par (agente nuevo vs cada rival anterior): manos Y partidas
  iteraciones.csv            una fila por agente: manos hasta el objetivo, win rates, cadenas
  cadenas.csv                cadena estricta más larga por iteración, umbral y modo

Nodos: 0 = Uniforme, m = Agente m.  

Reproducibilidad: TODA la aleatoriedad sale de semillas derivadas de (semilla, agente, rol);
los rivales congelados se cargan siempre desde disco con su generador re-sembrado. Por eso
reentrenar el agente m con la misma configuración reproduce la misma tabla Q (mismo hash).
"""
import csv
import json
import os
import platform
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, fields

import numpy as np

from .agente_q import AgenteQ
from .cadenas import (adyacencia, cadena_estricta_mas_larga, matriz_probabilidad,
                      transitividad)
from .entorno import PokerHeadsUpEnv
from .entrenamiento import jugar_mano_entrenamiento
from .evaluacion import evaluar_partidas, evaluar_politica, prob_racha
from .simulacion import PoliticaAleatoria, PoliticaMezcla

NOMBRE_UNIFORME = "Uniforme"

# roles para derivar semillas independientes
(ROL_AGENTE, ROL_ENV, ROL_UNIFORME_ENT, ROL_CONG_ENT, ROL_UNIFORME_EVAL, ROL_CONG_EVAL,
 ROL_PARADA, ROL_PAR, ROL_PAR_RIVALES) = range(1, 10)

#Extraemos los nombres de los agentes
def nombre_nodo(i: int) -> str:
    return NOMBRE_UNIFORME if i == 0 else f"Agente {i}"


@dataclass
class ConfigLiga:
    n_agentes: int = 10
    semilla: int = 0
    # --- definición de "ganar" (se usa para el criterio de parada Y para las cadenas) ---
    #Particularmente modificar esta condición de victoria pues es muy exigente y no aprende el agente
    criterio_victoria: str = "partida"
    k_racha: int = 3
    max_manos_partida: int = 3
    # --- objetivo / criterio de parada del entrenamiento de cada agente ---
    objetivo: float = 0.60
    criterio_parada: str = "cada_rival"
    max_episodios: int = 500_000
    min_episodios: int = 20_000
    #Cantidad de episodios que se juegan previo a evaluación
    bloque: int = 10_000
    #Para evitar falsos positivos de superar el umbral
    confirmaciones: int = 2
    manos_eval: int = 8_000
    partidas_eval: int = 1_200
    detener_si_no_alcanza: bool = False
    # --- aprendizaje ---
    alpha: float = 0.12
    alpha_min: float = 0.01
    decaimiento_alpha: float = 0.01
    gamma: float = 1.0
    eps_inicial: float = 1.0            #agente 1 (parte de cero)
    eps_inicial_warm: float = 0.20      #agentes 2.. (refina sin destruir lo aprendido)
    eps_final: float = 0.03
    episodios_decaimiento: int = 250_000
    tau_eval: float = 0.0               #greedy puro al evaluar/congelar
    peso_ultimo_rival: float = 0.45     #45% contra Agente m-1, 55% repartido entre 0..m-2 (Fuerza la evolución)
    # "auto" usa "victoria" (+1/-1) si criterio_victoria es "racha"/"mano",
    # y "fichas" si criterio_victoria es "partida" (alinea entrenamiento y evaluación)
    recompensa: str = "fichas"
    # --- rival uniforme ---
    rival_uniforme: str = "uniforme"
    peso_retiro_uniforme: float | None = None
    # --- mesa ---
    fichas_iniciales: int = 1000
    ciega_pequena: int = 10
    ciega_grande: int = 20
    permitir_retiro: bool = True
    # --- enfrentamientos al congelar cada agente ---
    manos_total: int = 18_000 #Antes era 60_000 
    manos_por_par: int | None = None
    partidas_total: int = 6_000
    partidas_por_par: int | None = None
    checkpoint_cada: int = 100_000
    # --- cadenas ---
    umbral_cadena: float = 0.9
    umbrales: tuple = (0.6, 0.7, 0.8, 0.9)
    # --- nuevos parámetros para experimentos ---
    peso_fichas: float = 0.5

    #Creamos una función para verificar que todo los paramétros estén en orden
    def __post_init__(self) -> None:
        if self.criterio_victoria not in ("partida", "mano", "racha"):
            raise ValueError("criterio_victoria debe ser 'partida', 'mano' o 'racha'")
        if self.criterio_parada not in ("cada_rival", "mezcla"):
            raise ValueError("criterio_parada debe ser 'cada_rival' o 'mezcla'")
        if self.recompensa not in ("auto", "fichas", "victoria", "mixta"):
            raise ValueError("recompensa debe ser 'auto', 'fichas' o 'victoria'")

    #Obtenemos como recibe recompensas el agente
    def modo_recompensa(self) -> str:
        if self.recompensa != "auto":
            return self.recompensa
        return "fichas" if self.criterio_victoria == "partida" else "victoria"
    #Mandamos la configuración de la liga a un diccionario
    def to_dict(self) -> dict:
        d = asdict(self)
        d["umbrales"] = list(self.umbrales)
        return d
    #Extraemos de un diccionario una configuración válida del diccionari
    @classmethod
    def from_dict(cls, d: dict) -> "ConfigLiga":
        nombres = {f.name for f in fields(cls)}
        d = {k: v for k, v in d.items() if k in nombres}
        if "umbrales" in d:
            d["umbrales"] = tuple(d["umbrales"])
        return cls(**d)

    #Una función que verifica que la liga funcione correctamente
    @classmethod
    def prueba_rapida(cls, **kw) -> "ConfigLiga":
        base = dict(n_agentes=3, max_episodios=20_000, min_episodios=5_000, bloque=5_000, manos_eval=2_000,
                    partidas_eval=300, manos_total=6_000, partidas_total=600, checkpoint_cada=10_000,
                    episodios_decaimiento=15_000)
        base.update(kw)
        return cls(**base)


# ----------------------------------------------------------------------------
# utilidades
# ----------------------------------------------------------------------------

#Seteamos la semilla
def _semilla(cfg: ConfigLiga, *claves: int) -> int:
    return int(np.random.SeedSequence([cfg.semilla, *claves]).generate_state(1)[0])

#Empaqueta los parametros útiles del enviroment
def _env_kwargs(cfg: ConfigLiga) -> dict:
    return {"fichas_iniciales": cfg.fichas_iniciales, "ciega_pequena": cfg.ciega_pequena,
            "ciega_grande": cfg.ciega_grande, "permitir_retiro": cfg.permitir_retiro}

#Genera una ruta reproducible y estandarizada para un agente
def ruta_agente(carpeta: str, m: int) -> str:
    return os.path.join(carpeta, "agentes", f"agente_{m:02d}.npz")

#Generamos el rival uniforme
def crear_uniforme(cfg: ConfigLiga, semilla: int) -> PoliticaAleatoria:
    rng = np.random.default_rng(semilla)
    if cfg.rival_uniforme == "uniforme":
        return PoliticaAleatoria.uniforme(rng, peso_retiro=cfg.peso_retiro_uniforme)
    return PoliticaAleatoria(rng, peso_retiro=cfg.peso_retiro_uniforme)

#Genera la pool de enfrentamientos del agente
def construir_pool(cfg: ConfigLiga, carpeta: str, m: int, rol_uniforme: int, rol_congelados: int):
    """Rivales del agente m: [Uniforme, Agente 1, ..., Agente m-1], con los agentes cargados
    desde disco y sus generadores re-sembrados (deterministas)."""
    #Creamos el rival uniforme
    pool = [crear_uniforme(cfg, _semilla(cfg, m, rol_uniforme))]
    #Para cada agente cargamos su versión congelada
    for i in range(1, m):
        ag = AgenteQ.cargar(ruta_agente(carpeta, i), np.random.default_rng(_semilla(cfg, m, rol_congelados, i)))
        #Los fijamos en modo determinista
        ag.epsilon = 0.0
        #Lo agregamos
        pool.append(ag)
    return pool, [nombre_nodo(i) for i in range(m)]

#Si queremos reiniciar la aleatoriedad por una nueva
def _reiniciar_rng(politica, semilla: int) -> None:
    politica.rng = np.random.default_rng(semilla)

#Escribe el csv
def _escribir_csv(ruta: str, columnas: list[str], filas: list[dict], anexar: bool = True) -> None:
    existe = os.path.exists(ruta) and anexar
    with open(ruta, "a" if anexar else "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columnas)
        if not existe:
            w.writeheader()
        for fila in filas:
            w.writerow({c: fila.get(c, "") for c in columnas})

#Establecemos las columnas para medir  los enfrentamientos tanto a nivel de mano como de partidas
COL_ENFRENTAMIENTOS = ["iteracion", "a", "b", "nombre_a", "nombre_b", "manos", "ganadas_a", "ganadas_b",
                       "empates", "winrate_a", "winrate_ic95", "bb_por_mano_a", "bb_ic95",
                       "partidas", "partidas_ganadas_a", "partidas_ganadas_b", "winrate_partida_a",
                       "winrate_partida_ic95", "manos_por_partida"]
#Establecemos las columnas para medir los largos de cadenas
COL_CADENAS = ["iteracion", "criterio", "k", "conservador", "umbral", "largo_agentes", "largo_nodos",
               "cadena", "caminos2", "caminos2_cerrados", "frac_transitiva"]
#Establecemos parametros de estudio del entrenamiento de los agente
COL_ITERACIONES = ["iteracion", "episodios", "alcanzo_objetivo", "tiempo_s", "winrate_mezcla_parada",
                   "winrate_min_parada", "winrate_vs_anterior", "winrate_vs_uniforme", "bb_vs_anterior",
                   "largo_cadena", "largo_cadena_conservador", "frac_transitiva", "hash_q"]

#Generamos matrices que resumen el comportamiento entre los agentes i y j
def cargar_enfrentamientos(carpeta: str, n_nodos: int):
    """(G, Gp, E, filas): G[i,j] = MANOS ganadas por i a j; Gp[i,j] = PARTIDAS ganadas por i a j;
    E[i,j] = empates de mano."""
    #Primero, G refleja cuantas manos le gana i a j
    G = np.zeros((n_nodos, n_nodos), dtype=np.int64)
    #Luego, Gp cuantas partidas le ganó i a j
    Gp = np.zeros((n_nodos, n_nodos), dtype=np.int64)
    #E cuantas partidas empató i y j
    E = np.zeros((n_nodos, n_nodos), dtype=np.int64)
    #Una lista para procesar los diccionarios
    filas = []
    #Leeamos el csv enfrentamientos y rellenamos las matrices
    ruta = os.path.join(carpeta, "enfrentamientos.csv")
    if os.path.exists(ruta):
        with open(ruta, newline="", encoding="utf-8") as f:
            for fila in csv.DictReader(f):
                a, b = int(fila["a"]), int(fila["b"])
                G[a, b], G[b, a] = int(fila["ganadas_a"]), int(fila["ganadas_b"])
                Gp[a, b], Gp[b, a] = int(fila["partidas_ganadas_a"]), int(fila["partidas_ganadas_b"])
                E[a, b] = E[b, a] = int(fila["empates"])
                filas.append(fila)
    return G, Gp, E, filas

#Función que recupera el valor que necesitamos si el agente cumplió la cuota de winrate
def valor_criterio(cfg: ConfigLiga, winrate_mano: float | None, winrate_partida: float | None) -> float:
    """Valor que se compara con el objetivo según el criterio de victoria."""
    if cfg.criterio_victoria == "partida":
        return winrate_partida
    if cfg.criterio_victoria == "racha":
        return prob_racha(winrate_mano, cfg.k_racha)
    return winrate_mano

#Llama a la función anterior sobre una fila especifica
def _valor_criterio_fila(cfg: ConfigLiga, fila: dict) -> float:
    return valor_criterio(cfg, fila["winrate_a"], fila["winrate_partida_a"])


# ----------------------------------------------------------------------------
# entrenamiento de UN agente
# ----------------------------------------------------------------------------
#Entrenamos al agente hasta conseguir el objetivo o que se hayan completado el máximo de episodios de entrenamiento
def entrenar_agente_liga(cfg: ConfigLiga, carpeta: str, m: int, imprimir=print) -> dict:
    """Entrena el agente m hasta alcanzar el objetivo (o max_episodios) y lo guarda
    congelado en agentes/agente_MM.npz. Requiere agentes/agente_01..m-1 en `carpeta`."""
    #Creamos el directorio agentes si es que no existe para guardar estos entrenados
    os.makedirs(os.path.join(carpeta, "agentes"), exist_ok=True)
    #Agregamos un directorio de entrenamiento para cada agente
    dir_ent = os.path.join(carpeta, "entrenamiento", f"agente_{m:02d}")
    os.makedirs(dir_ent, exist_ok=True)
    #Buscamos la ruta de métricas del entrenamiento
    ruta_csv = os.path.join(dir_ent, "metricas.csv")
    #Si ya existía entonces la borramos y reentrenamos
    if os.path.exists(ruta_csv):
        os.remove(ruta_csv)  # reentrenar = empezar de nuevo
    #Creamos el agente a partir de nuestra clase ya generada. Este extrae los parametros principales de la configuración de la liga
    agente = AgenteQ(
        nombre_nodo(m),
        cfg.alpha,
        cfg.gamma,
        rng=np.random.default_rng(_semilla(cfg, m, ROL_AGENTE)),
        alpha_min=cfg.alpha_min,
        decaimiento_alpha=cfg.decaimiento_alpha,
        tau_eval=cfg.tau_eval,
    )
    if m > 1:  #hereda pesos y conserva 60% de la confianza (visitas) para no olvidar al Uniforme
        #Si no es el primero, entonces hereda lo ya aprendido de los anteriores
        prev = AgenteQ.cargar(ruta_agente(carpeta, m - 1))
        #Copiamos su Q-table
        agente.Q = prev.Q.copy()
        #Copiamos sus visitas 
        agente.visitas = prev.visitas.copy()
        #Y generamos el warm start que permite al agente no ser del todo terco con respecto a aprender
        agente.preparar_warm_start(factor_visitas=0.60)
    #Definimos el factor de curiosidad
    eps0 = cfg.eps_inicial if m == 1 else cfg.eps_inicial_warm

    #Generamos las pools de entrenamiento y evaluación respectivas
    pool_ent, nombres = construir_pool(cfg, carpeta, m, ROL_UNIFORME_ENT, ROL_CONG_ENT)
    pool_eval, _ = construir_pool(cfg, carpeta, m, ROL_UNIFORME_EVAL, ROL_CONG_EVAL)

    #Prioriza al rival más reciente (m-1) sin dejar de jugar contra los anteriores (evita olvido)
    if m == 1:
        pesos_pool = [1.0]
    else:
        resto = (1.0 - cfg.peso_ultimo_rival) / (m - 1)
        pesos_pool = [resto] * (m - 1) + [cfg.peso_ultimo_rival]

    #Traemos la política mezcla de rivales
    mezcla = PoliticaMezcla(pool_ent, nombres, pesos=pesos_pool)
    #Traemos el ambiente de Gymnasium
    env = PokerHeadsUpEnv(oponente=mezcla, modo_recompensa=cfg.modo_recompensa(),peso_fichas=cfg.peso_fichas, **_env_kwargs(cfg))
    #Reseteamos el entorno con una semilla única
    env.reset(seed=_semilla(cfg, m, ROL_ENV))
    #Seteamos las columnas del csv de métricas
    columnas = (["episodio", "epsilon", "recompensa_entreno_media", "estados_visitados", "tiempo_s", "valor_mezcla"]
                + [f"valor_vs_{j}" for j in range(m)] + [f"ic95_vs_{j}" for j in range(m)])
    #Inicializamos el estado mutable para guardar la última evaluación y que sea de fácil acceso
    estado = {"ultima": None}

    def chequear(t0: float, suma_r: float, n_r: int) -> bool:
        """Evalúa al agente (greedy) contra cada rival con un número FIJO de manos o partidas,
        dividido por igual entre los rivales, y decide si cumple el objetivo."""
        #Primero obtenemos los datos principales de como va nuestro agente
        fila = {"episodio": agente.episodios, "epsilon": round(agente.epsilon, 5),
                "recompensa_entreno_media": suma_r / max(n_r, 1),
                "estados_visitados": agente.estados_visitados(), "tiempo_s": round(time.time() - t0, 1)}
        #Seteamos las variables en 0
        valores, ganadas, decididas = [], 0, 0
        #Para cada rival en la pool de evaluación
        for j, rival in enumerate(pool_eval):
            #La semilla cambia en cada bloque: así el agente no puede "congelarse" justo porque
            #le tocó un conjunto de cartas favorable repetido (sesgo de selección)
            sem = _semilla(cfg, m, ROL_PARADA, j, agente.episodios // cfg.bloque)
            #Reiniciamos el rng y seteamos la vista del agente greedy para evaluación segun la semilla
            _reiniciar_rng(rival, sem)
            vista = agente.vista_greedy(np.random.default_rng(sem + 1))
            #Si el criterio de victoria es partida
            if cfg.criterio_victoria == "partida":
                #Dividimos la cantidad de partidas que jugaremos de forma equitativa
                n = max(cfg.partidas_eval // len(pool_eval), 1)
                #Evaluamos esa cantidad de partida contra cada rival
                res = evaluar_partidas(vista, rival, n, cfg.max_manos_partida, sem, **_env_kwargs(cfg))
                #Y obtenemos el desempeño
                v, ic = res["winrate_partidas"], res["winrate_partidas_ic95"]
                ganadas += round(v * n)
                decididas += n
            else:
                #Análogo pero evaluando por mano
                n = max(cfg.manos_eval // len(pool_eval), 1)
                res = evaluar_politica(vista, rival, n, semilla=sem, **_env_kwargs(cfg))
                v = valor_criterio(cfg, res["winrate"], None)
                ic = res["winrate_ic95"]
                ganadas += res["ganadas"]
                decididas += res["ganadas"] + res["perdidas"]
            fila[f"valor_vs_{j}"], fila[f"ic95_vs_{j}"] = v, ic
            valores.append(v)
        #Guardamos los datos en el diccionario y actualizamos la última actualización
        fila["valor_mezcla"] = ganadas / max(decididas, 1) if cfg.criterio_victoria != "racha" else sum(valores) / len(valores)
        estado["ultima"] = fila
        #Rellenamos el csv con la data obtenida
        _escribir_csv(ruta_csv, columnas, [fila])
        #Si superamos el objetivo entonces cumplimos
        cumple = (min(valores) >= cfg.objetivo) if cfg.criterio_parada == "cada_rival" else (fila["valor_mezcla"] >= cfg.objetivo)
        #Imprimmos el resultado y retornamos si cumple
        imprimir(f"   [Agente {m}] ep {agente.episodios:>8,} | mezcla {fila['valor_mezcla']:.3f} | "
                 + " ".join(f"{nombre_nodo(j)[:3]}{j}:{w:.2f}" for j, w in enumerate(valores)) + (" ✓" if cumple else ""))
        return cumple
    #Fijamos el tiempo inicial
    t0 = time.time()
    #Y los parametros iniciales
    suma_r, n_r, consecutivos, alcanzo = 0.0, 0, 0, False
    #Mientras queden episodios de entrenamiento
    while agente.episodios < cfg.max_episodios:
        #Tomamos el epsilon (el cual se va reduciendo a lo largo del entrenamiento gradualmente)
        agente.epsilon = max(cfg.eps_final, eps0 - (eps0 - cfg.eps_final) * agente.episodios / cfg.episodios_decaimiento)
        #Jugamos una mano del entrenamiento y actualizamos la Q-table
        r, _ = jugar_mano_entrenamiento(agente, env)
        suma_r += r
        n_r += 1
        agente.episodios += 1
        #Si llegamos a un checkpoint guardamos los resultados
        if agente.episodios % cfg.checkpoint_cada == 0:
            agente.guardar(os.path.join(dir_ent, f"checkpoint_ep{agente.episodios:09d}.npz"))
            agente.guardar(os.path.join(dir_ent, "ultimo.npz"))
        #Si llegamos a un bloque de evaluación verificamos si se cumplió la condición solicitada
        if agente.episodios % cfg.bloque == 0:
            #Checkeamos si cumple y reseteamos las recompensas acumuladas
            cumple = chequear(t0, suma_r, n_r)
            suma_r, n_r = 0.0, 0
            #Si consigue superar el umbral dos veces consecutivas superamos lo pedido
            consecutivos = consecutivos + 1 if (cumple and agente.episodios >= cfg.min_episodios) else 0
            if consecutivos >= cfg.confirmaciones:
                alcanzo = True
                break
    #Si llegamos al final pero no ha ocurrido un chequear, lanzamos uno forzado para guardar la ultima data
    if estado["ultima"] is None or estado["ultima"]["episodio"] != agente.episodios:
        chequear(t0, suma_r, n_r)
    #Guardamos la data final del entrenamiento y los datos principales del entrenamiento.
    ultima = estado["ultima"]
    wrs = [ultima[f"valor_vs_{j}"] for j in range(m)]
    agente.guardar(ruta_agente(carpeta, m), {"alcanzo_objetivo": alcanzo, "episodios": agente.episodios})
    return {"episodios": agente.episodios, "alcanzo_objetivo": alcanzo, "tiempo_s": round(time.time() - t0, 1),
            "winrate_mezcla_parada": ultima["valor_mezcla"], "winrate_min_parada": min(wrs),
            "hash_q": agente.hash_q()}


# ----------------------------------------------------------------------------
# enfrentamientos con número FIJO de manos
# ----------------------------------------------------------------------------

def enfrentar_nuevo_contra_previos(cfg: ConfigLiga, carpeta: str, m: int, manos_por_par: int | None = None,
                                   partidas_por_par: int | None = None, etiqueta: int = 0) -> list[dict]:
    """Agente m (greedy) contra cada nodo 0..m-1, con número FIJO de manos y de partidas:
    `*_por_par`, o `*_total` dividido por igual entre los m rivales. `etiqueta` cambia las semillas
    (0 = corrida original; otro valor = re-validación con cartas distintas)."""
    #Extraemos el número de manos/partidas que jugará el agente greedy contra los rivales congelados
    n_manos = manos_por_par or cfg.manos_por_par or max(cfg.manos_total // m, 1)
    n_part = partidas_por_par or cfg.partidas_por_par or max(cfg.partidas_total // m, 1)
    #Construimos la pool de rivales
    rivales, _ = construir_pool(cfg, carpeta, m, ROL_UNIFORME_EVAL + 100 * etiqueta, ROL_PAR_RIVALES + 100 * etiqueta)
    #Recuperamos al agente entreado
    nuevo = AgenteQ.cargar(ruta_agente(carpeta, m))
    #Fijamos la curiosidad en 0
    nuevo.epsilon = 0.0
    #Seteamos las filas 
    filas = []
    #Revisamos el desempeño contra cada rival (por mano y partida) de nuestro agente y guardamos los resultados
    for j, rival in enumerate(rivales):
        sem = _semilla(cfg, m, ROL_PAR + 100 * etiqueta, j)
        _reiniciar_rng(rival, sem)
        nuevo.rng = np.random.default_rng(sem + 1)
        res = evaluar_politica(nuevo, rival, n_manos, semilla=sem, **_env_kwargs(cfg))
        _reiniciar_rng(rival, sem + 2)
        nuevo.rng = np.random.default_rng(sem + 3)
        par = evaluar_partidas(nuevo, rival, n_part, cfg.max_manos_partida, sem + 4, **_env_kwargs(cfg))
        victorias = round(par["winrate_partidas"] * n_part)
        #Appendeamos los resultados
        filas.append({"iteracion": m, "a": m, "b": j, "nombre_a": nombre_nodo(m), "nombre_b": nombre_nodo(j),
                      "manos": n_manos, "ganadas_a": res["ganadas"], "ganadas_b": res["perdidas"],
                      "empates": res["empates"], "winrate_a": res["winrate"], "winrate_ic95": res["winrate_ic95"],
                      "bb_por_mano_a": res["bb_por_mano"], "bb_ic95": res["bb_ic95"],
                      "partidas": n_part, "partidas_ganadas_a": victorias, "partidas_ganadas_b": n_part - victorias,
                      "winrate_partida_a": par["winrate_partidas"], "winrate_partida_ic95": par["winrate_partidas_ic95"],
                      "manos_por_partida": par["manos_promedio"]})
    return filas


# ----------------------------------------------------------------------------
# análisis de cadenas por iteración
# ----------------------------------------------------------------------------

#Extraemos la matriz que usaremos y el subcriterio que usa esa matriz
def matriz_para_criterio(cfg: ConfigLiga, G: np.ndarray, Gp: np.ndarray):
    """(matriz de conteos, criterio para matriz_probabilidad) según criterio_victoria."""
    if cfg.criterio_victoria == "partida":
        return Gp, "mano"          # P = proporción de partidas ganadas
    return G, ("racha" if cfg.criterio_victoria == "racha" else "mano")


def analizar_cadenas(cfg: ConfigLiga, G: np.ndarray, Gp: np.ndarray, iteracion: int) -> list[dict]:
    """Cadena estricta más larga para cada umbral y modo (puntual / conservador)."""
    filas = []
    n = iteracion + 1
    #Extraemos la matriz para evaluar
    base, crit = matriz_para_criterio(cfg, G[:n, :n], Gp[:n, :n])
    for conservador in (False, True):
        #Generamos las probabilidad primero de forma usual y luego de forma conservadora usando el IC-I de Wilson
        P = matriz_probabilidad(base, crit, cfg.k_racha, conservador)
        #Para cada umbral calculamos 
        for umbral in cfg.umbrales:
            #Las matrices de adyacencias de dominancia
            ady = adyacencia(P, umbral)
            #Buscamos la cadena estricta más larga
            cadena = cadena_estricta_mas_larga(ady)
            #Revisamos si cumple transitividad
            tr = transitividad(ady)
            #Empaquetamos las filas
            filas.append({"iteracion": iteracion, "criterio": cfg.criterio_victoria, "k": cfg.k_racha,
                          "conservador": int(conservador), "umbral": umbral,
                          "largo_agentes": len([x for x in cadena if x != 0]), "largo_nodos": len(cadena),
                          "cadena": " > ".join(nombre_nodo(x) for x in cadena), **tr})
    return filas

#Buscamos la fila que cumple las condiciones que pedimos
def _fila_cadena(filas: list[dict], conservador: int, umbral: float) -> dict:
    return next(f for f in filas if f["conservador"] == conservador and abs(f["umbral"] - umbral) < 1e-9)


# ----------------------------------------------------------------------------
# liga completa
# ----------------------------------------------------------------------------
#Funcion que entrega las versiones de las herramientas utiizadas en el código
def _versiones() -> dict:
    v = {"python": sys.version.split()[0], "numpy": np.__version__, "plataforma": platform.platform()}
    try:
        import gymnasium
        v["gymnasium"] = gymnasium.__version__
    except Exception:
        pass
    return v


def correr_liga(cfg: ConfigLiga, carpeta: str, reanudar: bool = False, imprimir=print) -> dict:
    #Creamos el directorio a la carpeta
    os.makedirs(carpeta, exist_ok=True)
    #Que es el manifest.json
    ruta_manifest = os.path.join(carpeta, "manifest.json")
    #Si la ruta existe
    if os.path.exists(ruta_manifest):
        #Y no hay que reanudar
        if not reanudar:
            #Entonces ya había una ligada en la carpeta
            raise FileExistsError(f"{carpeta} ya contiene una liga. Usa --reanudar para continuarla o elige otra carpeta.")
        #Cargamos el json
        with open(ruta_manifest, encoding="utf-8") as f:
            manifest = json.load(f)
        #Extraemos los parametros para configurar la liga como la previa
        previa = ConfigLiga.from_dict(manifest["config"])
        #Revisamos si hubo cambios de parametros y guardamos las tuplas en formato valor viejo valor nuevo en caso de existir diferencias
        cambios = {k: (v, getattr(cfg, k)) for k, v in previa.to_dict().items()
                   if k not in ("n_agentes",) and v != cfg.to_dict()[k]}
        #Si existen cambios entonces tiramos error y devolvemos los cambios
        if cambios:
            raise ValueError(f"La configuración difiere de la liga original (rompería la reproducibilidad): {cambios}")
        #LLevamos la configuración a un diccionario
        manifest["config"] = cfg.to_dict()
    else:
        #Si no existe entonces inicializamos una liga nueva
        manifest = {"config": cfg.to_dict(), "versiones": _versiones(), "agentes": {}}
    #Revisamos cuantos agentes si han entrenado y guardado
    hechos = max([int(k) for k in manifest["agentes"]] + [0])
    #Agregamos el número de agentes + el uniforme
    n_nodos = cfg.n_agentes + 1
    #Cargamos las matrices de enfrentamientos
    G, Gp, E, _ = cargar_enfrentamientos(carpeta, n_nodos)

    #Retomamos el entrenamiento desde el siguiente al último hecho
    for m in range(hechos + 1, cfg.n_agentes + 1):
        #Imprimimos contra quien se está enfrentando
        imprimir(f"\n=== Agente {m}: entrena contra {{{', '.join(nombre_nodo(i) for i in range(m))}}} "
                 f"({'desde cero' if m == 1 else 'partiendo de los pesos del Agente ' + str(m - 1)}) ===")
        #Lo entrenamos
        info = entrenar_agente_liga(cfg, carpeta, m, imprimir)
        #Verificamos si alcanzó o no el objetivo del umbral
        estado = "OBJETIVO ALCANZADO" if info["alcanzo_objetivo"] else "NO alcanzó el objetivo (se congela igual)"
        imprimir(f"   -> {estado} en {info['episodios']:,} manos ({info['tiempo_s']} s)")

        #Venis
        filas_enf = enfrentar_nuevo_contra_previos(cfg, carpeta, m)
        _escribir_csv(os.path.join(carpeta, "enfrentamientos.csv"), COL_ENFRENTAMIENTOS, filas_enf)
        for f in filas_enf:
            G[m, f["b"]], G[f["b"], m] = f["ganadas_a"], f["ganadas_b"]
            Gp[m, f["b"]], Gp[f["b"], m] = f["partidas_ganadas_a"], f["partidas_ganadas_b"]
            E[m, f["b"]] = E[f["b"], m] = f["empates"]
        imprimir(f"   Resultados al congelar del Agente {m} ({cfg.criterio_victoria}): "
                 + " | ".join(f"vs {f['nombre_b']}: {_valor_criterio_fila(cfg, f):.2f} (manos {f['winrate_a']:.2f}, "
                              f"partidas {f['winrate_partida_a']:.2f}, {f['bb_por_mano_a']:+.1f} bb/mano)" for f in filas_enf))
        #Obtenemos la data que nos da la función que busca la cadenas estrictas
        filas_cad = analizar_cadenas(cfg, G, Gp, m)
        _escribir_csv(os.path.join(carpeta, "cadenas.csv"), COL_CADENAS, filas_cad)
        #Extraemos la fila de la cadena que cumple lo que queremos para el caso conservativo y principal
        principal = _fila_cadena(filas_cad, 0, cfg.umbral_cadena)
        conserv = _fila_cadena(filas_cad, 1, cfg.umbral_cadena)
        #Verifica si alguna cadena supero largo 1 e imprimimos resultados
        def _txt(f: dict) -> str:
            return f["cadena"] if f["largo_nodos"] > 1 else "ninguna (ningún par supera el umbral)"
        imprimir(f"   Cadena estricta más larga (>= {cfg.umbral_cadena}, criterio {cfg.criterio_victoria}): "
                 f"{_txt(principal)}  [{principal['largo_nodos']} nodos]"
                 f"  | conservador: {_txt(conserv)} [{conserv['largo_nodos']} nodos]")

        fila_prev = next((f for f in filas_enf if f["b"] == m - 1), None) if m > 1 else None
        fila_uni = next(f for f in filas_enf if f["b"] == 0)
        #Valor del criterio (racha / mano / partida) del agente m frente al anterior y frente al uniforme
        wr_prev = _valor_criterio_fila(cfg, fila_prev) if fila_prev else float("nan")
        wr_uni = _valor_criterio_fila(cfg, fila_uni)
        #Y rellenamos el csv con los datos importantes
        _escribir_csv(os.path.join(carpeta, "iteraciones.csv"), COL_ITERACIONES, [{
            "iteracion": m, "episodios": info["episodios"], "alcanzo_objetivo": int(info["alcanzo_objetivo"]),
            "tiempo_s": info["tiempo_s"], "winrate_mezcla_parada": info["winrate_mezcla_parada"],
            "winrate_min_parada": info["winrate_min_parada"], "winrate_vs_anterior": wr_prev,
            "winrate_vs_uniforme": wr_uni,
            "bb_vs_anterior": fila_prev["bb_por_mano_a"] if fila_prev else float("nan"), "largo_cadena": principal["largo_agentes"],
            "largo_cadena_conservador": conserv["largo_agentes"], "frac_transitiva": principal["frac_transitiva"],
            "hash_q": info["hash_q"]}])
        #Actualizamos el manifest
        manifest["agentes"][str(m)] = {
            "archivo": os.path.relpath(ruta_agente(carpeta, m), carpeta).replace("\\", "/"),
            "hash_q_sha256": info["hash_q"], "episodios": info["episodios"],
            "alcanzo_objetivo": info["alcanzo_objetivo"],
            "semillas": {"agente": _semilla(cfg, m, ROL_AGENTE), "entorno": _semilla(cfg, m, ROL_ENV)}}
        manifest["config"] = cfg.to_dict()
        with open(ruta_manifest, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        #Freno de seguridad
        if not info["alcanzo_objetivo"] and cfg.detener_si_no_alcanza:
            imprimir("   La liga se detiene: el agente no alcanzó el objetivo (--detener-si-no-alcanza).")
            break
    #Guardamos las matrices y luego retornamos el manifest
    _guardar_matrices(carpeta, G, Gp, cfg, manifest)
    return manifest


def _guardar_matrices(carpeta: str, G: np.ndarray, Gp: np.ndarray, cfg: ConfigLiga, manifest: dict) -> None:
    """Matrices finales (fila = agente, columna = rival): win rate por mano, por partida y
    probabilidad de la carrera a k manos seguidas."""
    n = max([int(k) for k in manifest["agentes"]] + [0]) + 1
    nombres = [nombre_nodo(i) for i in range(n)]
    matrices = {"matriz_mano.csv": matriz_probabilidad(G[:n, :n], "mano"),
                "matriz_partida.csv": matriz_probabilidad(Gp[:n, :n], "mano"),
                "matriz_racha.csv": matriz_probabilidad(G[:n, :n], "racha", cfg.k_racha)}
    for nombre, M in matrices.items():
        with open(os.path.join(carpeta, nombre), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([""] + nombres)
            for et, fila in zip(nombres, M):
                w.writerow([et] + [f"{v:.4f}" for v in fila])


def reentrenar_y_comparar(carpeta: str, m: int, imprimir=print) -> tuple[str, str, bool]:
    """Reentrena el agente m en una carpeta temporal (usando los agentes 1..m-1 guardados) y
    compara el hash de su tabla Q con el del manifest. -> (hash_nuevo, hash_original, igual)."""
    with open(os.path.join(carpeta, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    cfg = ConfigLiga.from_dict(manifest["config"])
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "agentes"))
        for i in range(1, m):
            shutil.copy(ruta_agente(carpeta, i), ruta_agente(tmp, i))
        info = entrenar_agente_liga(cfg, tmp, m, imprimir)
    original = manifest["agentes"][str(m)]["hash_q_sha256"]
    return info["hash_q"], original, info["hash_q"] == original