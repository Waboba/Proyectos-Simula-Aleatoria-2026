"""
Análisis de cadenas estrictas de agentes.

Datos de entrada: una matriz G de enteros, G[i][j] = manos que el agente i le
ganó al agente j (los empates no cuentan). Con eso:

  p_ij = G[i][j] / (G[i][j] + G[j][i])       win rate por mano de i sobre j

  criterio "mano" : "i le gana a j" con probabilidad p_ij
  criterio "racha": con probabilidad P_ij = prob_racha(p_ij, k), la de ganar la
                    carrera "primero en ganar k manos seguidas" (k=3 por defecto)

Una CADENA ESTRICTA al nivel `umbral` (p. ej. 0.9) es una sucesión a1, a2, ..., am
en la que CADA agente le gana a TODOS los que vienen después con probabilidad
>= umbral (no solo al siguiente): a1>a2, a2>a3 y también a1>a3, etc. Es decir,
la transitividad se exige, no se supone.

Modo conservador: en vez de la estimación puntual se usa la cota INFERIOR del
intervalo de confianza de Wilson al 95% de p_ij. Una cadena que sobrevive en
modo conservador es difícil de atribuir al azar de las simulaciones.
"""
import numpy as np

from .evaluacion import prob_racha

#Creamos un intervalo de wilson, notar que por manos y por partidas funciona de la misma manera
#pues son valores binarios de ganar o no lo que genera un comportamiento similar.
#Por qué Wilson? Nos salva de condiciones del tipo 100% winrate cuando estas pueden ser por generadas por azar
#Si en liga tomamos modo conservador usamos el intervalo inferior de Wilson 
#que nos da una cota de confianza al posible intervalo de real de victoria para las cotas de dominancia
def wilson_ic(exitos: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalo de Wilson para una proporción."""
    #El código es la fórmula de wilson
    if n <= 0:
        return 0.0, 1.0
    p = exitos / n
    d = 1 + z * z / n
    centro = p + z * z / (2 * n)
    margen = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float((centro - margen) / d), float((centro + margen) / d)


def matriz_probabilidad(G, criterio: str = "racha", k: int = 3, conservador: bool = False) -> np.ndarray:
    """P[i, j] = probabilidad (según el criterio) de que i le gane a j. Pares sin datos: 0.5."""
    n = len(G)
    #Creamos una matriz cuadrada de nxn donde por defecto el porcentaje de ganar es de 50% para todos
    P = np.full((n, n), 0.5)
    #Recorremos todos los agentes
    for i in range(n):
        for j in range(n):
            #Si son iguales pasamos
            if i == j:
                continue
            #Si son distintos, tomamos las partidas totales jugadas entre ellos
            decididas = int(G[i][j] + G[j][i])
            #Si es 0 continuamos
            if decididas == 0:
                continue
            #Si no obtenemos el intervalo de confiaza de Wilson y calculamos la probabilidad de que gane dependiendo en que modo estamos
            p = wilson_ic(int(G[i][j]), decididas)[0] if conservador else G[i][j] / decididas
            #Finalmente, rellenamos la matriz de probabilidad
            P[i, j] = prob_racha(p, k) if criterio == "racha" else p
    return P

#Transformamos la matriz de probabilidades en una matriz de adyacencia
def adyacencia(P: np.ndarray, umbral: float) -> list[set[int]]:
    #Extraemos el tamaño
    n = len(P)
    #Se obtiene un lista donde cada elemento contiene un conjunto con los indices que el agente i domina
    return [{j for j in range(n) if j != i and P[i, j] >= umbral} for i in range(n)]


def cadena_estricta_mas_larga(ady: list[set[int]]) -> list[int]:
    """Cadena más larga en la que cada elemento vence a todos los posteriores.
    Búsqueda exhaustiva con poda (rápida para decenas de agentes)."""
    #Inicializamos una lista para buscar el mejor
    mejor: list[int] = []
    #Definimos dfs que busca el mejor candidato por poda
    def dfs(cadena: list[int], candidatos: set[int]) -> None:
        nonlocal mejor
        #Si el largo de nuestra cadena supera al que teniamos guardado como mejor lo reemplazamos
        if len(cadena) > len(mejor):
            mejor = cadena[:]
        #Aquí podamos si es que vemos que no podemos superar al largo del mejor candidato
        if len(cadena) + len(candidatos) <= len(mejor):
            return
        #Luego, ordenamos los candidatos por orden decreciente del tamaño de dominancia
        for j in sorted(candidatos, key=lambda x: -len(ady[x])):
            #Agregamos a j a la cadena
            cadena.append(j)
            #Luego aplicamos DFS en la cadena con nuestros candidatos y tal que el nuevo candidato sea dominado por j
            dfs(cadena, candidatos & ady[j])
            cadena.pop()
    #Tomamos a cada agente 
    for i in range(len(ady)):
        #Exploramos sus candidatos via DFS, para ello le entregamos su diccionario de dominados. y aplicamos la función
        dfs([i], set(ady[i]))
    return mejor


def transitividad(ady: list[set[int]]) -> dict:
    """Entre los caminos a->b->c (a != c), cuántos tienen también a->c."""
    #Buscamos evaluar los caminos y ciclos
    caminos = cerrados = 0
    #Para cada a 
    for a in range(len(ady)):
        #Recorremos los que son dominados por a
        for b in ady[a]:
            #Y con ello recorremos los que son dominados por b
            for c in ady[b]:
                #Si no a
                if c != a:
                    #Entonces hay un camino que nos lleva de a hacia c
                    caminos += 1
                    #Si c es dominado por a entonces cerramos el camino
                    cerrados += c in ady[a]
    #Retornamos los valores útiles
    return {"caminos2": caminos, "caminos2_cerrados": cerrados,
            "frac_transitiva": (cerrados / caminos) if caminos else float("nan")}


def verificar_cadena(G, cadena: list[int], criterio: str, k: int, umbral: float) -> list[dict]:
    """Detalle de TODOS los pares (i antes que j) de la cadena: p, IC95 y si cumplen el umbral."""
    filas = []
    #Iteramos sobre la cadena 
    for a in range(len(cadena)):
        for b in range(a + 1, len(cadena)):
            #Definimos los pares a evaluar
            i, j = cadena[a], cadena[b]
            #Revisamos las partidas jugadas
            decididas = int(G[i][j] + G[j][i])
            #Calculamos la probabildiad
            p = G[i][j] / decididas if decididas else float("nan")
            #Y sus intervalos de confianza
            lo, hi = wilson_ic(int(G[i][j]), decididas)
            #Definimos una función auxiliar que en nuestro casa devuelve la identidad
            f = (lambda x: prob_racha(x, k)) if criterio == "racha" else (lambda x: x)
            #Agregamos la data
            filas.append({"i": i, "j": j, "manos": decididas, "p_mano": p, "p_ic_inf": lo, "p_ic_sup": hi,
                          "prob": f(p), "prob_ic_inf": f(lo), "cumple_puntual": f(p) >= umbral,
                          "cumple_conservador": f(lo) >= umbral})
    return filas