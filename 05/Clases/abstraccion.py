"""
Abstracción de estados mejorada para Q-learning TABULAR.
(Se podría usar otro método que no fuera Chen, como equity montecarlo, que es buenísimo pero demasiado caro computacionalmente)
Se eligió Chen, que es una heuristica, pero simplifica la evaluación y ha sido usada bastante en estudios del poker.
Dimensiones del estado (N_ESTADOS = 12,096):
    fase       (4)  : 0=preflop, 1=flop, 2=turn, 3=river
    fuerza     (14) :
                      - preflop: 12 grupos según puntuación de Chen (0-11)
                      - postflop:
                          0: carta alta baja (< J)
                          1: carta alta alta (>= J)
                          2: par solo de la mesa
                          3: par medio/bajo mío
                          4: top pair con kicker débil (< J)
                          5: top pair con buen kicker (>= J) u overpair
                          6: doble par
                          7: trío con par en mesa (trips)
                          8: trío oculto con par en mano (set)
                          9: escalera
                         10: color
                         11: full house
                         12: póker
                         13: escalera de color
    proyecto   (3)  : 0=ninguno, 1=proyecto simple (color o escalera),
                      2=combo draw (color+escalera) en flop/turn O proyecto fallido en river
    presion    (4)  : cuánto debo igualar respecto al pozo previo
                      (0=nada, 1=<=40%, 2=<=100%, 3=>100%)
    posicion   (2)  : 1 si soy dealer (ciega chica), 0 si no
    spr        (3)  : fichas efectivas / pozo (0=<1.5 comprometido, 1=<4.0 medio, 2=>=4.0 profundo)
    iniciativa (3)  : dinámica de agresión en la mano
                      (0=bote sin subidas relevantes, 1=bote subido sin apuesta rival, 2=rival apostando)
"""
from bisect import bisect_right
from math import ceil

from .evaluadorPoker import evaluar
#Definimos los números que generan nuestro número de estados
N_FASES = 4
N_FUERZA = 14
N_PROYECTO = 3
N_PRESION = 4
N_POSICION = 2
N_SPR = 3
N_INICIATIVA = 3
N_ESTADOS = N_FASES * N_FUERZA * N_PROYECTO * N_PRESION * N_POSICION * N_SPR * N_INICIATIVA

# 12 cubetas preflop (0..11) para separar mejor primas, suited connectors y basura
#Definimos las casillas para empaquetar el poder de las cartas
_CORTES_PREFLOP = (2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15)
#Definimos mediante un mapeo los puntos que nosotros asignamos a las cartas a los puntos de Chen
_PUNTOS_ALTA = {14: 10, 13: 8, 12: 7, 11: 6}
#Penalización por huecos entre cartas que dificultan escalera
_PENAL_HUECO = (0, 1, 2, 4, 5)

#Función que calcula la fuerza de la mano inicial
def fuerza_preflop(c1: int, c2: int) -> int:
    """Grupo 0-11 según la puntuación de Chen (pares, suited, conectadas)."""
    #Usamos nuestras fórmula de representación para obtener los valores de las cartas
    v1, v2 = (c1 >> 2) + 2, (c2 >> 2) + 2
    #Verificamos cual es la más alta
    alta, baja = (v1, v2) if v1 >= v2 else (v2, v1)
    #Obtenemos cuantos puntos vale la carta alta (si no está en los puntos alta la dividimos a la mitad)
    puntos = _PUNTOS_ALTA.get(alta, alta / 2)
    #Si tenemos un par, se multiplica el puntaje de la alta por 2 o bien como mínimo valdrá 5 (par de 2 tendría 4 de puntaje)
    if v1 == v2:
        puntos = max(puntos * 2, 5)
    else:
        #Si son del mismo palo, entonces sumamos 2 puntos
        if (c1 & 3) == (c2 & 3):
            puntos += 2
        #Calculamos el espacio entre nuestras cartas
        hueco = alta - baja - 1
        #Penalizamos dependiendo del tamaño del hueco
        puntos -= _PENAL_HUECO[min(hueco, 4)]
        #Condición especial para sumar una carta
        if hueco <= 1 and alta < 12:
            puntos += 1
    #Finalmente, encasillamos el poder nuestras cartas según el techo del puntaje calculado a partir de Chen
    #La función recorre los índices de cortes y se pregunta si tu puntaje es menor o igual, si lo eres te devuelve a ese indice,
    #Si no, pasa al siguiente. Si llega al final y eres mayor a 15 entonces quedas encasillado en 15.
    return bisect_right(_CORTES_PREFLOP, ceil(puntos))

#Funcion que devuelve el nivel del proyecto
def nivel_proyecto(mano, mesa, categoria: int) -> int:
    """Devuelve:
      0 -> sin proyecto
      1 -> proyecto de color O proyecto de escalera (4 cartas)
      2 -> combo draw (color + escalera) en flop/turn, O proyecto fallido en river (carta alta)
    """
    #Si ya tenemos una mano armada (Escalera para arriba en puntaje) entonces no buscamos proyectos
    if categoria >= 5:
        return 0

    #Aquí revisamos si tenemos proyecto de color
    pintas = [0, 0, 0, 0]
    mascara = 0
    #Contamos las pintas de las cartas de la mesa y las almacenamos en la lista.
    for c in mano + mesa:
        pintas[c & 3] += 1
        #Se crea una máscara que prende los bits de los valores presentes en la partida para proyecto escalera
        mascara |= 1 << ((c >> 2) + 2)
    #Revisamos si tenemos proyecto de color, para esto revisamos si alguna de las 4 cartas que forman el proyecto la tenemos nosotros
    proy_color = any(pintas[p] == 4 and any((c & 3) == p for c in mano) for p in range(4))

    #Ahora generamos una máscara para estudiar que valores tenemos en nuestra mano
    mascara_mano = 0
    for c in mano:
        mascara_mano |= 1 << ((c >> 2) + 2)
    #Luego, el mismo truco que hicimos en evaluador prendemos el 2 si hay un As presente para estudiar la posibilidad de Rueda
    if mascara & (1 << 14):
        mascara |= 2
    if mascara_mano & (1 << 14):
        mascara_mano |= 2

    proy_escalera = False
    #Revisamos las cartas que pueden iniciar una escalera
    for bajo in range(1, 11):
        #Desplazamos la máscara bajo posiciones, revisamos si hay 4 1s prendidos y si al menos alguno de esos nos pertenece y con ello concluir si hay proyecto.
        ventana = (mascara >> bajo) & 31
        if bin(ventana).count("1") >= 4 and ((mascara_mano >> bajo) & 31):
            proy_escalera = True
            break

    #En el river ya no se reparten más cartas, pero recordar que tuvimos proyecto fallido (y tenemos carta alta) ayuda al agente a elegir candidatos para farolear.
    if len(mesa) >= 5:
        return 2 if (categoria == 1 and (proy_color or proy_escalera)) else 0

    if proy_color and proy_escalera:
        return 2
    if proy_color or proy_escalera:
        return 1
    return 0

#Entrega la fuerza postflop de nuestra mano
def fuerza_postflop(mano, mesa) -> tuple[int, int]:
    """(grupo de fuerza 0-13, categoría de póker 1-9)."""
    #Usamos nuestra función para evaluar la fuerza de la mano
    rango = evaluar(mano + mesa)
    #Extraemos la categoría de la mano
    cat = rango >> 20
    #Obtenemos los valores de las cartas de nuestra mano
    n1, n2 = (mano[0] >> 2) + 2, (mano[1] >> 2) + 2
    #Definimos la mano alta y la mano baja
    alta_mano, baja_mano = (n1, n2) if n1 >= n2 else (n2, n1)
    #Ahora clasificamos el puntaje dependiendo de la categoría y lo descrito arriba
    #Carta alta
    if cat == 1: 
        #Retorna 1 si es que somos mayor a una J y 0 si no.
        return (1 if alta_mano >= 11 else 0), cat
    #Par
    if cat == 2:  
        #Extraemos el valor del par
        v = (rango >> 16) & 15
        #Si el valor del par no coincide con ninguna de las cartas de nuestra mano entonces es comunitario y es una mala mano
        if v != n1 and v != n2:
            return 2, cat  #
        #Obtenemos el valor más alto de la mesa
        mejor_mesa = max((c >> 2) + 2 for c in mesa)
        #Si el valor es mayor o igual al máximo valor de la mesa entonces o tenemos Top Pair u Overpair
        if v >= mejor_mesa:
            #Si mi par fue realizado con mi carta alta, mi kicker será mi carta baja, en caso contrario la carta alta.
            kicker = baja_mano if v == alta_mano else alta_mano
            #Si tenemos Overpait o nuestro kicker es fuerte (Top Pair) calificamos con 5 puntos, si no 4 (Top Pair kicker débil)
            return (5 if (n1 == n2 or kicker >= 11) else 4), cat
        #Si no le ganamos a mesa, entonces es un Par medio o bajo
        return 3, cat  
    #Doble Par
    if cat == 3:
        return 6, cat
    #Trío
    if cat == 4:  #Distinguir set oculto (par en mano) vs trips (par en mesa)
        return (8 if n1 == n2 else 7), cat
    #Y el resto se extrae directamente de un diccionario correspondiente a su mano.
    return {5: 9, 6: 10, 7: 11, 8: 12, 9: 13}.get(cat, 13), cat

#Ahora pasamos los datos crudos observados por el motor en un indice de mi Q-table
def estado_de_obs(obs, ciega_grande: float = 20.0) -> int:
    """Observación (motor.observacion) -> índice de estado 0..N_ESTADOS-1."""
    #Recuperamos los datos de la observación que nos da motor
    mano = [int(obs[0]), int(obs[1])]
    mesa = [int(x) for x in obs[2:7] if x >= 0]
    fase = min(int(obs[7]), N_FASES - 1)
    pozo = float(obs[8])
    a_igualar = float(obs[11])
    #Si estamos en el preflop, entonces calculamos nuestra fuerza con la función respectiva
    if fase == 0:
        fuerza = fuerza_preflop(mano[0], mano[1])
        proyecto = 0
    #En cualquier otra fase de la partida la calculamos con la fuerza postflop
    #Y obtenemos los posibles proyectos
    else:
        fuerza, cat = fuerza_postflop(mano, mesa)
        proyecto = nivel_proyecto(mano, mesa, cat)
    #Ahora se evalua la presión financiera (que tan caro sale igualar c/r al pozo central)
    if a_igualar <= 0:
        #Si no nos han apostado nada que igualar no hay presión
        presion = 0
    #Si al menos hay algo que igualar 
    else:
        #Obtenemos cuanto porcentualmente corresponde lo a igualar c/r al pozo
        fraccion = a_igualar / max(pozo - a_igualar, 1.0)
        #Dependiendo del valor de ese porcentaje es la presión que nos hacen
        presion = 1 if fraccion <= 0.4 else (2 if fraccion <= 1.0 else 3)
    #Ahora calculamos el SPR
    #Obtenemos la cantidad mínima de fichas de los jugadores, pues si la diferencia no sirve para calcular
    efectivo = min(float(obs[9]), float(obs[10]))
    #Obtenemos la razón de esas fichas con respecto al pozo
    razon = efectivo / max(pozo, 1.0)
    #Calculamos el spr 
    spr = 0 if razon < 1.5 else (1 if razon < 4.0 else 2)
    posicion = 1 if obs[12] >= 0.5 else 0

    #Dinámica de agresión:
    #Si hay apuesta que igualar -> rival agrediendo (2)
    if a_igualar > 0:
        iniciativa = 2
    #Si no hay apuesta que igualar, pero el pozo supera el bote mínimo limpeado (~2.5 BB) -> iniciativa propia / bote subido (1)
    elif pozo > 2.5 * ciega_grande:
        iniciativa = 1
    else:
        iniciativa = 0
    #Empaquetamos el resultado
    idx = fase
    idx = idx * N_FUERZA + fuerza
    idx = idx * N_PROYECTO + proyecto
    idx = idx * N_PRESION + presion
    idx = idx * N_POSICION + posicion
    idx = idx * N_SPR + spr
    idx = idx * N_INICIATIVA + iniciativa
    return idx

#Finalmente, una función para desempaquetar lo realizado por estado de obs
def decodificar_estado(estado: int) -> dict:
    """Inverso de estado_de_obs (útil para inspeccionar la tabla Q)."""
    estado, iniciativa = divmod(estado, N_INICIATIVA)
    estado, spr = divmod(estado, N_SPR)
    estado, posicion = divmod(estado, N_POSICION)
    estado, presion = divmod(estado, N_PRESION)
    estado, proyecto = divmod(estado, N_PROYECTO)
    fase, fuerza = divmod(estado, N_FUERZA)
    return {
        "fase": ("PREFLOP", "FLOP", "TURN", "RIVER")[fase],
        "fuerza": fuerza,
        "proyecto": proyecto,
        "presion": presion,
        "dealer": posicion,
        "spr": spr,
        "iniciativa": iniciativa,
    }