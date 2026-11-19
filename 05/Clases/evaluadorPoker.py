"""
Evaluador de manos de póker (5 a 7 cartas) sobre cartas-entero (0-51).

`evaluar(cartas)` devuelve UN solo entero comparable: mayor entero = mejor mano.

    rango = categoria << 20 | k1 << 16 | k2 << 12 | k3 << 8 | k4 << 4 | k5

donde k1..k5 son los valores de desempate (4 bits cada uno, 2..14), en orden
de importancia. Así, determinar el ganador es simplemente comparar enteros
(max / argmax / np.argmax sobre un arreglo de rangos), y el entero ya trae
codificado el desempate.

No se prueban las 21 combinaciones de 5 cartas: se cuentan valores y pintas
una sola vez y se decide la categoría directamente.
"""

from .carta import Carta
#Establecemos la jerarquía de las manos de póker, donde cada mano tiene un valor numérico.
JERARQUIA = {
    "Carta Alta": 1, "Par": 2, "Doble Par": 3, "Trío": 4, "Escalera": 5,
    "Color": 6, "Full House": 7, "Póker": 8, "Escalera de Color": 9,
}
NOMBRE_CATEGORIA = {valor: nombre for nombre, valor in JERARQUIA.items()}

#Genera un numero binario que codificará las manos, para ello, como los números caben en 4 bits hacemos lo siguiente:
#Primero, la categoría de la mano, luego las cartas que desempatan y con OR quitamos los 0 entremedio 
#obteniendo un entero grande, pero comparable que representa el valor de mi mano.
def _empaquetar(cat: int, a: int = 0, b: int = 0, c: int = 0, d: int = 0, e: int = 0) -> int:
    return (cat << 20) | (a << 16) | (b << 12) | (c << 8) | (d << 4) | e

#Encuentra la carta más alta de una escalera
def _alta_escalera(mascara: int) -> int:
    """Carta alta de la mejor escalera contenida en la máscara de valores
    (bit v encendido = hay un valor v). 0 si no hay escalera. El As también
    cuenta como 1 (rueda A-2-3-4-5, que vale '5 alta')."""
    #Comparamos nuestra mascara (en este caso tiene bits prendidos en las cartas que tenemos) con la máscara con un 1 en la posición 14
    #Si nuestra mascara original tenía ese bit prendido (ie tenía el 14) entonces prendemos el bit 2 para que detectemos la escalera A-2-3-4-5.
    if mascara & (1 << 14):
        #Prendemos el bit para detectar la escalera
        mascara |= 2
    #Recorremos las cartas de mayor a menor.
    for alta in range(14, 4, -1):
        #Desplazamos la máscara a la derecha las posiciones de la escalera alta que estemos evaluando
        #Luego, quedaran las 5 candidatos a escalare (y sus bits) en las primeras 5 posiciones de la máscara, y si es una escalera, entonces los 5 bits estarán prendidos (1).
        #Y por ende al filtrar con 31 (11111) veremos si en efecto había una escalera.
        if ((mascara >> (alta - 4)) & 31) == 31:
            return alta
    return 0

#Encuentra los valores de desempate de una mano, excluyendo los valores que ya se usaron en la categoría de la mano.
#Aquí conteo representa una lista con la cantidad de cartas de cada valor que tenemos,
#excluidos representa los valores que ya se usaron en la categoría de la mano (Por ejemplo, si tenemos un par de 10, entonces 10 ya se usó y no lo consideramos para desempate),
# y n representa la cantidad de valores distintos más altos que queremos encontrar. (Por ejemplo, si tenemos un par de 10 queremos 3 valores distintos más altos para desempate).
def _mejores(conteo: list, excluidos: tuple, n: int) -> list:
    """Los n valores distintos más altos presentes, sin los excluidos
    (se rellena con 0 si faltan)."""
    #Generamos una lista para guardar los valores de desempate que encontremos.
    res = []
    #Recorremos los valores de mayor a menor
    for v in range(14, 1, -1):
        #Si tenemos cartas de ese valor y no está en los excluidos, lo agregamos a la lista de resultados.
        if conteo[v] and v not in excluidos:
            res.append(v)
            #Si ya encontramos la cantidad de valores que queríamos, salimos del bucle.
            if len(res) == n:
                break
    #Si no encontramos suficientes valores, rellenamos con 0 hasta tener n valores.
    res.extend([0] * (n - len(res)))
    return res

#Recorre la lista de cartas para evaluar cual es la mejor mano posible y devuelve un entero que representa esa mano. (considerando mano las cartas del jugador y mesa las cartas comunitarias)
def evaluar(cartas) -> int:
    """Rango entero de la mejor mano de 5 que se puede armar con 5-7 cartas."""
    #Conteo representa una lista de tamaño 15 (2 extra por si acaso) y cada posición desde el 2 hasta al final representan la cantidad de cartas de ese valor.
    conteo = [0] * 15
    #Almacena bits binarios que representan las cartas de cada pinta que tenemos. 
    #(por ejemplo, si tenemos un 2 de corazones y un 3 de corazones, entonces por_pinta[0] tendrá los bits 2 y 3 prendidos)
    por_pinta = [0, 0, 0, 0]
    #Cuenta cuantas cartas tenemos por cada pinta
    n_pinta = [0, 0, 0, 0]
    #A diferencia de por pinta aquí solo nos interesa su valor (sirve para escaleras)
    mascara = 0
    #Ahora recorremos todas las cartas que tenemos y las vamos clasificando en las estructuras que creamos arriba.
    for c in cartas:
        #Usamos las fórmulas de antes para obtener el valor y la pinta de la carta.
        v = (c >> 2) + 2
        p = c & 3
        #Incrementamos el conteo en la posición de valor
        conteo[v] += 1
        #Entramos a la posición p de por_pinta, prendemos el bit correspondiente al valor de la carta y aumentamos el contador de cartas de esa pinta.
        por_pinta[p] |= 1 << v
        n_pinta[p] += 1
        #Finalmente, prendemos el bit correspondiente al valor de la carta en la máscara general.
        mascara |= 1 << v

    #En este fragmento analizamos la presencia de color y escalera de color.
    pinta_color = -1 #Fijamos un valor imposible
    #Recorremos las pintas
    for p in range(4):
        #Si tenemos 5 o más cartas de esa pinta, entonces tenemos color.
        if n_pinta[p] >= 5:
            #Fijamos la variable pinta_color para indicar que tenemos color y de que pinta es.
            pinta_color = p
            #Obtenos la carta más alta de la escalera de color (si es que hay)
            alta = _alta_escalera(por_pinta[p])
            #Si es que había una escalera de color, entonces empaquetamos la mano como escalera de color y devolvemos el entero correspondiente.
            if alta:
                return _empaquetar(9, alta)
    #Ahora revisamos trios y pares para ver si tenemos full house o póker. También los usamos para desempate en otras manos.
    trios = []
    pares = []
    #Recorremos los valores de mayor a menor,
    for v in range(14, 1, -1):
        #Verificamos cuantas cartas tenemos de ese valor
        k = conteo[v]
        #Si tenemos 4, entonces tenemos póker y devolvemos el entero correspondiente.
        if k == 4:
            #Empaquetamos la mano como poker y usamos mejores para encontrar el kicker más alto que no sea el valor del póker.
            return _empaquetar(8, v, _mejores(conteo, (v,), 1)[0])
        #Si tenemos 3, entonces agregamos ese valor a la lista de trios.
        if k == 3:
            trios.append(v)
        #Si tenemos 2, entonces agregamos ese valor a la lista de pares.
        elif k == 2:
            pares.append(v)
    #Verificamos si tenemos full house (ie tenemos al menos un trío y al menos un par, o al menos dos tríos)
    if trios and (len(trios) > 1 or pares):
        #Luego, si tenemos full house, construimos la mejor opción posible de par con el resto de cartas
        segundo = max(trios[1] if len(trios) > 1 else 0, pares[0] if pares else 0)
        #Retornamos el empaquetado devolviendo el full house y los correspondientes desempates.
        return _empaquetar(7, trios[0], segundo)
    #Verificamos si tenemos color
    if pinta_color >= 0:
        #Recuperamos la máscara con los bits prendidos del color que  teníamos arriba
        m = por_pinta[pinta_color]
        #Generamos una lista vacia para recuperar nuestro color 
        top = []
        #Recorremos los valores de mayor a menor.
        for v in range(14, 1, -1):
            #Si al desplazar la máscara (ie llevar el bit correspondiente a la primera posición) y filtrarlo con 1 nos da 1 entonces lo teníamos y se agrega
            if (m >> v) & 1:
                top.append(v)
                #Si ya tenemos 5 cartas, salimos del bucle.
                if len(top) == 5:
                    break
        #Empaquetamos la mano como color y le entregamos los valores desempaquetados para desempate.
        return _empaquetar(6, *top)
    #Verificamos si tenemos escalera
    alta = _alta_escalera(mascara)
    if alta:
        #Si tenemos escalera, empaquetamos la mano como escalera y le entregamos la carta más alta de la escalera para desempate.
        return _empaquetar(5, alta)
    #Verificamos si tenemos trío,
    if trios:
        #Extraemos los kickers de desempate
        a, b = _mejores(conteo, (trios[0],), 2)
        #Empaquetamos la mano como trío y le entregamos los valores desempaquetados para desempate.
        return _empaquetar(4, trios[0], a, b)
    #Verificamos si tenemos doble par
    if len(pares) >= 2:
        #Extraemos la carta de desempate extra al doble par
        k = _mejores(conteo, (pares[0], pares[1]), 1)[0]
        #Empaquetamos la mano como doble par y le entregamos los valores desempaquetados para desempate.
        return _empaquetar(3, pares[0], pares[1], k)
    #Verificamos si tenemos un par
    if pares:
        #Extraemos los kickers de desempate del par
        a, b, c = _mejores(conteo, (pares[0],), 3)
        #Empaquetamos la mano como par y le entregamos los valores desempaquetados para desempate.
        return _empaquetar(2, pares[0], a, b, c)
    #Si no tenemos ninguna de las manos anteriores, entonces tenemos carta alta y le entregamos los valores desempaquetados para desempate.
    return _empaquetar(1, *_mejores(conteo, (), 5))


def categoria(rango: int) -> int:
    #Operación inversa a lo de arriba para obtener el número que representa la categoría de la mano (1-9)
    return rango >> 20


def decodificar(rango: int) -> tuple:
    """(categoria, (desempates...)) -- p. ej. la rueda da (5, (5,))."""
    #Extraemos los kickers de desempate de la mano, que están codificados en los bits del rango.
    kickers = tuple(
        k for k in ((rango >> 16) & 15, (rango >> 12) & 15, (rango >> 8) & 15,
                    (rango >> 4) & 15, rango & 15) if k
    )
    #Devolvemos la categoría de la mano y los kickers de desempate.
    return rango >> 20, kickers

#Finalmente definimos el evaluador que usa todo lo anterior.
class EvaluadorPoker:
    """Envoltorio con la interfaz anterior. Acepta cartas-entero o Carta.
    obtener_rango_completo() ahora devuelve un int (antes una tupla)."""

    JERARQUIA = JERARQUIA

    def __init__(self, cartas_disponibles: list) -> None:
        #Obtenemos las cartas en formato entero (0-51) y evaluamos la mejor mano posible con esas cartas.
        self.cartas = [c.to_int() if isinstance(c, Carta) else int(c) for c in cartas_disponibles]
        self._rango = evaluar(self.cartas)

    def obtener_rango_completo(self) -> int:
        return self._rango

    def obtener_jugada(self) -> str:
        return NOMBRE_CATEGORIA[self._rango >> 20]

    @staticmethod
    def evaluar(cartas) -> int:
        return evaluar(cartas)

    @staticmethod
    def decodificar(rango: int) -> tuple:
        return decodificar(rango)