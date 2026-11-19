import numpy as np


class Mazo:
    """Mazo de 52 cartas-entero. En vez de crear/mutar listas de objetos,
    guarda una permutación (arreglo numpy) y un puntero `pos` a la próxima
    carta: entregar una carta es leer una posición, y reiniciar es barajar
    el mismo arreglo (sin crear nada nuevo). La aleatoriedad viene de un
    np.random.Generator inyectable (reproducible con semilla)."""
    #Fijamos el total de cartas
    TOTAL = 52
    #Y con ello creamos un arreglo de enteros de 0 a 51 que representa el mazo base
    _BASE = np.arange(52, dtype=np.int8)

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        #Generamos una semilla aleatoria (notar que es distinta para cada instancia de la clase)
        self.rng = rng if rng is not None else np.random.default_rng()
        #Creamos un arreglo de enteros de 0 a 51 que representa el mazo y un puntero a la posición actual del mazo (cuantas cartas se han entregado)
        self.orden = np.arange(self.TOTAL, dtype=np.int8)
        self.pos = 0

    #Devuelve la cantidad de cartas restantes en el mazo (total - pos)
    @property
    def cantidad(self) -> int:
        return self.TOTAL - self.pos

    #Mezcla la baraja
    def mezclar(self) -> None:
        """Baraja el mazo completo y lo deja listo para repartir desde cero.
        Parte siempre del orden base 0..51: el resultado depende SOLO del estado
        del generador (reproducible con semilla), no del orden anterior del mazo."""
        #Fijamos el orden del mazo al orden totalemente ordenado (notar que no genera una copia lo que lo hace eficiente)
        self.orden[:] = self._BASE
        #Luego barajamos el mazo usando el generador de números aleatorios
        self.rng.shuffle(self.orden)
        #Y reseteamos las cartas entregadas a 0
        self.pos = 0

    #Entrega una carta del mazo (entero 0-51) y avanza el puntero pos
    def entregar_carta(self) -> int:
        #En caso que no queden cartas entrega un error
        if self.pos >= self.TOTAL:
            raise ValueError("El mazo está vacío.")
        #Sacamos la carta de la posición actual del mazo y avanzamos el puntero pos
        carta = int(self.orden[self.pos])
        self.pos += 1
        return carta
    #En caso que queramos bifurcar el estado de decisiones de una partida clonando el mazo.
    def clonar(self) -> "Mazo":
        nuevo = Mazo(self.rng)  #Comparte el generador
        nuevo.orden = self.orden.copy()
        nuevo.pos = self.pos
        return nuevo

    def to_dict(self) -> dict:
        #Se guarda el orden completo y el puntero: al recargar,
        #entregar_carta() devuelve exactamente las mismas cartas.
        return {"orden": self.orden.tolist(), "pos": self.pos}

    @classmethod
    #Recupera la instancia de un mazo a partir de un diccionario.
    def from_dict(cls, data: dict, rng: np.random.Generator | None = None) -> "Mazo":
        mazo = cls(rng)
        mazo.orden = np.array(data["orden"], dtype=np.int8)
        mazo.pos = data["pos"]
        return mazo