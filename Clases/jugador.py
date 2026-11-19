#Creamos la clase jugador
class Jugador:
    def __init__(self, nombre: str, fichas_iniciales: int = 1000) -> None:
        #Nombre del jugador.
        self.nombre: str = nombre
        #Generamos una lista con su mano.
        self.mano: list[int] = []
        #Generamos una variable con la cantidad de fichas que tiene el jugador.
        self.fichas: int = fichas_iniciales
        #Generamos una variable con la cantidad de fichas que ha apostado en la mano actual.
        self.apuesta_actual: int = 0
        #Y finalmente un indicador si sigue activo en la mano (no se ha retirado).
        self.esta_activo: bool = True

    #Creamos una función que permita al jugador recibir una carta y agregarla a su mano.
    def recibir_carta(self, nueva_carta) -> None:
        if nueva_carta is not None:
            self.mano.append(nueva_carta)

    #Devolvemos cartas de la mano del jugador.
    def devolver_cartas(self) -> list:
        #Creamos una copia de la mano.
        cartas_entregadas = self.mano.copy()
        #vaciamos la mano del jugador y retornamos la copia de la mano.
        self.mano.clear()
        return cartas_entregadas

    #Un string que nos permita ver el estado del jugador (nombre, fichas y mano).
    def __str__(self) -> str:
        from .carta import nombre_de
        return f"{self.nombre} tiene {[nombre_de(c) for c in self.mano]}."

    #Función para que el jugador pueda apostar.
    def apostar(self, cantidad: int) -> int:
        #Creamos la cantidad real que es lo que puede apostar.
        cantidad_real = min(cantidad, self.fichas)
        #Restamos la cantidad real de fichas del jugador y sumamos la cantidad real a la apuesta actual.
        self.fichas -= cantidad_real
        self.apuesta_actual += cantidad_real
        return cantidad_real

    #Función para que el jugador pueda retirarse de la mano.
    def retirarse(self) -> None:
        #Simplemente cambiamos el estado del jugador a retirado.
        self.esta_activo = False

    #Función para que el jugador pueda recibir ganancias.
    def recibir_ganacias(self, cantidad: int) -> None:
        #Sumamos la cantidad de fichas recibidas a las fichas del jugador.
        self.fichas += cantidad

    #Función para preparar al jugador para una nueva mano.
    def preparar_nueva_mano(self) -> None:
        self.apuesta_actual = 0
        self.esta_activo = True
        self.mano.clear()

    #Si queremos clonar el estado de un jugador.
    def clonar(self) -> "Jugador":
        nuevo = Jugador(self.nombre, self.fichas)
        nuevo.mano = list(self.mano)
        nuevo.apuesta_actual = self.apuesta_actual
        nuevo.esta_activo = self.esta_activo
        return nuevo

    def to_dict(self) -> dict:
        return {
            "nombre": self.nombre,
            "mano": list(self.mano),
            "fichas": self.fichas,
            "apuesta_actual": self.apuesta_actual,
            "esta_activo": self.esta_activo,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Jugador":
        jugador = cls(nombre=data["nombre"], fichas_iniciales=data["fichas"])
        jugador.mano = list(data["mano"])
        jugador.apuesta_actual = data["apuesta_actual"]
        jugador.esta_activo = data["esta_activo"]
        return jugador