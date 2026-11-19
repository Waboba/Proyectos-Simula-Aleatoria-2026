import numpy as np

from .jugador import Jugador
from .mazo import Mazo
from .evaluadorPoker import evaluar

#Iniciamos la mesa
class Mesa:
    def __init__(self, rng: np.random.Generator | None = None) -> None:
        #Partimos de un pozo vacío, sin cartas comunitarias y con un mazo mezclado
        self.pozo: int = 0
        self.cartas_comunitarias: list[int] = []
        self.mazo = Mazo(rng)
        self.mazo.mezclar()
    #Recolectar apuestas de todos los jugadores y sumarlas al pozo, reiniciando la apuesta actual de cada jugador a 0.
    def recolectar_apuestas(self, jugadores: list[Jugador]) -> None:
        for jugador in jugadores:
            self.pozo += jugador.apuesta_actual
            jugador.apuesta_actual = 0
    # Entregar el pozo al ganador y reiniciar el pozo a 0.
    def entregar_pozo(self, ganador: Jugador) -> None:
        ganador.recibir_ganacias(self.pozo)
        self.pozo = 0
    #Reiniciar la mesa para una nueva mano: pozo a 0, cartas comunitarias vacías, preparar a cada jugador y mezclar el mazo.
    def reiniciar_mesa(self, jugadores: list[Jugador]) -> None:
        self.pozo = 0
        self.cartas_comunitarias.clear()
        for jugador in jugadores:
            jugador.preparar_nueva_mano()
        self.mazo.mezclar()  # reutiliza el mismo mazo: solo re-baraja
    #Repartir las cartas comunitarias: flop (3), turn (1) y river (1). Se quema una carta antes de cada ronda.
    def repartir_flop(self) -> None:
        self.mazo.entregar_carta()  # carta quemada
        for _ in range(3):
            self.cartas_comunitarias.append(self.mazo.entregar_carta())

    def repartir_turn(self) -> None:
        self.mazo.entregar_carta()
        self.cartas_comunitarias.append(self.mazo.entregar_carta())

    def repartir_river(self) -> None:
        self.mazo.entregar_carta()
        self.cartas_comunitarias.append(self.mazo.entregar_carta())
    # Determinar el/los ganador/es de la mano entre los jugadores activos, evaluando sus manos con las cartas comunitarias. 
    def determinar_ganador(self, jugadores: list[Jugador]) -> list[Jugador]:
        activos = [j for j in jugadores if j.esta_activo]
        if len(activos) <= 1:
            return activos
        comunitarias = self.cartas_comunitarias
        rangos = [evaluar(j.mano + comunitarias) for j in activos]
        mejor = max(rangos)
        return [j for j, r in zip(activos, rangos) if r == mejor]
    #Repartimos el pozo entre los ganadores (en caso de empate) y reiniciamos el pozo a 0. El sobrante (si el pozo no es divisible) se le da al primer ganador.
    def repartir_pozo(self, ganadores: list[Jugador]) -> None:
        """Reparte el pozo entre uno o varios ganadores (empate)."""
        if not ganadores:
            return
        parte, sobrante = divmod(self.pozo, len(ganadores))
        for i, ganador in enumerate(ganadores):
            #el sobrante (pozo no divisible) se lo lleva el primero: no se pierden fichas
            ganador.recibir_ganacias(parte + (sobrante if i == 0 else 0))
        self.pozo = 0
    # Clonar la mesa: se crea una nueva instancia de Mesa con el mismo pozo, cartas comunitarias y un mazo clonado.
    def clonar(self) -> "Mesa":
        nueva = Mesa.__new__(Mesa)
        nueva.pozo = self.pozo
        nueva.cartas_comunitarias = list(self.cartas_comunitarias)
        nueva.mazo = self.mazo.clonar()
        return nueva

    def to_dict(self) -> dict:
        return {
            "pozo": self.pozo,
            "cartas_comunitarias": list(self.cartas_comunitarias),
            "mazo": self.mazo.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict, rng: np.random.Generator | None = None) -> "Mesa":
        mesa = cls.__new__(cls)
        mesa.pozo = data["pozo"]
        mesa.cartas_comunitarias = list(data["cartas_comunitarias"])
        mesa.mazo = Mazo.from_dict(data["mazo"], rng)
        return mesa