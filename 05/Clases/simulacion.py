"""
Simulación de partidas de Poker 1v1 para entrenar/evaluar agentes.

Una "política" es una función (o un objeto con __call__):

    politica(obs: np.ndarray, mascara: np.ndarray) -> int

(Si el objeto tiene el atributo `usa_obs = False`, recibe obs=None y no se
construye la observación: útil para políticas que no la necesitan.)

  obs     : vector de observación (ver motor.observacion / TAM_OBS)
  mascara : bool (5,) con las acciones discretas legales
  retorna : índice 0-4 de la acción (ver motor.NOMBRES_ACCION)

Toda la aleatoriedad usa np.random.Generator (reproducible con `semilla`).
"""

import numpy as np

from .jugador import Jugador
from .motor import MotorPoker, N_ACCIONES, RETIRARSE

#Clase encargada de entrenar el agente aleatorio
class PoliticaAleatoria:
    """Elige entre las acciones legales con probabilidades proporcionales a
    `pesos` (orden: retirarse, igualar/pasar, subir 1/2 pozo, subir pozo, all-in).

    El peso de retirarse es pequeño por defecto (0.15 frente a 1.0 de igualar):
    así el rival Random no regala manos retirándose a cada rato. Con
    peso_retiro=0 nunca se retira (salvo que sea lo único legal, que no ocurre)."""

    #Definimos pesos por defecto, particularmente damos poco peso a jugadas muy arriesgadas en el contexto de las partidas.
    PESOS_POR_DEFECTO = (0.15, 1.0, 0.35, 0.25, 0.10)
    usa_obs = False  # no mira la observación: Jugar se ahorra construirla

    def __init__(self, rng: np.random.Generator | None = None, pesos=None,
                 peso_retiro: float | None = None) -> None:
        #Extraemos la aleatoriedad
        self.rng = rng if rng is not None else np.random.default_rng()
        #Seteamos los pesos por defecto antes definidos
        self.pesos = np.array(self.PESOS_POR_DEFECTO if pesos is None else pesos, dtype=np.float64)
        if peso_retiro is not None:
            #Si consideramos la posibilidad de retiro, guardamos ese peso.
            self.pesos[RETIRARSE] = peso_retiro
    #Generamos el agente que elige unif. entre sus posibilidades
    @classmethod
    def uniforme(cls, rng: np.random.Generator | None = None,
                 peso_retiro: float | None = None) -> "PoliticaAleatoria":
        """Rival UNIFORME: todas las acciones legales con la misma probabilidad
        (salvo que se indique `peso_retiro`)."""
        return cls(rng, pesos=(1.0, 1.0, 1.0, 1.0, 1.0), peso_retiro=peso_retiro)

    def __call__(self, obs: np.ndarray, mascara: np.ndarray) -> int:
        #Calculamos el vector acumulado de los pesos con sus respectivas máscaras de acciones legales
        acumulado = np.cumsum(self.pesos * mascara)
        #Si por alguna razón la acumulación da menor o igual a cero, pasamos a uniforme
        if acumulado[-1] <= 0:  # pesos anulados: uniforme sobre las legales
            acumulado = np.cumsum(mascara.astype(np.float64))
        #Luego, generamos un número aleatorio y lo ponderamos por el acumulado final (es como lanzar un dardo en una regla de tamaño acumulado[-1])
        # y en base a donde cayó devuelve el indice correspondiente
        return int(np.searchsorted(acumulado, self.rng.random() * acumulado[-1], side="right"))

#Creamos una política que va rotando su forma de jugar entre las politicas posibles
class PoliticaMezcla:
    """Rival que EN CADA MANO elige al azar (uniforme o ponderado por `pesos`) una de varias
    políticas y juega esa mano completa con ella. El entorno llama a `nueva_mano(rng)` en cada reset().
    `conteo[i]` = cuántas manos se jugó contra la política i (para verificar el reparto)."""

    usa_obs = True  #algunas políticas (AgenteQ) necesitan la observación

    def __init__(self, politicas, nombres=None, pesos=None) -> None:
        #Recibe una lista de las politicas que puede usar
        self.politicas = list(politicas)
        #Les ponemos nombre
        self.nombres = list(nombres) if nombres is not None else [f"rival_{i}" for i in range(len(self.politicas))]
        #Generamos un conteo de cuantas y cuales politicas se ha enfrentado
        self.conteo = [0] * len(self.politicas)
        #Indice de la politica que esta jugando
        self.actual = 0
        #Establecemos los pesos y sus probabilidades
        if pesos is not None:
            p = np.asarray(pesos, dtype=np.float64)
            self.probs = p / p.sum()
        else:
            self.probs = None

    def nueva_mano(self, rng: np.random.Generator) -> None:
        #Generamos la estrategia que seguirá el agente en una nueva mano
        if self.probs is None:
            self.actual = int(rng.integers(len(self.politicas)))
        else:
            self.actual = int(rng.choice(len(self.politicas), p=self.probs))
        self.conteo[self.actual] += 1

    def __call__(self, obs, mascara) -> int:
        return self.politicas[self.actual](obs, mascara)


class Jugar:
    """Partida completa (manos sucesivas hasta que alguien se quede sin
    fichas, o hasta max_manos) entre dos políticas."""
    #Inicializamos la clase de juego, con las variables necesarias para una partida
    def __init__(
        self,
        politica_j1,
        politica_j2,
        nombre_j1: str = "Jugador 1",
        nombre_j2: str = "Jugador 2",
        fichas_iniciales: int = 1000,
        ciega_pequena: int = 10,
        ciega_grande: int = 20,
        max_manos: int | None = None,
        semilla: int | None = None,
        rng: np.random.Generator | None = None,
        permitir_retiro: bool = True,
        registrar_historial: bool = False,
    ) -> None:
        self.nombre_j1 = nombre_j1
        self.nombre_j2 = nombre_j2
        self.politicas = [politica_j1, politica_j2]
        self.fichas_iniciales = fichas_iniciales
        self.ciega_pequena = ciega_pequena
        self.ciega_grande = ciega_grande
        self.max_manos = max_manos
        self.rng = rng if rng is not None else np.random.default_rng(semilla)
        self.permitir_retiro = permitir_retiro
        self.registrar_historial = registrar_historial
    #Entrega dos objetos nuevos de jugadores reiniciados
    def _nuevos_jugadores(self) -> tuple[Jugador, Jugador]:
        return (
            Jugador(self.nombre_j1, fichas_iniciales=self.fichas_iniciales),
            Jugador(self.nombre_j2, fichas_iniciales=self.fichas_iniciales),
        )
    #Definimos una función para jugar las manos
    def _jugar_mano(self, motor: MotorPoker, dealer_idx: int) -> None:
        #Extraemos quién será el dealer
        motor.dealer_idx = dealer_idx
        #Iniciamos la partida, apostando ciegas y entregando cartas iniciales
        motor.iniciar_mano()
        #Extraemos las políticas de juego de ambos jugadores
        politicas = self.politicas
        #Mientras no esté terminada la mano
        while not motor.esta_terminada():
            #Vemos a quien le toca
            idx = motor.turno_idx
            if idx is None:
                break
            #Obtenemos su politica
            pol = politicas[idx]
            #Generamos las observaciones que puede ver de la partida
            obs = motor.observacion(idx) if getattr(pol, "usa_obs", True) else None
            #Procesa con las observaciones y las acciones posible que le conviene hacer segun la politica
            accion = pol(obs, motor.mascara_acciones())
            #Aplica la acción
            motor.aplicar_accion_id(accion)

    def simular(self) -> dict:
        """Juega la partida y devuelve resumen (+ historial por mano si
        registrar_historial=True)."""
        #Generamos los jugadores y el motor de partida
        j1, j2 = self._nuevos_jugadores()
        motor = MotorPoker(
            j1, j2,
            ciega_pequena=self.ciega_pequena,
            ciega_grande=self.ciega_grande,
            rng=self.rng,
            permitir_retiro=self.permitir_retiro,
            registrar_historial=self.registrar_historial,
        )
        dealer_idx = 0
        manos = 0
        historial_manos = []
        #Mientras los jugadores tengan fichas, pueden jugar
        while j1.fichas > 0 and j2.fichas > 0:
            #Juegan sus manos
            self._jugar_mano(motor, dealer_idx)
            manos += 1
            #Registramos la data
            if self.registrar_historial:
                historial_manos.append(motor.historial)
            #Rotamos el dealer para la siguiente mano
            dealer_idx = 1 - dealer_idx
            if self.max_manos is not None and manos >= self.max_manos:
                break
        #Si alguno de los jugadores quiebra, entonces ganó el otro
        if j1.fichas == 0:
            ganador = j2.nombre
        elif j2.fichas == 0:
            ganador = j1.nombre
        else:
            # Corte por max_manos: gana el que tenga mayor cantidad de fichas
            if j1.fichas > j2.fichas:
                ganador = j1.nombre
            elif j2.fichas > j1.fichas:
                ganador = j2.nombre
            else:
                ganador = "Empate"  # Por si tienen exactamente las mismas fichas
        #Retorna los resultados de la partida
        return {
            "ganador": ganador,
            "fichas_finales": {j1.nombre: j1.fichas, j2.nombre: j2.fichas},
            "manos_jugadas": manos,
            "historial": historial_manos,
        }
