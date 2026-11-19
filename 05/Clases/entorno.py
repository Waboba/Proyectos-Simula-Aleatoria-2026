"""
Entorno Gymnasium para Texas Hold'em heads-up.

Un episodio = UNA MANO. Al empezar cada mano ambos jugadores tienen
`fichas_iniciales` (se reinician), así el estado no necesita "recordar" el
historial de fichas de la partida. El agente que aprende es siempre el
jugador 0; el rival es cualquier política `f(obs, mascara) -> int`
(PoliticaAleatoria, otro AgenteQ congelado, etc.) que el entorno
ejecuta automáticamente entre las decisiones del agente. Cambiar el rival
(`env.oponente = ...`) es lo que permite entrenar "contra la versión anterior".

  observation_space : Box(13,) float32  (ver motor.observacion; -1 = vacío)
  action_space      : Discrete(5)       (0 retirarse, 1 igualar/pasar,
                                         2 subir 1/2 pozo, 3 subir pozo, 4 all-in)
  recompensa        : 0 durante la mano; al terminar:
                      modo_recompensa="fichas"   -> fichas ganadas/perdidas en bb
                      modo_recompensa="victoria" -> +1 gana, -1 pierde, 0 empata
                      (en ambos casos info["recompensa_fichas"] guarda las fichas)
  info              : "mascara" (acciones legales, bool[5]); al terminar también
                      "recompensa_fichas" y "gano" (True / False / None si empate)

El dealer se alterna entre manos (posición justa). reset(seed=s) reinicia la
secuencia: misma semilla -> mismas cartas y mismas posiciones.

OJO (poco común pero posible): si el rival actúa primero y se retira, la mano
termina ANTES de que el agente decida nada. En ese caso reset() devuelve
info["mano_terminada"] = True y la recompensa en info["recompensa"]; el
agente no tiene decisión que tomar. Así no se descartan (ni se sesgan) esas manos.

La máscara de acciones también está en env.action_masks() (convención de
sb3-contrib MaskablePPO).
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .jugador import Jugador
from .motor import MotorPoker, N_ACCIONES, TAM_OBS, IGUALAR_PASAR
from .simulacion import PoliticaAleatoria


class PokerHeadsUpEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        oponente=None,
        fichas_iniciales: int = 1000,
        ciega_pequena: int = 100,
        ciega_grande: int = 200,
        permitir_retiro: bool = True,
        alternar_dealer: bool = True,
        modo_ilegal: str = "igualar",
        modo_recompensa: str = "fichas",
        peso_fichas: float = 0.5,
    ) -> None:
        super().__init__()
        if modo_ilegal not in ("igualar", "error"):
            raise ValueError("modo_ilegal debe ser 'igualar' o 'error'")
        if modo_recompensa not in ("fichas", "victoria", "mixta"):
            raise ValueError("modo_recompensa debe ser 'fichas', 'victoria' o 'mixta'")
        self.modo_recompensa = modo_recompensa
        self.peso_fichas = peso_fichas
        self.fichas_iniciales = fichas_iniciales
        self.ciega_grande = ciega_grande
        self.alternar_dealer = alternar_dealer
        #Definimos que hacer si se hace una acción ilegal
        self.modo_ilegal = modo_ilegal
        #Configuramos el espacio observable y accionable del agente
        self.observation_space = spaces.Box(low=-1.0, high=np.inf, shape=(TAM_OBS,), dtype=np.float32)
        self.action_space = spaces.Discrete(N_ACCIONES)
        #Definimos el oponente de nuestro agente a entrenar
        self._oponente_por_defecto = oponente is None
        self.oponente = oponente if oponente is not None else PoliticaAleatoria()
        #Lista de jugadores
        self.jugadores = [Jugador("Agente", fichas_iniciales), Jugador("Rival", fichas_iniciales)]
        #Iniciamos el motor de la partida
        self.motor = MotorPoker(
            self.jugadores[0], self.jugadores[1],
            ciega_pequena=ciega_pequena, ciega_grande=ciega_grande,
            permitir_retiro=permitir_retiro, registrar_historial=False,
        )
        #
        self._n_manos = 0
        self._recompensa_pendiente = 0.0
        self._info_pendiente: dict = {}

    #Funciones útiles

    #Crea una máscara de acciones legales
    def action_masks(self) -> np.ndarray:
        return self.motor.mascara_acciones()
    #Crea una copia de la máscara y la almacena en un diccionario
    def _info(self) -> dict:
        return {"mascara": self.motor.mascara_acciones().copy()}
    #Método para determinar la recompesa para el agente
    def _info_terminal(self) -> tuple[float, dict]:
        fichas = self.motor.recompensa(0)
        gano = True if fichas > 0 else (False if fichas < 0 else None)
        if self.modo_recompensa == "victoria":
            recompensa = 1.0 if gano is True else (-1.0 if gano is False else 0.0)
        elif self.modo_recompensa == "mixta":
            v = 1.0 if gano is True else (-1.0 if gano is False else 0.0)
            w = getattr(self, "peso_fichas", 0.5)
            # Asegúrate de usar la variable de fichas iniciales de tu entorno (suele llamarse fichas_iniciales)
            recompensa = (1.0 - w) * v + w * (fichas / self.fichas_iniciales)
        else:
            recompensa = fichas / self.ciega_grande
        info = {"mascara": np.zeros(N_ACCIONES, dtype=bool),
                "recompensa_fichas": fichas, "gano": gano}
        return recompensa, info
    #Método para obligar al rival a jugar para que le toque a nuestro agente
    def _jugar_rival_hasta_turno_del_agente(self) -> None:
        motor = self.motor
        while not motor.esta_terminada() and motor.turno_idx == 1:
            obs_rival = motor.observacion(1) if getattr(self.oponente, "usa_obs", True) else None
            accion = self.oponente(obs_rival, motor.mascara_acciones())
            motor.aplicar_accion_id(int(accion))

    #Gymnasium
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        #Establecemos la aleatoriedad para reproducibilidad
        rng = self.np_random
        if seed is not None:
            self._n_manos = 0
        self.motor.rng = rng
        self.motor.mesa.mazo.rng = rng
        if self._oponente_por_defecto:
            self.oponente.rng = rng
        if hasattr(self.oponente, "nueva_mano"):  #PoliticaMezcla: elige rival para ESTA mano
            self.oponente.nueva_mano(rng)
        #Luego, invocamos a los jugadores, les entregamos sus fichas iniciales
        for jugador in self.jugadores:
            jugador.fichas = self.fichas_iniciales
        #Si en options forzamos un delear se respeta, si no se alterna de manera usual
        if options and "dealer_idx" in options:
            self.motor.dealer_idx = int(options["dealer_idx"])
        elif self.alternar_dealer:
            self.motor.dealer_idx = self._n_manos % 2
        self._n_manos += 1
        #Iniciamos la mano
        self.motor.iniciar_mano()
        #Juega el rival hasta que nos toque
        self._jugar_rival_hasta_turno_del_agente()
        #Aquí vemos el fold de rival en preflop que rompe el esquema, si el rival se rinde antes de que el agente juegue este queda en el aire.
        obs = self.motor.observacion(0)
        #Dado que reset no puede entregar recompensas generamos unas variables internas con los datos de la victoria del agente 
        if self.motor.esta_terminada():
            recompensa, info = self._info_terminal()
            info["mano_terminada"] = True
            info["recompensa"] = recompensa
            self._recompensa_pendiente, self._info_pendiente = recompensa, info
            return obs, info
        #Si el rival no se rindió limpiamos la info pendiente
        self._info_pendiente = {}
        #Genera máscara permitidos
        info = self._info()
        #Si la mano está actividada
        info["mano_terminada"] = False
        #Le entrega lo necesario al agente para jugar
        return obs, info

    #Ahora definimos el método step que es el encargado de que el agente juegue
    def step(self, accion: int):
        motor = self.motor
        #Si tenemos info pendiente guardada del caso borde
        if self._info_pendiente:  
            #Limpiamos la información
            info, self._info_pendiente = self._info_pendiente, {}
            #Retornamos toda la info y parametros necesarios para el step: observation, reward, terminated, truncated, info (Ver Glosario)
            return motor.observacion(0), self._recompensa_pendiente, True, False, info
        #Si la partida ya estaba terminada entonces no hay nada que hacer
        if motor.esta_terminada():
            raise RuntimeError("La mano terminó: llama a reset().")
        #Colocamos la acción en formato correcto para que sea leeible por nuestro motor
        accion = int(accion)
        #Verificamos si la acción es ilegal (se sale del rango o bien no está dentro de las posibles acciones)
        ilegal = not (0 <= accion < N_ACCIONES and motor.mascara_acciones()[accion])
        #Si la jugada es ilegal
        if ilegal:
            #Lanza error
            if self.modo_ilegal == "error":
                raise ValueError(f"Acción ilegal: {accion}")
            #Si no simplemente igualamos pasamos
            accion = IGUALAR_PASAR  # siempre legal cuando el agente tiene el turno
        #Aplica la acción legal (o ilegal check pass)
        motor.aplicar_accion_id(accion)
        #Luego, activamos al rival y no se detiene hasta que nos toque o se acabe la mano
        self._jugar_rival_hasta_turno_del_agente()
        #Tomamos la foto del estado actual del agente
        obs = motor.observacion(0)
        #Si está terminada
        if motor.esta_terminada():
            #Damos recompensas y retornamos lo necesario del step
            recompensa, info = self._info_terminal()
            info["accion_ilegal"] = ilegal
            return obs, recompensa, True, False, info
        info = self._info()
        info["accion_ilegal"] = ilegal
        return obs, 0.0, False, False, info