"""
Motor de Poker Texas Hold'em 1v1 (heads-up).

Módulo "puro": no imprime ni pide input(). Recibe acciones como datos y
transiciona el estado. Permite guardar/cargar manos como JSON, clonar el
estado para "bifurcar", y ahora además expone una interfaz pensada para RL:

- Cartas como enteros 0-51 (ver carta.py).
- Aleatoriedad vía np.random.Generator (parámetro `rng`, reproducible).
- Espacio de acciones DISCRETO de 5 acciones con máscara de legalidad:
        0 RETIRARSE | 1 IGUALAR/PASAR | 2 SUBIR medio pozo | 3 SUBIR pozo | 4 ALL-IN
    (mascara_acciones() + aplicar_accion_id()).
- Observación numérica fija (observacion()) y recompensa por mano (recompensa()).
- El historial de eventos se puede desactivar (registrar_historial=False).

La interfaz anterior por texto (aplicar_accion("subir", monto=...)) se mantiene.

Reglas de apuestas (cambios respecto a la versión anterior):
- Solo se puede retirar quien enfrenta una apuesta (retirarse cuando se
    podía pasar gratis era siempre una jugada dominada). Con
    permitir_retiro=False la opción desaparece por completo.
- Si alguien queda all-in, el rival SIEMPRE puede decidir igualar o retirarse
    (antes la ronda se cerraba sin darle esa opción) y no puede "subir" contra
    un rival sin fichas.
- La parte de una apuesta que el rival no puede igualar se devuelve a quien
    la puso (apuesta no igualada), así no hay side pots ni fichas en juego
    que nadie pueda ganar.
"""
import json
import numpy as np

from .jugador import Jugador
from .mesa import Mesa

FASES = ("PREFLOP", "FLOP", "TURN", "RIVER", "TERMINADA")
FASE_ID = {fase: i for i, fase in enumerate(FASES)}
ACCIONES_VALIDAS = ("pasar", "igualar", "subir", "retirarse")

# Espacio de acciones discreto (índices de la máscara / salida de la política)
RETIRARSE, IGUALAR_PASAR, SUBIR_MEDIO_POZO, SUBIR_POZO, ALL_IN = range(5)
N_ACCIONES = 5
#Aquí subir 1/2 pozo equivale a calcular el pozo tras igualar y subir la mitad de eso, y subir pozo = subir el pozo tras igualar.
NOMBRES_ACCION = ("retirarse", "igualar/pasar", "subir 1/2 pozo", "subir pozo", "all-in")

# Layout de la observación (float32, tamaño TAM_OBS). Cartas = id 0-51, -1 = vacío.
#   [0:2] mano propia | [2:7] mesa | [7] fase 0-3 | [8] pozo total (incl. apuestas
#   en curso) | [9] fichas propias | [10] fichas rival | [11] a igualar |
#   [12] 1 si soy dealer (ciega chica)
TAM_OBS = 13

#Definimos la clase motor poker
class MotorPoker:
    #Primero se establecen los atributos de la clase
    def __init__(
        self,
        jugador_a: Jugador,
        jugador_b: Jugador,
        ciega_pequena: int = 100,
        ciega_grande: int = 20,
        dealer_idx: int = 0,
        rng: np.random.Generator | None = None,
        permitir_retiro: bool = True,
        registrar_historial: bool = True,
    ) -> None:
        self.jugadores: list[Jugador] = [jugador_a, jugador_b]
        self.rng = rng if rng is not None else np.random.default_rng()
        self.mesa = Mesa(self.rng)
        self.ciega_pequena = ciega_pequena
        self.ciega_grande = ciega_grande
        self.dealer_idx = dealer_idx  # 0 o 1: quien reparte (ciega pequeña)
        self.permitir_retiro = permitir_retiro
        self.registrar_historial = registrar_historial
        #En que fase estamos  
        self.fase: str | None = None
        #De quien es el turno
        self.turno_idx: int | None = None
        #Guarda el estado de las actuaciones de los jugadores
        self.ya_actuo: list[bool] = [False, False]
        #Quienes ganaron
        self.ganadores: list[str] | None = None
        #El historial de partida
        self.historial: list[dict] = []
        #Cuanto tenía al inicio de la mano para comparar al final y calcular la recompensa
        self.fichas_inicio_mano: list[int] = [jugador_a.fichas, jugador_b.fichas]
        #Guarda un caché de acciones durante el turno para no recalcularlo varias veces
        self._cache_turno: tuple | None = None  # (mascara, montos) del turno actual

    # ------------------------------------------------------------------
    # Inicio de mano
    # ------------------------------------------------------------------

    def iniciar_mano(self) -> None:
        #Reseteamos las variables de la mano
        self.mesa.reiniciar_mesa(self.jugadores)
        self.ganadores = None
        self._cache_turno = None
        self.historial = []  #lista nueva: referencias a historiales viejos no se pisan
        self.fichas_inicio_mano = [j.fichas for j in self.jugadores]

        #Establecemos el dealer y el no dealer, y hacemos que posteen las ciegas
        dealer = self.jugadores[self.dealer_idx]
        no_dealer = self.jugadores[1 - self.dealer_idx]
        dealer.apostar(self.ciega_pequena)
        no_dealer.apostar(self.ciega_grande)

        #Entregamos las cartas correspondientes
        for _ in range(2):
            for jugador in self.jugadores:
                jugador.recibir_carta(self.mesa.mazo.entregar_carta())
        #Inicializamos el PREFLOP
        self.fase = "PREFLOP"
        #Fijamos que nadie haya jugado
        self.ya_actuo = [False, False]
        #Y en caso de estar guardando historial, registramos el evento de inicio de mano
        if self.registrar_historial:
            self._registrar_evento({
                "tipo": "iniciar_mano",
                "dealer": dealer.nombre,
                "cartas": {j.nombre: list(j.mano) for j in self.jugadores},
            })

        # Si alguien quedó all-in solo por postear la ciega y el rival ya
        # igualó/superó su apuesta, no hay nada que decidir: se reparten
        # las calles hasta el showdown. Si el rival aún debe igualar, decide.
        if self._ronda_cerrada():
            self.turno_idx = None
            self._cerrar_ronda_apuestas()
        else:
            self.turno_idx = self.dealer_idx  # heads-up: dealer actúa primero pre-flop

    # ------------------------------------------------------------------
    # Consulta de estado / acciones legales
    # ------------------------------------------------------------------

    #En base al indice del turno, extraemos al jugador que tiene el turno, o None si no hay turno
    def jugador_en_turno(self) -> Jugador | None:
        return None if self.turno_idx is None else self.jugadores[self.turno_idx]

    #Verificamos si terminó
    def esta_terminada(self) -> bool:
        return self.fase == "TERMINADA"

    #Nos entrega los jugadores y como está el contexto de las apuestas
    def _contexto_turno(self) -> tuple[Jugador, Jugador, int]:
        if self.turno_idx is None:
            raise ValueError("No hay un jugador en turno actualmente.")
        jugador = self.jugadores[self.turno_idx]
        otro = self.jugadores[1 - self.turno_idx]
        return jugador, otro, max(otro.apuesta_actual - jugador.apuesta_actual, 0)

    #Entrega las posibles acciones de un jugador
    def _acciones_str(self, jugador: Jugador, otro: Jugador, a_igualar: int) -> list[str]:
        acciones: list[str] = []
        if jugador.fichas > 0:
            acciones.append("igualar" if a_igualar > 0 else "pasar")
            if jugador.fichas > a_igualar and otro.fichas > 0:
                acciones.append("subir")
            if self.permitir_retiro and a_igualar > 0:
                acciones.append("retirarse")
        return acciones

    def _mascara_y_montos(self, jugador: Jugador, otro: Jugador, a_igualar: int):
        """Máscara bool de las 5 acciones discretas y los montos de subida
        (medio pozo, pozo, all-in) -- el monto es lo que se sube POR ENCIMA de igualar."""
        #Creamos una máscara de N_acciones falsos que iremos desbloqueando a medida que se cumplan las condiciones
        mascara = np.zeros(N_ACCIONES, dtype=bool)
        #La máxima subida
        max_sub = jugador.fichas - a_igualar
        #Calculamos el pozo tras igualar
        pozo_tras_igualar = self.mesa.pozo + jugador.apuesta_actual + otro.apuesta_actual + a_igualar
        #Y calculamos los valores numéricos de la subida
        medio = max(self.ciega_grande, pozo_tras_igualar // 2)
        pozo = max(self.ciega_grande, pozo_tras_igualar)
        #Ahora verificamos que acciones son válidas
        if jugador.fichas > 0: #Si tiene fichas
            #Entonces puede igualar y pasar
            mascara[IGUALAR_PASAR] = True
            #Si permitimos el retiro y debemos igualar
            if self.permitir_retiro and a_igualar > 0:
                #Permitimos retirarse
                mascara[RETIRARSE] = True
            #Si la máxima cantidad de fichas que podemos apostar despues de igualar es >0 y el rival tiene fichas
            if max_sub > 0 and otro.fichas > 0:
                #Desbloqueamos el ALL IN
                mascara[ALL_IN] = True
                #Si nuestra máxima apuesta es mayor el medio pozo
                if medio < max_sub:
                    #Desbloqueamos medio pozo
                    mascara[SUBIR_MEDIO_POZO] = True
                #Si medio pozo es menor al pozo y esto a su vez menor a nuestra máxima apuesta
                if medio < pozo < max_sub:
                    #Desbloqueamos subir pozo
                    mascara[SUBIR_POZO] = True
        #Finalmente, retornamos la máscara de acciones permitidas y los valores que se representan cada acción que apuesta.
        return mascara, (medio, pozo, max_sub)

    #Función que usa lo anterior para determinar que acciones se pueden hacer y como hacerlas en un diccionario
    def acciones_legales(self) -> dict:
        """Qué puede hacer el jugador en turno (formato original + 'mascara')."""
        if self.fase == "TERMINADA" or self.turno_idx is None:
            return {"jugador": None, "acciones": [], "a_igualar": 0,
                    "fichas_disponibles": 0, "mascara": np.zeros(N_ACCIONES, dtype=bool)}
        jugador, otro, a_igualar = self._contexto_turno()
        return {
            "jugador": jugador.nombre,
            "acciones": self._acciones_str(jugador, otro, a_igualar),
            "a_igualar": a_igualar,
            "fichas_disponibles": jugador.fichas,
            "mascara": self._mascara_y_montos(jugador, otro, a_igualar)[0],
        }
    #Obtiene la máscara de acciones permitidas
    def mascara_acciones(self) -> np.ndarray:
        """Máscara bool (N_ACCIONES,) de acciones discretas legales ahora."""
        if self.fase == "TERMINADA" or self.turno_idx is None:
            return np.zeros(N_ACCIONES, dtype=bool)
        return self._mascara_turno()[0]
    #Obtiene el contexto del turno y con ello calcula el contexto completo de las acciones guardandolo en caché
    def _mascara_turno(self):
        if self._cache_turno is None:
            jugador, otro, a_igualar = self._contexto_turno()
            self._cache_turno = self._mascara_y_montos(jugador, otro, a_igualar)
        return self._cache_turno
    #Función para crear el vector de las cosas que conoce el agente
    def observacion(self, idx: int | None = None) -> np.ndarray:
        """Vector float32 (TAM_OBS,) desde el punto de vista del jugador idx
        (por defecto quien tiene el turno). No incluye las cartas del rival."""
        if idx is None:
            idx = self.turno_idx if self.turno_idx is not None else 0
        yo = self.jugadores[idx]
        rival = self.jugadores[1 - idx]
        obs = np.full(TAM_OBS, -1.0, dtype=np.float32)

        if yo.mano:
            obs[0:len(yo.mano)] = [float(c) for c in yo.mano]

        comunitarias = self.mesa.cartas_comunitarias
        if comunitarias:
            obs[2:2 + len(comunitarias)] = [float(c) for c in comunitarias]

        obs[7] = float(FASE_ID.get(self.fase, -1)) if self.fase is not None else -1.0
        obs[8] = float(self.mesa.pozo + yo.apuesta_actual + rival.apuesta_actual)
        obs[9] = float(yo.fichas)
        obs[10] = float(rival.fichas)
        obs[11] = float(max(rival.apuesta_actual - yo.apuesta_actual, 0))
        obs[12] = 1.0 if idx == self.dealer_idx else 0.0
        return obs
    #Función para calcular la recompensa
    def recompensa(self, idx: int) -> int:
        """Ganancia neta de fichas del jugador idx en la mano actual
        (definida al terminar la mano)."""
        return self.jugadores[idx].fichas - self.fichas_inicio_mano[idx]

    # ------------------------------------------------------------------
    # Aplicar acciones
    # ------------------------------------------------------------------

    #Aplica las acciones revisando los casos borde no válidos
    def aplicar_accion(self, accion: str, monto: int = 0) -> None:
        if self.fase == "TERMINADA":
            raise ValueError("La mano ya terminó; no se pueden aplicar más acciones.")
        if accion not in ACCIONES_VALIDAS:
            raise ValueError(f"Acción desconocida: {accion!r}")
        if self.turno_idx is None:
            raise ValueError("No hay un jugador en turno para aplicar la acción.")

        jugador, otro, a_igualar = self._contexto_turno()
        if accion not in self._acciones_str(jugador, otro, a_igualar):
            raise ValueError(f"Acción ilegal ahora para {jugador.nombre}: {accion!r}")
        if accion == "subir":
            if monto <= 0:
                raise ValueError("Para 'subir' hay que indicar un monto positivo.")
            monto = min(monto, jugador.fichas - a_igualar)
        self._ejecutar(jugador, otro, a_igualar, accion, monto)
    #Lo mismo de arriba pero en el formato ID para la IA
    def aplicar_accion_id(self, accion_id: int) -> None:
        """Aplica una acción del espacio discreto (0-4); ver NOMBRES_ACCION."""
        if self.fase == "TERMINADA" or self.turno_idx is None:
            raise ValueError("No hay un jugador en turno para aplicar la acción.")
        jugador, otro, a_igualar = self._contexto_turno()
        mascara, montos = self._mascara_turno()
        if not 0 <= accion_id < N_ACCIONES or not mascara[accion_id]:
            raise ValueError(f"Acción ilegal ahora para {jugador.nombre}: id={accion_id}")

        if accion_id == RETIRARSE:
            self._ejecutar(jugador, otro, a_igualar, "retirarse", 0)
        elif accion_id == IGUALAR_PASAR:
            self._ejecutar(jugador, otro, a_igualar, "igualar" if a_igualar > 0 else "pasar", 0)
        else:
            medio, pozo, max_sub = montos
            monto = {SUBIR_MEDIO_POZO: medio, SUBIR_POZO: pozo, ALL_IN: max_sub}[accion_id]
            self._ejecutar(jugador, otro, a_igualar, "subir", monto)
    #Realiza la accion solicitada utilizando los métodos de jugador y guarda el historial
    def _ejecutar(self, jugador: Jugador, otro: Jugador, a_igualar: int, accion: str, monto: int) -> None:
        self._cache_turno = None
        if accion == "retirarse":
            jugador.retirarse()
            if self.registrar_historial:
                self._registrar_evento({"fase": self.fase, "jugador": jugador.nombre, "accion": "retirarse"})
            self._finalizar_por_retiro()
            return

        if accion == "subir":
            jugador.apostar(a_igualar + monto)
            assert self.turno_idx is not None
            self.ya_actuo[self.turno_idx] = True
            self.ya_actuo[1 - self.turno_idx] = False  #el otro debe responder
        else:  #igualar / pasar
            if a_igualar > 0:
                jugador.apostar(a_igualar)
            assert self.turno_idx is not None
            self.ya_actuo[self.turno_idx] = True

        if self.registrar_historial:
            evento = {"fase": self.fase, "jugador": jugador.nombre, "accion": accion}
            if accion == "subir":
                evento["monto"] = monto
            self._registrar_evento(evento)
        self._avanzar_turno()

    # ------------------------------------------------------------------
    # Internos: avance de turno / calle / showdown
    # ------------------------------------------------------------------
    #Verifica cuando se cierra una ronda (revisa duelo de all in, uno all in y ninguno all in)
    def _ronda_cerrada(self) -> bool:
        a, b = self.jugadores
        a_all_in = a.fichas == 0
        b_all_in = b.fichas == 0
        if a_all_in or b_all_in:
            if a_all_in and b_all_in:
                return True
            # Uno solo all-in: la ronda se cierra solo si el otro ya igualó
            # (o superó) esa apuesta; si no, el otro todavía debe decidir.
            con_fichas, sin_fichas = (b, a) if a_all_in else (a, b)
            return con_fichas.apuesta_actual >= sin_fichas.apuesta_actual
        return a.apuesta_actual == b.apuesta_actual and self.ya_actuo[0] and self.ya_actuo[1]
    #Cambia de turno
    def _avanzar_turno(self) -> None:
        if self.turno_idx is None:
            return
        if self._ronda_cerrada():
            self._cerrar_ronda_apuestas()
        else:
            self.turno_idx = 1 - self.turno_idx
    #Devuelve el exceso cuando un jugador tiene menos fichas de las que puede para igualar
    def _devolver_apuesta_no_igualada(self) -> None:
        a, b = self.jugadores
        if a.apuesta_actual == b.apuesta_actual:
            return
        mayor, menor = (a, b) if a.apuesta_actual > b.apuesta_actual else (b, a)
        exceso = mayor.apuesta_actual - menor.apuesta_actual
        mayor.apuesta_actual -= exceso
        mayor.fichas += exceso
    #Cierra la ronda de apuestas y avanza a la siguiente fase
    def _cerrar_ronda_apuestas(self) -> None:
        #Devolvemos las apuestas con excedentes y la mesa recoge lo que se haya apostado
        self._cache_turno = None
        self._devolver_apuesta_no_igualada()
        self.mesa.recolectar_apuestas(self.jugadores)
        #Dependiendo de la fase, realizamos o bien el showdown o pasamos a la siguiente etapa
        if self.fase == "RIVER":
            self._showdown()
            return
        if self.fase is None or self.fase == "TERMINADA":
            return

        siguiente_fase = {"PREFLOP": "FLOP", "FLOP": "TURN", "TURN": "RIVER"}.get(self.fase)
        if siguiente_fase is None:
            return

        #Repartimos las cartas de la siguiente fase
        reparto = {"PREFLOP": self.mesa.repartir_flop, "FLOP": self.mesa.repartir_turn, "TURN": self.mesa.repartir_river}
        reparto[self.fase]()
        self.fase = siguiente_fase
        if self.registrar_historial:
            self._registrar_evento({
                "tipo": "nueva_calle",
                "fase": self.fase,
                "mesa": list(self.mesa.cartas_comunitarias),
            })

        #Alguien all-in: ya no hay más apuestas posibles; se reparten las
        #calles que falten sin pedir acciones, hasta el showdown.
        if any(j.fichas == 0 for j in self.jugadores):
            self._cerrar_ronda_apuestas()
            return

        self.turno_idx = 1 - self.dealer_idx  # postflop: actúa primero el no-dealer
        self.ya_actuo = [False, False]
    #Que termine la ronda si alguien se retiró
    def _finalizar_por_retiro(self) -> None:
        #Las apuestas en curso todavía no están en el pozo; hay que
        #recolectarlas antes de repartir para conservar el total de fichas.
        self.mesa.recolectar_apuestas(self.jugadores)
        #Como el otro foldeó solo hay un ganador
        ganadores = self.mesa.determinar_ganador(self.jugadores)
        self.mesa.repartir_pozo(ganadores)
        self.ganadores = [g.nombre for g in ganadores]
        self.fase = "TERMINADA"
        self.turno_idx = None
        #Registramos
        if self.registrar_historial:
            self._registrar_evento({"tipo": "fin_mano", "motivo": "retiro", "ganadores": self.ganadores})
    #Fase final de la mano
    def _showdown(self) -> None:
        ganadores = self.mesa.determinar_ganador(self.jugadores)
        self.mesa.repartir_pozo(ganadores)
        self.ganadores = [g.nombre for g in ganadores]
        self.fase = "TERMINADA"
        self.turno_idx = None
        if self.registrar_historial:
            self._registrar_evento({
                "tipo": "fin_mano",
                "motivo": "showdown",
                "ganadores": self.ganadores,
                "mesa": list(self.mesa.cartas_comunitarias),
            })

    def _registrar_evento(self, evento: dict) -> None:
        self.historial.append(evento)

    # ------------------------------------------------------------------
    # Serialización y clonación
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "jugadores": [j.to_dict() for j in self.jugadores],
            "mesa": self.mesa.to_dict(),
            "ciega_pequena": self.ciega_pequena,
            "ciega_grande": self.ciega_grande,
            "dealer_idx": self.dealer_idx,
            "permitir_retiro": self.permitir_retiro,
            "registrar_historial": self.registrar_historial,
            "fase": self.fase,
            "turno_idx": self.turno_idx,
            "ya_actuo": list(self.ya_actuo),
            "ganadores": None if self.ganadores is None else list(self.ganadores),
            "historial": self.historial,
            "fichas_inicio_mano": list(self.fichas_inicio_mano),
        }

    @classmethod
    def from_dict(cls, data: dict, rng: np.random.Generator | None = None) -> "MotorPoker":
        motor = cls(
            Jugador.from_dict(data["jugadores"][0]),
            Jugador.from_dict(data["jugadores"][1]),
            ciega_pequena=data["ciega_pequena"],
            ciega_grande=data["ciega_grande"],
            dealer_idx=data["dealer_idx"],
            rng=rng,
            permitir_retiro=data.get("permitir_retiro", True),
            registrar_historial=data.get("registrar_historial", True),
        )
        motor.mesa = Mesa.from_dict(data["mesa"], motor.rng)
        motor.fase = data["fase"]
        motor.turno_idx = data["turno_idx"]
        motor.ya_actuo = list(data["ya_actuo"])
        motor.ganadores = data["ganadores"]
        motor.historial = data["historial"]
        motor.fichas_inicio_mano = list(data["fichas_inicio_mano"])
        return motor

    def guardar_json(self, ruta: str) -> None:
        with open(ruta, "w", encoding="utf-8") as archivo:
            json.dump(self.to_dict(), archivo, indent=2, ensure_ascii=False)

    @classmethod
    def cargar_json(cls, ruta: str, rng: np.random.Generator | None = None) -> "MotorPoker":
        with open(ruta, "r", encoding="utf-8") as archivo:
            return cls.from_dict(json.load(archivo), rng)

    def clonar(self) -> "MotorPoker":
        """Copia independiente del estado actual (sin pasar por JSON). El clon
        comparte el generador `rng`; el orden del mazo restante ya está fijo,
        así que dentro de la mano el clon es determinista."""
        nuevo = MotorPoker.__new__(MotorPoker)
        nuevo.jugadores = [j.clonar() for j in self.jugadores]
        nuevo.rng = self.rng
        nuevo.mesa = self.mesa.clonar()
        nuevo.ciega_pequena = self.ciega_pequena
        nuevo.ciega_grande = self.ciega_grande
        nuevo.dealer_idx = self.dealer_idx
        nuevo.permitir_retiro = self.permitir_retiro
        nuevo.registrar_historial = self.registrar_historial
        nuevo.fase = self.fase
        nuevo.turno_idx = self.turno_idx
        nuevo.ya_actuo = list(self.ya_actuo)
        nuevo.ganadores = None if self.ganadores is None else list(self.ganadores)
        nuevo.historial = list(self.historial)
        nuevo.fichas_inicio_mano = list(self.fichas_inicio_mano)
        nuevo._cache_turno = None
        return nuevo


class Juego(MotorPoker):
    """Juego(j1, j2) funciona exactamente igual que MotorPoker(j1, j2)."""
    pass