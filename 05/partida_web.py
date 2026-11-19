"""
Partida de un humano contra un bot (AgenteQ, PoliticaAleatoria o cualquier política
`f(obs, mascara) -> int`) sobre el MotorPoker. No depende de ninguna interfaz: la usa
app_web.py (Streamlit), pero sirve igual para una GUI de escritorio o una API.

  p = PartidaContraBot(politica_bot, nombre_bot="agente_05")
  p.acciones_humano()      -> botones legales [{"id", "etiqueta", "monto"}]
  p.actuar(id)             -> el humano juega una de las 5 acciones; el bot responde solo
  p.subir(monto)           -> subida personalizada (monto por ENCIMA de igualar)
  p.nueva_mano()           -> siguiente mano (alterna el dealer)
  p.vista()                -> todo lo que la interfaz puede mostrar

SEGURIDAD: vista() nunca incluye las cartas del bot salvo en un showdown real (ambos
jugadores siguen activos al terminar la mano). El historial que expone tampoco las contiene.

El humano es el jugador 0 y el bot el 1. Las fichas se arrastran entre manos hasta que
alguien quiebra (o se reinician cada mano con reiniciar_fichas=True, como en el entrenamiento).
"""
import numpy as np

from Clases.evaluadorPoker import NOMBRE_CATEGORIA, evaluar
from Clases.jugador import Jugador
from Clases.motor import IGUALAR_PASAR, MotorPoker, RETIRARSE, SUBIR_MEDIO_POZO, SUBIR_POZO, ALL_IN

HUMANO, BOT = 0, 1


class PartidaContraBot:
    def __init__(self, politica_bot, nombre_bot: str = "Bot", fichas_iniciales: int = 1000,
                 ciega_pequena: int = 10, ciega_grande: int = 20, reiniciar_fichas: bool = False,
                 semilla: int | None = None, nombre_humano: str = "Tú") -> None:
        self.politica = politica_bot
        self.fichas_iniciales = fichas_iniciales
        self.ciega_grande = ciega_grande
        self.reiniciar_fichas = reiniciar_fichas
        self.rng = np.random.default_rng(semilla)

        self.jugadores = [Jugador(nombre_humano, fichas_iniciales), Jugador(nombre_bot, fichas_iniciales)]
        self.dealer_idx = int(self.rng.integers(2))
        self.motor = MotorPoker(self.jugadores[0], self.jugadores[1], ciega_pequena=ciega_pequena,
                                ciega_grande=ciega_grande, dealer_idx=self.dealer_idx, rng=self.rng)
        self.manos = 0                 # manos terminadas
        self.netos: list[int] = []     # ganancia del humano en cada mano terminada (fichas)
        self._contada = True           # ¿la mano actual ya se anotó en netos?
        self._primera = True
        self.nueva_mano()

    # ------------------------------------------------------------------ flujo
    @property
    def partida_terminada(self) -> bool:
        """Solo en modo partida completa: alguien se quedó sin fichas al cerrar la mano."""
        return (not self.reiniciar_fichas) and self.motor.esta_terminada() \
            and any(j.fichas <= 0 for j in self.jugadores)

    def nueva_mano(self) -> None:
        if not self.motor.esta_terminada() and not self._primera:
            raise ValueError("La mano actual todavía no terminó.")
        if self.partida_terminada:
            raise ValueError("La partida terminó: crea una nueva.")
        if not self._primera:
            self.dealer_idx = 1 - self.dealer_idx
        self._primera = False
        if self.reiniciar_fichas:
            for j in self.jugadores:
                j.fichas = self.fichas_iniciales
        self.motor.dealer_idx = self.dealer_idx
        self.motor.iniciar_mano()
        self._contada = False
        self._bot_juega()
        self._cerrar_si_terminada()

    def _bot_juega(self) -> None:
        m = self.motor
        while not m.esta_terminada() and m.turno_idx == BOT:
            mascara = m.mascara_acciones()
            obs = m.observacion(BOT) if getattr(self.politica, "usa_obs", True) else None
            accion = int(self.politica(obs, mascara))
            if not (0 <= accion < len(mascara) and mascara[accion]):
                accion = IGUALAR_PASAR  # siempre legal cuando se tiene el turno (igual que el entorno)
            m.aplicar_accion_id(accion)

    def _cerrar_si_terminada(self) -> None:
        if self.motor.esta_terminada() and not self._contada:
            self.netos.append(int(self.motor.recompensa(HUMANO)))
            self.manos += 1
            self._contada = True

    def _verificar_turno(self) -> None:
        if self.motor.esta_terminada():
            raise ValueError("La mano terminó.")
        if self.motor.turno_idx != HUMANO:
            raise ValueError("No es tu turno.")

    def actuar(self, accion_id: int) -> None:
        """El humano juega una acción discreta (0 retirarse, 1 igualar/pasar, 2 subir 1/2 pozo,
        3 subir pozo, 4 all-in); luego responde el bot hasta que vuelva a tocarle al humano."""
        self._verificar_turno()
        self.motor.aplicar_accion_id(int(accion_id))
        self._bot_juega()
        self._cerrar_si_terminada()

    def subir(self, monto: int) -> None:
        """Subida personalizada: `monto` fichas por ENCIMA de igualar (se recorta al máximo posible)."""
        self._verificar_turno()
        self.motor.aplicar_accion("subir", int(monto))
        self._bot_juega()
        self._cerrar_si_terminada()

    # ------------------------------------------------------------- consultas
    def _a_igualar(self) -> int:
        h, b = self.jugadores
        return max(b.apuesta_actual - h.apuesta_actual, 0)

    def acciones_humano(self) -> list[dict]:
        """Acciones legales del humano con etiqueta y monto. [] si no es su turno."""
        m = self.motor
        if m.esta_terminada() or m.turno_idx != HUMANO:
            return []
        mascara, (medio, pozo, max_sub) = m._mascara_turno()
        a_igualar = self._a_igualar()
        h = self.jugadores[HUMANO]
        etiquetas = {
            RETIRARSE: ("Retirarse", 0),
            IGUALAR_PASAR: (f"Igualar {a_igualar}" if a_igualar > 0 else "Pasar", a_igualar),
            SUBIR_MEDIO_POZO: (f"Subir ½ pozo (+{medio})", medio),
            SUBIR_POZO: (f"Subir pozo (+{pozo})", pozo),
            ALL_IN: (f"All-in ({h.fichas})", max_sub),
        }
        return [{"id": i, "etiqueta": etiquetas[i][0], "monto": etiquetas[i][1]}
                for i in range(len(mascara)) if mascara[i]]

    def rango_subida(self) -> tuple[int, int] | None:
        """(mínimo, máximo) para la subida personalizada, o None si no se puede subir."""
        m = self.motor
        if m.esta_terminada() or m.turno_idx != HUMANO:
            return None
        h, b = self.jugadores
        max_sub = h.fichas - self._a_igualar()
        if h.fichas <= 0 or max_sub <= 0 or b.fichas <= 0:
            return None
        return min(self.ciega_grande, max_sub), max_sub

    def _historial_texto(self) -> list[str]:
        """Acciones de la mano en texto. Deliberadamente NO incluye cartas (evento 'iniciar_mano')."""
        lineas = []
        for e in self.motor.historial:
            if e.get("tipo") == "nueva_calle":
                lineas.append(f"— {e['fase']} —")
            elif "accion" in e:
                verbo = {"retirarse": "se retira", "pasar": "pasa", "igualar": "iguala",
                         "subir": f"sube +{e.get('monto', 0)}"}[e["accion"]]
                lineas.append(f"{e['fase']} · {e['jugador']} {verbo}")
        return lineas

    def vista(self) -> dict:
        m = self.motor
        h, b = self.jugadores
        terminada = m.esta_terminada()
        showdown = terminada and h.esta_activo and b.esta_activo
        mesa = list(m.mesa.cartas_comunitarias)

        jugada_h = jugada_b = None
        if showdown and len(mesa) >= 3:
            jugada_h = NOMBRE_CATEGORIA[evaluar(h.mano + mesa) >> 20]
            jugada_b = NOMBRE_CATEGORIA[evaluar(b.mano + mesa) >> 20]

        resultado = None
        if terminada:
            g = m.ganadores or []
            resultado = {"ganadores": g, "gana_humano": g == [h.nombre], "gana_bot": g == [b.nombre],
                         "empate": len(g) == 2, "neto_humano": int(m.recompensa(HUMANO)),
                         "por_retiro": not showdown}
        campeon = None
        if self.partida_terminada:
            campeon = b.nombre if h.fichas <= 0 else h.nombre

        total = sum(self.netos)
        return {
            "fase": m.fase,
            "mesa": mesa,
            "pozo": m.mesa.pozo + h.apuesta_actual + b.apuesta_actual,
            "mano_humano": list(h.mano),
            "mano_bot": list(b.mano) if showdown else None,   # <-- solo en showdown
            "jugada_humano": jugada_h, "jugada_bot": jugada_b,
            "fichas": [h.fichas, b.fichas],
            "apuestas": [h.apuesta_actual, b.apuesta_actual],
            "dealer_idx": m.dealer_idx,
            "turno_humano": (not terminada) and m.turno_idx == HUMANO,
            "terminada": terminada,
            "resultado": resultado,
            "partida_terminada": self.partida_terminada,
            "campeon": campeon,
            "nombres": [h.nombre, b.nombre],
            "historial": self._historial_texto(),
            "manos": self.manos,
            "neto_total": total,
            "bb_por_mano": (total / self.ciega_grande / self.manos) if self.manos else 0.0,
        }
