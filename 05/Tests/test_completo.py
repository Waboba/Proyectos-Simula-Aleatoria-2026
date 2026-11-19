"""
Pruebas de punta a punta para el proyecto de Poker 1v1.

Corre esto (desde la carpeta que contiene Clases/) con:
    python3 test_completo.py
"""

import numpy as np
import os
import tempfile

from Clases.carta import Carta, id_de, nombre_de
from Clases.jugador import Jugador
from Clases.mazo import Mazo
from Clases.evaluadorPoker import EvaluadorPoker, evaluar, decodificar
from Clases.motor import (
    MotorPoker, Juego, N_ACCIONES, TAM_OBS,
    RETIRARSE, IGUALAR_PASAR, SUBIR_MEDIO_POZO, SUBIR_POZO, ALL_IN,
)
from Clases.simulacion import Jugar, JugarSolo, Sesion, PoliticaAleatoria


def c(numero, pinta):
    """Atajo: c(14, 'Pica') = As de Pica como id 0-51."""
    return id_de(numero, pinta)


# ----------------------------- Cartas y mazo -----------------------------

def test_ids_de_cartas():
    assert sorted(Mazo().orden.tolist()) == list(range(52))
    for i in range(52):
        carta = Carta.from_int(i)
        assert carta.to_int() == i
    assert nombre_de(c(14, "Pica")) == "As de Pica"
    assert nombre_de(c(7, "Corazón")) == "7 de Corazón"


def test_mazo_reproducible_con_semilla():
    m1, m2 = Mazo(np.random.default_rng(5)), Mazo(np.random.default_rng(5))
    m1.mezclar(); m2.mezclar()
    assert [m1.entregar_carta() for _ in range(52)] == [m2.entregar_carta() for _ in range(52)]


# ----------------------------- Evaluador -----------------------------

def test_desempate_evaluador():
    mano_reyes = [c(13, "Pica"), c(13, "Trébol"), c(2, "Diamante"), c(5, "Corazón"), c(9, "Pica")]
    mano_ases = [c(14, "Pica"), c(14, "Trébol"), c(2, "Diamante"), c(5, "Corazón"), c(9, "Pica")]
    e1, e2 = EvaluadorPoker(mano_reyes), EvaluadorPoker(mano_ases)
    assert e1.obtener_jugada() == "Par" and e2.obtener_jugada() == "Par"
    assert e2.obtener_rango_completo() > e1.obtener_rango_completo(), \
        "El par de Ases debería ganarle al par de Reyes"


def test_escalera_rueda():
    rueda = [c(14, "Pica"), c(2, "Trébol"), c(3, "Diamante"), c(4, "Corazón"), c(5, "Pica")]
    e = EvaluadorPoker(rueda)
    assert e.obtener_jugada() == "Escalera"
    assert decodificar(e.obtener_rango_completo())[1] == (5,), "A-2-3-4-5 vale '5 alta'"
    seis_alta = [c(2, "Pica"), c(3, "Trébol"), c(4, "Diamante"), c(5, "Corazón"), c(6, "Pica")]
    assert evaluar(seis_alta) > evaluar(rueda)


def test_mejor_mano_de_7():
    siete = [c(13, "Pica"), c(13, "Trébol"), c(13, "Diamante"), c(14, "Corazón"),
             c(14, "Pica"), c(2, "Trébol"), c(4, "Diamante")]
    assert EvaluadorPoker(siete).obtener_jugada() == "Full House"


def test_categorias_basicas():
    casos = {
        "Escalera de Color": [c(9, "Pica"), c(10, "Pica"), c(11, "Pica"), c(12, "Pica"), c(13, "Pica"), c(2, "Trébol"), c(3, "Trébol")],
        "Póker": [c(9, "Pica"), c(9, "Trébol"), c(9, "Diamante"), c(9, "Corazón"), c(13, "Pica"), c(2, "Trébol"), c(3, "Trébol")],
        "Color": [c(2, "Pica"), c(5, "Pica"), c(7, "Pica"), c(9, "Pica"), c(13, "Pica"), c(2, "Trébol"), c(3, "Trébol")],
        "Trío": [c(9, "Pica"), c(9, "Trébol"), c(9, "Diamante"), c(4, "Corazón"), c(13, "Pica"), c(2, "Trébol"), c(7, "Trébol")],
        "Doble Par": [c(9, "Pica"), c(9, "Trébol"), c(4, "Diamante"), c(4, "Corazón"), c(13, "Pica"), c(2, "Trébol"), c(7, "Trébol")],
        "Carta Alta": [c(9, "Pica"), c(11, "Trébol"), c(4, "Diamante"), c(6, "Corazón"), c(13, "Pica"), c(2, "Trébol"), c(7, "Trébol")],
    }
    for esperado, cartas in casos.items():
        assert EvaluadorPoker(cartas).obtener_jugada() == esperado, esperado


# ----------------------------- Reglas del motor -----------------------------

def test_motor_mano_completa_sin_apuestas():
    j1, j2 = Jugador("Ana", 1000), Jugador("Beto", 1000)
    motor = MotorPoker(j1, j2, dealer_idx=0, rng=np.random.default_rng(1))
    motor.iniciar_mano()

    for _ in range(4):
        motor.aplicar_accion("igualar" if motor.acciones_legales()["a_igualar"] > 0 else "pasar")
        if not motor.esta_terminada():
            motor.aplicar_accion("pasar")

    assert motor.esta_terminada()
    assert len(motor.mesa.cartas_comunitarias) == 5
    assert j1.fichas + j2.fichas == 2000, "Las fichas deben conservarse"


def test_no_se_puede_retirar_si_puede_pasar_gratis():
    j1, j2 = Jugador("Ana", 1000), Jugador("Beto", 1000)
    motor = MotorPoker(j1, j2, rng=np.random.default_rng(2))
    motor.iniciar_mano()
    motor.aplicar_accion("igualar")  # Ana (ciega chica) iguala
    info = motor.acciones_legales()  # Beto puede pasar gratis
    assert info["a_igualar"] == 0
    assert "retirarse" not in info["acciones"]
    assert not motor.mascara_acciones()[RETIRARSE]


def test_permitir_retiro_false_quita_la_opcion():
    j1, j2 = Jugador("Ana", 1000), Jugador("Beto", 1000)
    motor = MotorPoker(j1, j2, permitir_retiro=False, rng=np.random.default_rng(3))
    motor.iniciar_mano()
    assert "retirarse" not in motor.acciones_legales()["acciones"]  # Ana enfrenta la ciega grande
    motor.aplicar_accion("subir", monto=100)
    assert "retirarse" not in motor.acciones_legales()["acciones"]
    assert not motor.mascara_acciones()[RETIRARSE]


def test_motor_igualar_con_fichas_insuficientes():
    j1, j2 = Jugador("Rico", 1000), Jugador("Pobre", 35)
    motor = MotorPoker(j1, j2, ciega_pequena=10, ciega_grande=20, dealer_idx=0, rng=np.random.default_rng(4))
    total_inicial = j1.fichas + j2.fichas

    motor.iniciar_mano()
    motor.aplicar_accion("subir", monto=300)

    info = motor.acciones_legales()
    assert "subir" not in info["acciones"]
    assert "igualar" in info["acciones"]

    motor.aplicar_accion("igualar")
    assert motor.esta_terminada()
    assert len(motor.mesa.cartas_comunitarias) == 5
    assert j1.fichas + j2.fichas == total_inicial
    # La parte no igualada (285) se devolvió: Rico solo arriesgaba 35.
    assert j1.fichas in (965, 1000, 1035), f"Rico quedó con {j1.fichas}"  # pierde, empata o gana 35


def test_rival_puede_retirarse_ante_all_in():
    j1, j2 = Jugador("Ana", 1000), Jugador("Beto", 1000)
    motor = MotorPoker(j1, j2, dealer_idx=0, rng=np.random.default_rng(5))
    motor.iniciar_mano()
    motor.aplicar_accion_id(ALL_IN)  # Ana se va all-in

    assert not motor.esta_terminada(), "Beto debe poder decidir antes de que se cierre la ronda"
    info = motor.acciones_legales()
    assert info["jugador"] == "Beto"
    assert info["acciones"] == ["igualar", "retirarse"], "No se puede subir contra un rival sin fichas"

    motor.aplicar_accion("retirarse")
    assert motor.ganadores == ["Ana"]
    assert j1.fichas + j2.fichas == 2000
    assert j1.fichas == 1020 and j2.fichas == 980  # Beto solo perdió su ciega grande


def test_ciega_all_in_deja_decidir_al_rival():
    # La ciega grande (Pobre, 15 fichas) queda all-in; Rico (ciega chica) debe poder decidir.
    j1, j2 = Jugador("Rico", 1000), Jugador("Pobre", 15)
    motor = MotorPoker(j1, j2, dealer_idx=0, rng=np.random.default_rng(6))
    motor.iniciar_mano()
    assert not motor.esta_terminada()
    assert motor.acciones_legales()["jugador"] == "Rico"
    motor.aplicar_accion("igualar")
    assert motor.esta_terminada()
    assert j1.fichas + j2.fichas == 1015


def test_acciones_ilegales_lanzan_error():
    j1, j2 = Jugador("Ana", 1000), Jugador("Beto", 1000)
    motor = MotorPoker(j1, j2, rng=np.random.default_rng(7))
    motor.iniciar_mano()
    for accion in ("pasar",):  # Ana enfrenta la ciega grande: no puede pasar
        try:
            motor.aplicar_accion(accion)
            assert False, "debía fallar"
        except ValueError:
            pass
    try:
        motor.aplicar_accion_id(SUBIR_MEDIO_POZO + 10)
        assert False
    except ValueError:
        pass


def test_motor_fold_no_pierde_fichas():
    j1, j2 = Jugador("Ana", 1000), Jugador("Beto", 1000)
    motor = MotorPoker(j1, j2, ciega_pequena=10, ciega_grande=20, dealer_idx=0, rng=np.random.default_rng(8))
    motor.iniciar_mano()
    motor.aplicar_accion("subir", monto=100)
    motor.aplicar_accion("retirarse")

    assert motor.esta_terminada()
    assert motor.ganadores == ["Ana"]
    assert j1.fichas + j2.fichas == 2000
    assert motor.recompensa(0) == 20 and motor.recompensa(1) == -20


def test_guardar_cargar_y_bifurcar():
    j1, j2 = Jugador("Ana", 1000), Jugador("Beto", 1000)
    motor = MotorPoker(j1, j2, dealer_idx=0, rng=np.random.default_rng(9))
    motor.iniciar_mano()
    motor.aplicar_accion("igualar")
    motor.aplicar_accion("pasar")  # ahora estamos en el FLOP

    ruta = os.path.join(tempfile.gettempdir(), "_test_mano.json")
    motor.guardar_json(ruta)

    try:
        rama_a = MotorPoker.cargar_json(ruta)
        rama_a.aplicar_accion("pasar")
        rama_a.aplicar_accion("pasar")

        rama_b = MotorPoker.cargar_json(ruta)
        rama_b.aplicar_accion("subir", monto=200)
    finally:
        if os.path.exists(ruta):
            os.remove(ruta)

    assert rama_a.jugadores[1].fichas != rama_b.jugadores[1].fichas
    assert motor.fase == "FLOP"


def test_clonar_es_independiente_e_identico():
    j1, j2 = Jugador("Ana", 1000), Jugador("Beto", 1000)
    motor = MotorPoker(j1, j2, rng=np.random.default_rng(10))
    motor.iniciar_mano()
    motor.aplicar_accion("igualar")
    clon = motor.clonar()
    assert clon.to_dict() == motor.to_dict()
    clon.aplicar_accion("pasar")
    assert motor.fase == "PREFLOP" and clon.fase == "FLOP"
    assert motor.mesa.cartas_comunitarias == [] and len(clon.mesa.cartas_comunitarias) == 3


def test_clase_juego_es_motor_poker():
    juego = Juego(Jugador("Ana", 1000), Jugador("Beto", 1000), dealer_idx=0, rng=np.random.default_rng(11))
    assert isinstance(juego, MotorPoker)
    juego.iniciar_mano()
    assert juego.fase == "PREFLOP"


# ----------------------------- Interfaz para RL -----------------------------

def test_mascara_y_observacion():
    j1, j2 = Jugador("Ana", 1000), Jugador("Beto", 1000)
    motor = MotorPoker(j1, j2, rng=np.random.default_rng(12))
    motor.iniciar_mano()
    mascara = motor.mascara_acciones()
    assert mascara.shape == (N_ACCIONES,) and mascara.dtype == bool
    assert mascara[IGUALAR_PASAR] and mascara[RETIRARSE] and mascara[ALL_IN]
    obs = motor.observacion()
    assert obs.shape == (TAM_OBS,) and obs.dtype == np.float32
    assert list(obs[:2]) == list(motor.jugadores[0].mano) and (obs[2:7] == -1).all()
    assert obs[12] == 1.0  # Ana es dealer
    assert motor.observacion(1)[12] == 0.0
    # los montos discretos son crecientes y no superan las fichas
    motor.aplicar_accion_id(SUBIR_POZO) if mascara[SUBIR_POZO] else motor.aplicar_accion_id(ALL_IN)
    assert j1.fichas + j1.apuesta_actual + j2.fichas + j2.apuesta_actual == 2000


# ----------------------------- Simulación -----------------------------

def test_reproducibilidad_con_semilla():
    def correr(semilla):
        rng = np.random.default_rng(semilla)
        p = PoliticaAleatoria(rng)
        return Jugar(p, p, nombre_j1="A", nombre_j2="B", rng=rng, max_manos=100).simular()
    r1, r2, r3 = correr(42), correr(42), correr(43)
    assert r1["fichas_finales"] == r2["fichas_finales"] and r1["manos_jugadas"] == r2["manos_jugadas"]
    assert r1["fichas_finales"] != r3["fichas_finales"] or r1["manos_jugadas"] != r3["manos_jugadas"]


def test_jugar_dos_politicas_conserva_fichas():
    rng = np.random.default_rng(123)
    p = PoliticaAleatoria(rng)
    resultado = Jugar(p, p, nombre_j1="A", nombre_j2="B", rng=rng, max_manos=300).simular()
    assert sum(resultado["fichas_finales"].values()) == 2000
    assert resultado["manos_jugadas"] > 0


def test_jugar_solo_vs_azar():
    rng = np.random.default_rng(456)
    solo = JugarSolo(PoliticaAleatoria(rng), nombre_j1="Humano", rng=rng, max_manos=300)
    resultado = solo.simular()
    assert "Azar" in resultado["fichas_finales"]
    assert sum(resultado["fichas_finales"].values()) == 2000


def test_sesion_n_partidas():
    solo = JugarSolo(PoliticaAleatoria(np.random.default_rng(1)), nombre_j1="Humano", semilla=789, max_manos=300)
    stats = Sesion(15, solo).ejecutar()
    assert stats["partidas_jugadas"] == 15
    assert sum(stats["victorias"].values()) == 15


def test_historial_opcional():
    rng = np.random.default_rng(3)
    p = PoliticaAleatoria(rng)
    con = Jugar(p, p, rng=rng, max_manos=5, registrar_historial=True).simular()
    sin = Jugar(p, p, rng=rng, max_manos=5, registrar_historial=False).simular()
    assert len(con["historial"]) == 5 and sin["historial"] == []


def test_sin_retiro_en_politica_aleatoria():
    rng = np.random.default_rng(0)
    pol = PoliticaAleatoria(rng, peso_retiro=0.0)
    mascara = np.ones(N_ACCIONES, dtype=bool)
    dummy_obs = np.zeros(TAM_OBS, dtype=np.float32)

    elegidas = {pol(dummy_obs, mascara) for _ in range(2000)}
    assert RETIRARSE not in elegidas and elegidas == {IGUALAR_PASAR, SUBIR_MEDIO_POZO, SUBIR_POZO, ALL_IN}


def test_muchas_partidas_aleatorias_sin_fugas_de_fichas():
    """100 partidas con políticas 100% aleatorias (con y sin retiro)."""
    for semilla in range(100):
        rng = np.random.default_rng(semilla)
        p = PoliticaAleatoria(rng, peso_retiro=0.5 if semilla % 2 else None)
        resultado = Jugar(p, p, nombre_j1="A", nombre_j2="B", rng=rng, max_manos=300,
                          permitir_retiro=(semilla % 5 != 0)).simular()
        suma = sum(resultado["fichas_finales"].values())
        assert suma == 2000, f"Fuga de fichas con seed={semilla}: suma={suma}"


# ----------------------------------------------------------------------

def main():
    pruebas = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    fallos = 0
    for prueba in pruebas:
        nombre = prueba.__name__
        try:
            prueba()
            print(f"OK   - {nombre}")
        except AssertionError as error:
            fallos += 1
            print(f"FALLO - {nombre}: {error}")
        except Exception as error:
            fallos += 1
            print(f"ERROR - {nombre}: {type(error).__name__}: {error}")

    print()
    if fallos == 0:
        print(f"Las {len(pruebas)} pruebas pasaron correctamente.")
    else:
        print(f"{fallos} de {len(pruebas)} pruebas fallaron.")


if __name__ == "__main__":
    main()