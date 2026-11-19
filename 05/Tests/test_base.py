"""
Pruebas de los componentes base (motor, evaluador, abstracción, agente, entorno, evaluación).

Desde la raíz del proyecto:   python -m Tests.test_base
"""
import os
import sys
import tempfile

import numpy as np

from Clases.abstraccion import N_ESTADOS, decodificar_estado, estado_de_obs
from Clases.agente_q import AgenteQ
from Clases.carta import id_de
from Clases.entorno import PokerHeadsUpEnv
from Clases.entrenamiento import jugar_mano_entrenamiento
from Clases.evaluacion import evaluar_partidas, evaluar_politica, prob_racha, simular_rachas
from Clases.evaluadorPoker import categoria, evaluar
from Clases.jugador import Jugador
from Clases.mazo import Mazo
from Clases.motor import ALL_IN, IGUALAR_PASAR, MotorPoker, N_ACCIONES, RETIRARSE
from Clases.simulacion import Jugar, PoliticaAleatoria, PoliticaMezcla


def c(numero, pinta):
    return id_de(numero, pinta)


# ---------- evaluador ----------

def test_evaluador_orden_de_categorias():
    escalera_color = [c(10, 0), c(11, 0), c(12, 0), c(13, 0), c(14, 0), c(2, 1), c(3, 2)]
    poker = [c(9, 0), c(9, 1), c(9, 2), c(9, 3), c(2, 0), c(5, 1), c(7, 2)]
    full = [c(9, 0), c(9, 1), c(9, 2), c(4, 3), c(4, 0), c(5, 1), c(7, 2)]
    color = [c(2, 0), c(5, 0), c(7, 0), c(9, 0), c(12, 0), c(3, 1), c(4, 2)]
    escalera = [c(5, 0), c(6, 1), c(7, 2), c(8, 3), c(9, 0), c(2, 1), c(2, 2)]
    trio = [c(9, 0), c(9, 1), c(9, 2), c(4, 3), c(6, 0), c(11, 1), c(13, 2)]
    doble = [c(9, 0), c(9, 1), c(4, 2), c(4, 3), c(6, 0), c(11, 1), c(13, 2)]
    par = [c(9, 0), c(9, 1), c(3, 2), c(4, 3), c(6, 0), c(11, 1), c(13, 2)]
    alta = [c(2, 0), c(9, 1), c(3, 2), c(5, 3), c(7, 0), c(11, 1), c(13, 2)]
    manos = [escalera_color, poker, full, color, escalera, trio, doble, par, alta]
    rangos = [evaluar(m) for m in manos]
    assert [categoria(r) for r in rangos] == [9, 8, 7, 6, 5, 4, 3, 2, 1]
    assert rangos == sorted(rangos, reverse=True)


def test_evaluador_rueda_desempate_y_kicker():
    rueda = evaluar([c(14, 0), c(2, 1), c(3, 2), c(4, 3), c(5, 0), c(9, 1), c(11, 2)])
    seis = evaluar([c(2, 1), c(3, 2), c(4, 3), c(5, 0), c(6, 0), c(9, 1), c(11, 2)])
    assert categoria(rueda) == 5 and seis > rueda          # la rueda vale "5 alta"
    par_k_alto = evaluar([c(9, 0), c(9, 1), c(14, 2), c(4, 3), c(6, 0), c(2, 1), c(3, 2)])
    par_k_bajo = evaluar([c(9, 2), c(9, 3), c(13, 2), c(4, 1), c(6, 1), c(2, 0), c(3, 0)])
    assert par_k_alto > par_k_bajo                          # mismo par, mejor kicker gana


# ---------- mazo ----------

def test_mazo_reparte_52_cartas_distintas_y_es_reproducible():
    m1, m2 = Mazo(np.random.default_rng(3)), Mazo(np.random.default_rng(3))
    m1.mezclar()
    m2.mezclar()
    a = [m1.entregar_carta() for _ in range(52)]
    b = [m2.entregar_carta() for _ in range(52)]
    assert sorted(a) == list(range(52)) and a == b


# ---------- motor ----------

def _jugar_mano_aleatoria(motor, rng):
    motor.iniciar_mano()
    n = 0
    while not motor.esta_terminada():
        mascara = motor.mascara_acciones()
        assert mascara[IGUALAR_PASAR], "igualar/pasar debe ser siempre legal para quien tiene el turno"
        assert not (mascara[ALL_IN] and motor.jugadores[1 - motor.turno_idx].fichas == 0)
        motor.aplicar_accion_id(int(rng.choice(np.flatnonzero(mascara))))
        n += 1
        assert n < 200, "la mano no termina"


def test_motor_conserva_fichas_en_miles_de_manos():
    rng = np.random.default_rng(0)
    for fichas in (1000, 35):                                # 35 < 2 ciegas grandes: casos de all-in forzado
        j1, j2 = Jugador("a", fichas), Jugador("b", fichas)
        motor = MotorPoker(j1, j2, rng=rng, registrar_historial=False)
        for i in range(1500):
            for j in (j1, j2):
                j.fichas = fichas
            motor.dealer_idx = i % 2
            _jugar_mano_aleatoria(motor, rng)
            assert j1.fichas + j2.fichas == 2 * fichas
            assert motor.recompensa(0) == -motor.recompensa(1)
            assert motor.mesa.pozo == 0


def test_partida_completa_conserva_fichas_y_termina():
    rng = np.random.default_rng(1)
    for _ in range(200):
        r = Jugar(PoliticaAleatoria(rng), PoliticaAleatoria(rng), max_manos=50, rng=rng).simular()
        assert sum(r["fichas_finales"].values()) == 2000


def test_retirarse_solo_si_hay_apuesta_que_igualar():
    j1, j2 = Jugador("a"), Jugador("b")
    m = MotorPoker(j1, j2, rng=np.random.default_rng(0), registrar_historial=False)
    m.iniciar_mano()
    for _ in range(40):
        if m.esta_terminada():
            break
        mascara = m.mascara_acciones()
        a_igualar = m.observacion()[11]
        assert bool(mascara[RETIRARSE]) == (a_igualar > 0)
        m.aplicar_accion_id(IGUALAR_PASAR)


# ---------- abstracción ----------

def test_abstraccion_estados_en_rango_y_decodificables():
    rng = np.random.default_rng(2)
    j1, j2 = Jugador("a"), Jugador("b")
    motor = MotorPoker(j1, j2, rng=rng, registrar_historial=False)
    vistos = set()
    for i in range(1500):
        for j in (j1, j2):
            j.fichas = 1000
        motor.dealer_idx = i % 2
        motor.iniciar_mano()
        while not motor.esta_terminada():
            s = estado_de_obs(motor.observacion())
            assert 0 <= s < N_ESTADOS
            d = decodificar_estado(s)
            assert d["fase"] in ("PREFLOP", "FLOP", "TURN", "RIVER")
            vistos.add(s)
            motor.aplicar_accion_id(int(rng.choice(np.flatnonzero(motor.mascara_acciones()))))
    assert len(vistos) > 150                                 # la abstracción distingue situaciones distintas


# ---------- agente ----------

def test_agente_actualiza_con_la_regla_de_q_learning():
    ag = AgenteQ(alpha=0.5, gamma=1.0, rng=np.random.default_rng(0))
    ag.Q[7] = [0, 2.0, 0, 0, 9.0]
    mascara = np.array([True, True, True, True, False])      # la acción 4 (valor 9) NO es legal
    ag.actualizar(3, 1, 1.0, 7, mascara, False)             # objetivo = 1 + max legal(2.0) = 3 -> Q = 0.5*3
    assert abs(ag.Q[3, 1] - 1.5) < 1e-6 and ag.visitas[3, 1] == 1
    ag.actualizar(3, 1, -1.0, None, None, True)             # terminal: objetivo = -1
    assert abs(ag.Q[3, 1] - 0.25) < 1e-6


def test_agente_solo_elige_acciones_legales():
    ag = AgenteQ(rng=np.random.default_rng(0))
    ag.Q[5] = [0, 0, 100.0, 0, 0]
    mascara = np.array([True, True, False, False, False])
    assert all(ag._mejor_accion(5, mascara) in (0, 1) for _ in range(50))


def test_agente_guardar_cargar_conserva_tabla_y_hash():
    ag = AgenteQ("x", rng=np.random.default_rng(0))
    ag.Q = np.random.default_rng(1).normal(size=ag.Q.shape).astype(np.float32)
    ag.episodios = 123
    with tempfile.TemporaryDirectory() as tmp:
        ruta = os.path.join(tmp, "a.npz")
        ag.guardar(ruta)
        otro = AgenteQ.cargar(ruta)
    assert otro.hash_q() == ag.hash_q() and otro.episodios == 123 and np.array_equal(otro.Q, ag.Q)
    otro.Q[0, 0] += 1e-3
    assert otro.hash_q() != ag.hash_q()


def test_vista_greedy_comparte_q_y_no_consume_el_rng_del_agente():
    ag = AgenteQ(rng=np.random.default_rng(0))
    estado_rng = ag.rng.bit_generator.state
    v = ag.vista_greedy(np.random.default_rng(9))
    assert v.Q is ag.Q and v.epsilon == 0.0
    env = PokerHeadsUpEnv(oponente=PoliticaAleatoria(np.random.default_rng(1)))
    obs, info = env.reset(seed=0)
    for _ in range(20):
        v(obs, info["mascara"])                             # la tabla en 0 obliga a desempatar con el rng
    assert ag.rng.bit_generator.state == estado_rng


# ---------- entorno ----------

def test_entorno_reproducible_con_semilla_y_fichas_conservadas():
    def jugar(seed):
        env = PokerHeadsUpEnv(oponente=PoliticaAleatoria(np.random.default_rng(4)))
        rng = np.random.default_rng(5)
        out = []
        obs, info = env.reset(seed=seed)
        for i in range(60):
            if i:
                obs, info = env.reset()
            if info["mano_terminada"]:
                out.append(info["recompensa"])
                continue
            while True:
                obs, r, term, _, info = env.step(int(rng.choice(np.flatnonzero(info["mascara"]))))
                if term:
                    out.append(r)
                    assert sum(j.fichas for j in env.jugadores) == 2000
                    break
        return out
    assert jugar(11) == jugar(11)


def test_jugar_mano_entrenamiento_aprende_y_devuelve_recompensa_finita():
    ag = AgenteQ(rng=np.random.default_rng(0))
    env = PokerHeadsUpEnv(oponente=PoliticaAleatoria.uniforme(np.random.default_rng(1)))
    env.reset(seed=0)
    for _ in range(300):
        ag.epsilon = 1.0
        r, n = jugar_mano_entrenamiento(ag, env)
        assert np.isfinite(r) and n >= 0
    assert ag.visitas.sum() > 300 and ag.estados_visitados() > 20 and np.any(ag.Q != 0)


def test_politica_uniforme_reparte_acciones_por_igual():
    p = PoliticaAleatoria.uniforme(np.random.default_rng(0))
    conteo = np.zeros(N_ACCIONES)
    for _ in range(5000):
        conteo[p(None, np.ones(N_ACCIONES, dtype=bool))] += 1
    assert np.all(np.abs(conteo / 5000 - 0.2) < 0.03)
    assert p(None, np.array([False, True, False, False, False])) == 1


# ---------- evaluación ----------

def test_evaluar_politica_es_determinista_con_la_misma_semilla():
    def corre():
        return evaluar_politica(PoliticaAleatoria(np.random.default_rng(1)),
                                PoliticaAleatoria(np.random.default_rng(2)), 800, semilla=7)
    a, b = corre(), corre()
    assert a["ganadas"] == b["ganadas"] and a["bb_por_mano"] == b["bb_por_mano"]
    assert a["ganadas"] + a["perdidas"] + a["empates"] == 800


def test_random_contra_random_es_justo():
    r = evaluar_politica(PoliticaAleatoria(np.random.default_rng(1)), PoliticaAleatoria(np.random.default_rng(2)), 8000)
    assert abs(r["winrate"] - 0.5) < 0.03
    assert abs(r["bb_por_mano"]) < 1.5 * r["bb_ic95"]       # dentro de ~3 desviaciones (el póker tiene mucha varianza)


def test_agente_entrenado_le_gana_al_uniforme():
    ag = AgenteQ(rng=np.random.default_rng(0))
    env = PokerHeadsUpEnv(oponente=PoliticaAleatoria.uniforme(np.random.default_rng(1)), modo_recompensa="victoria")
    env.reset(seed=0)
    for ep in range(15000):
        ag.epsilon = max(0.05, 1 - 0.95 * ep / 10000)
        jugar_mano_entrenamiento(ag, env)
    r = evaluar_politica(ag, PoliticaAleatoria.uniforme(np.random.default_rng(2)), 4000, semilla=3)
    assert r["winrate"] > 0.6, r["winrate"]
    p = evaluar_partidas(ag, PoliticaAleatoria.uniforme(np.random.default_rng(2)), 200, semilla=3)
    assert p["winrate_partidas"] > 0.45


# ---------- racha ----------

def test_prob_racha_propiedades():
    assert abs(prob_racha(0.5, 3) - 0.5) < 1e-12
    assert abs(prob_racha(0.6, 3) - 0.7287) < 1e-3 and abs(prob_racha(0.71, 3) - 0.9011) < 1e-3
    assert prob_racha(1.0, 3) == 1.0 and abs(prob_racha(0.0, 3)) < 1e-12
    assert abs(prob_racha(0.3, 3) + prob_racha(0.7, 3) - 1.0) < 1e-12       # simetría
    assert abs(prob_racha(0.37, 1) - 0.37) < 1e-12
    ps = np.linspace(0.05, 0.95, 19)
    vals = [prob_racha(p, 3) for p in ps]
    assert all(x < y for x, y in zip(vals, vals[1:]))                        # monótona
    assert prob_racha(0.6, 5) > prob_racha(0.6, 3) > prob_racha(0.6, 1)      # más manos seguidas amplifican la ventaja


def test_simular_rachas_coincide_con_la_formula_con_rivales_distintos():
    """Corroboración empírica: con un par DESIGUAL, la carrera simulada coincide con la fórmula."""
    a = PoliticaAleatoria(np.random.default_rng(1), pesos=(0.0, 1.0, 0.0, 0.0, 0.0))   # siempre iguala
    b = PoliticaAleatoria.uniforme(np.random.default_rng(2))
    p = evaluar_politica(a, PoliticaAleatoria.uniforme(np.random.default_rng(2)), 12000, semilla=1)["winrate"]
    sim = simular_rachas(a, b, k=3, n_carreras=2500, semilla=4)
    esperado = prob_racha(p, 3)
    assert abs(sim["prob_racha_simulada"] - esperado) < 3 * sim["ic95"] / 1.96 + 0.03, (sim, esperado, p)


# ----------------------------------------------------------------------

def main():
    pruebas = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    fallos = 0
    for prueba in pruebas:
        try:
            prueba()
            print(f"OK   - {prueba.__name__}")
        except AssertionError as error:
            fallos += 1
            print(f"FALLO - {prueba.__name__}: {error}")
        except Exception as error:
            fallos += 1
            print(f"ERROR - {prueba.__name__}: {type(error).__name__}: {error}")
    print()
    print(f"Las {len(pruebas)} pruebas pasaron correctamente." if fallos == 0 else f"{fallos} de {len(pruebas)} pruebas fallaron.")
    sys.exit(1 if fallos else 0)


if __name__ == "__main__":
    main()
