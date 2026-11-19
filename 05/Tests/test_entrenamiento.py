"""
Pruebas del entorno Gymnasium, la abstracción de estados, el agente Q,
el entrenamiento con checkpoints y las herramientas de evaluación.

Desde la raíz del repo:   python -m Tests.test_entrenamiento
"""
import csv
import json
import os
import tempfile

import numpy as np

from Clases.abstraccion import N_ESTADOS, estado_de_obs, decodificar_estado, fuerza_preflop, tiene_proyecto
from Clases.agente_q import AgenteQ
from Clases.cadenas import adyacencia, cadena_estricta_mas_larga, transitividad
from Clases.carta import id_de
from Clases.entorno import PokerHeadsUpEnv
from Clases.entrenamiento import entrenar_q
from Clases.evaluacion import evaluar_politica, evaluar_partidas, prob_racha
from Clases.motor import N_ACCIONES, TAM_OBS
from Clases.simulacion import PoliticaAleatoria


def c(n, p):
    return id_de(n, p)


def jugar_mano_aleatoria(env, rng):
    obs, info = env.reset()
    total = 0.0
    if info["mano_terminada"]:
        return info["recompensa"]
    while True:
        accion = int(rng.choice(np.flatnonzero(info["mascara"])))
        obs, r, terminado, trunc, info = env.step(accion)
        assert obs.shape == (TAM_OBS,) and obs.dtype == np.float32
        total += r
        if terminado:
            return total


# ---------- abstracción ----------

def test_fuerza_preflop_ordena_manos():
    aa = fuerza_preflop(c(14, "Pica"), c(14, "Trébol"))
    ak_suited = fuerza_preflop(c(14, "Pica"), c(13, "Pica"))
    siete_dos = fuerza_preflop(c(7, "Pica"), c(2, "Trébol"))
    assert aa > ak_suited > siete_dos
    assert aa == 7 and siete_dos == 0


def test_proyectos():
    mano = [c(14, "Pica"), c(9, "Pica")]
    assert tiene_proyecto(mano, [c(2, "Pica"), c(5, "Pica"), c(11, "Trébol")], 1)   # proyecto de color
    assert not tiene_proyecto(mano, [c(2, "Pica"), c(5, "Pica"), c(11, "Pica"), c(4, "Pica"), c(7, "Trébol")], 6)
    assert tiene_proyecto([c(9, "Pica"), c(8, "Trébol")], [c(7, "Diamante"), c(6, "Pica"), c(2, "Corazón")], 1)  # escalera abierta


def test_estados_en_rango_y_decodificables():
    env = PokerHeadsUpEnv()
    rng = np.random.default_rng(0)
    vistos = set()
    for _ in range(300):
        obs, info = env.reset()
        if info["mano_terminada"]:
            continue
        while True:
            e = estado_de_obs(obs)
            assert 0 <= e < N_ESTADOS
            vistos.add(e)
            decodificar_estado(e)
            obs, r, terminado, _, info = env.step(int(rng.choice(np.flatnonzero(info["mascara"]))))
            if terminado:
                break
    assert len(vistos) > 50


# ---------- entorno ----------

def test_env_reset_step_basico():
    env = PokerHeadsUpEnv()
    obs, info = env.reset(seed=1)
    assert env.observation_space.contains(obs)
    assert info["mascara"].shape == (N_ACCIONES,) and info["mascara"].dtype == bool
    obs, r, terminado, truncado, info = env.step(1)
    assert isinstance(r, float) and truncado is False


def test_env_reproducible_con_semilla():
    def corrida():
        env = PokerHeadsUpEnv()
        obs, _ = env.reset(seed=7)
        rng = np.random.default_rng(1)
        rs = [jugar_mano_aleatoria(env, rng) for _ in range(30)]
        return obs.tolist(), rs
    assert corrida() == corrida()


def test_env_alterna_dealer():
    env = PokerHeadsUpEnv()
    posiciones = []
    env.reset(seed=3)
    posiciones.append(env.motor.dealer_idx)
    for _ in range(5):
        env.reset()
        posiciones.append(env.motor.dealer_idx)
    assert posiciones == [0, 1, 0, 1, 0, 1]


def test_env_conserva_fichas_y_recompensa_cero_suma():
    env = PokerHeadsUpEnv()
    env.reset(seed=11)
    rng = np.random.default_rng(5)
    for _ in range(300):
        jugar_mano_aleatoria(env, rng)
        total = env.jugadores[0].fichas + env.jugadores[1].fichas
        assert total == 2 * env.fichas_iniciales
        assert env.motor.recompensa(0) == -env.motor.recompensa(1)


def test_env_accion_ilegal():
    env = PokerHeadsUpEnv(modo_ilegal="error")
    env.reset(seed=1)
    try:
        env.step(99)
        assert False
    except ValueError:
        pass
    env2 = PokerHeadsUpEnv(modo_ilegal="igualar")
    env2.reset(seed=1)
    _, _, _, _, info = env2.step(99)
    assert info["accion_ilegal"] is True


def test_env_rival_retira_antes_de_actuar_el_agente():
    # Rival que siempre se retira cuando puede: si el agente es BB, la mano acaba en reset()
    def siempre_retira(obs, mascara):
        return 0 if mascara[0] else 1
    siempre_retira.usa_obs = False
    env = PokerHeadsUpEnv(oponente=siempre_retira)
    env.reset(seed=1)                    # primera mano: el agente es dealer y actúa primero
    env.step(1)
    obs, info = env.reset()              # segunda mano: el rival es dealer y se retira de entrada
    assert info["mano_terminada"] is True and info["recompensa"] == 0.5  # gana la ciega chica (10/20)
    obs, r, terminado, _, info2 = env.step(0)
    assert terminado and r == 0.5 and info2["gano"] is True


def test_env_checker_oficial_si_gymnasium_esta_instalado():
    try:
        from gymnasium.utils.env_checker import check_env
    except ImportError:
        return  # entorno de pruebas sin gymnasium real
    check_env(PokerHeadsUpEnv(), skip_render_check=True)


# ---------- agente ----------

def test_agente_guardar_cargar():
    a = AgenteQ("x", alpha=0.2, rng=np.random.default_rng(0))
    a.Q[5, 2] = 1.5
    a.visitas[5, 2] = 3
    a.episodios = 42
    ruta = os.path.join(tempfile.gettempdir(), "_agente_test.npz")
    a.guardar(ruta)
    try:
        b = AgenteQ.cargar(ruta)
    finally:
        os.remove(ruta)
    assert b.episodios == 42 and b.alpha == 0.2 and b.Q[5, 2] == 1.5 and b.visitas[5, 2] == 3


def test_agente_actualiza_y_respeta_mascara():
    a = AgenteQ(rng=np.random.default_rng(0), alpha=0.5)
    a.actualizar(10, 3, 2.0, None, None, True)
    assert a.Q[10, 3] == 1.0
    a.Q[20] = [5, 1, 1, 1, 9]
    mascara = np.array([True, True, True, True, False])
    assert a._mejor_accion(20, mascara) == 0   # ignora la acción 4 (ilegal) aunque valga más


def test_agente_aprende_bandit_trivial():
    pol = PoliticaAleatoria(np.random.default_rng(0))
    env = PokerHeadsUpEnv(oponente=PoliticaAleatoria(np.random.default_rng(1)))
    with tempfile.TemporaryDirectory() as tmp:
        agente = AgenteQ("t", alpha=0.1, rng=np.random.default_rng(0))
        entrenar_q(agente, env, 3000, tmp, {"random": pol}, eval_cada=1500, eval_manos=300,
                   eval_partidas=20, checkpoint_cada=1500, imprimir=lambda *a: None)
        assert agente.episodios == 3000
        assert agente.estados_visitados() > 20


# ---------- entrenamiento: checkpoints y métricas ----------

def test_entrenamiento_escribe_checkpoints_y_csv_y_reanuda():
    with tempfile.TemporaryDirectory() as tmp:
        env = PokerHeadsUpEnv(oponente=PoliticaAleatoria(np.random.default_rng(1)))
        rival_eval = PoliticaAleatoria(np.random.default_rng(2))
        agente = AgenteQ("t", rng=np.random.default_rng(0))
        kw = dict(eval_cada=1000, eval_manos=200, eval_partidas=10, checkpoint_cada=2000, imprimir=lambda *a: None)
        entrenar_q(agente, env, 4000, tmp, {"random": rival_eval}, **kw)

        archivos = set(os.listdir(tmp))
        assert {"metricas.csv", "ultimo.npz", "mejor.npz", "config.json"} <= archivos
        assert {"checkpoint_ep000002000.npz", "checkpoint_ep000004000.npz"} <= archivos
        with open(os.path.join(tmp, "metricas.csv"), newline="") as f:
            filas = list(csv.DictReader(f))
        assert len(filas) == 4 and float(filas[-1]["episodio"]) == 4000
        assert "winrate_partidas_random" in filas[0] and "bb_por_mano_random" in filas[0]

        # reanudar: continúa el contador de episodios y agrega filas al mismo CSV
        reanudado = AgenteQ.cargar(os.path.join(tmp, "ultimo.npz"), np.random.default_rng(1))
        assert reanudado.episodios == 4000
        entrenar_q(reanudado, env, 2000, tmp, {"random": rival_eval}, **kw)
        with open(os.path.join(tmp, "metricas.csv"), newline="") as f:
            assert len(list(csv.DictReader(f))) == 6
        with open(os.path.join(tmp, "config.json")) as f:
            assert json.load(f)["ep_inicio"] == 4000


# ---------- evaluación ----------

def test_evaluar_politica_es_reproducible():
    a = evaluar_politica(AgenteQ(rng=np.random.default_rng(3)), PoliticaAleatoria(np.random.default_rng(1)), 300, semilla=5)
    b = evaluar_politica(AgenteQ(rng=np.random.default_rng(3)), PoliticaAleatoria(np.random.default_rng(99)), 300, semilla=5)
    assert a["winrate"] == b["winrate"] and a["bb_por_mano"] == b["bb_por_mano"]
    assert a["ganadas"] + a["perdidas"] + a["empates"] == 300


def test_evaluar_partidas_devuelve_probabilidad():
    r = evaluar_partidas(PoliticaAleatoria(), PoliticaAleatoria(), n_partidas=40, semilla=1)
    assert 0 <= r["winrate_partidas"] <= 1 and r["manos_promedio"] > 0


def test_prob_racha_valores_conocidos():
    assert abs(prob_racha(0.5, 3) - 0.5) < 1e-9
    assert abs(prob_racha(0.6, 3) - 0.7287) < 1e-3
    assert abs(prob_racha(0.7, 3) - 0.8897) < 1e-3
    assert prob_racha(1.0, 3) == 1.0
    assert abs(prob_racha(0.3, 3) + prob_racha(0.7, 3) - 1.0) < 1e-9   # simetría


def _matriz(n, pares):
    """P[i,j] = prob. de que i le gane a j; pares = {(i, j): p} (se rellena P[j,i] = 1-p)."""
    P = np.full((n, n), 0.5)
    for (i, j), p in pares.items():
        P[i, j], P[j, i] = p, 1 - p
    return P


def test_cadena_estricta_mas_larga():
    # cadena estricta: cada nodo le gana a TODOS los posteriores (no solo al siguiente)
    todos = {(i, j): 0.95 for i in range(4) for j in range(i + 1, 4)}
    assert cadena_estricta_mas_larga(adyacencia(_matriz(4, todos), 0.9)) == [0, 1, 2, 3]

    # si 3 ya no pierde contra 0, 1 ni 2, la cadena se corta en [0, 1, 2]
    parcial = dict(todos)
    parcial.update({(2, 3): 0.60, (1, 3): 0.60, (0, 3): 0.60})
    assert cadena_estricta_mas_larga(adyacencia(_matriz(4, parcial), 0.9)) == [0, 1, 2]

    # la transitividad se EXIGE: 0>1 y 1>2 pero 0 no vence a 2 -> no hay cadena de 3
    no_trans = {(0, 1): 0.95, (1, 2): 0.95, (0, 2): 0.50}
    cadena = cadena_estricta_mas_larga(adyacencia(_matriz(3, no_trans), 0.9))
    assert len(cadena) == 2


def test_transitividad():
    todos = {(i, j): 0.95 for i in range(3) for j in range(i + 1, 3)}
    assert transitividad(adyacencia(_matriz(3, todos), 0.9))["frac_transitiva"] == 1.0
    ciclo = {(0, 1): 0.95, (1, 2): 0.95, (2, 0): 0.95}
    assert transitividad(adyacencia(_matriz(3, ciclo), 0.9))["frac_transitiva"] == 0.0


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


if __name__ == "__main__":
    main()