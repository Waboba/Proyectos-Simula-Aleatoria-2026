"""
Pruebas de la liga de agentes (rival mezcla, cadenas estrictas, entrenamiento por etapas,
reproducibilidad, reanudación y verificación).

Desde la raíz del proyecto:   python -m Tests.test_liga
"""
import csv
import json
import os
import sys
import tempfile

import numpy as np

from Clases.agente_q import AgenteQ
from Clases.cadenas import (adyacencia, cadena_estricta_mas_larga, matriz_probabilidad, transitividad,
                            verificar_cadena, wilson_ic)
from Clases.entorno import PokerHeadsUpEnv
from Clases.evaluacion import evaluar_partidas, prob_racha, simular_rachas
from Clases.liga import (ConfigLiga, correr_liga, cargar_enfrentamientos, reentrenar_y_comparar, ruta_agente)
from Clases.simulacion import PoliticaAleatoria, PoliticaMezcla


def cfg_mini(**kw):
    base = dict(n_agentes=2, max_episodios=4_000, min_episodios=2_000, bloque=2_000, manos_eval=400,
                partidas_eval=60, manos_total=800, partidas_total=80, checkpoint_cada=2_000,
                episodios_decaimiento=3_000, objetivo=0.0, confirmaciones=1)
    base.update(kw)
    return ConfigLiga(**base)


# ---------- rival mezcla / recompensa ----------

def test_politica_mezcla_usa_todos_los_rivales_por_igual():
    rng = np.random.default_rng(0)
    mezcla = PoliticaMezcla([PoliticaAleatoria(rng), PoliticaAleatoria(rng), PoliticaAleatoria(rng)], ["a", "b", "c"])
    for _ in range(3000):
        mezcla.nueva_mano(rng)
    assert all(abs(c / 3000 - 1 / 3) < 0.04 for c in mezcla.conteo)


def test_env_llama_nueva_mano_en_cada_reset():
    mezcla = PoliticaMezcla([PoliticaAleatoria(np.random.default_rng(1)), PoliticaAleatoria(np.random.default_rng(2))])
    env = PokerHeadsUpEnv(oponente=mezcla)
    env.reset(seed=0)
    for _ in range(50):
        env.reset()
    assert sum(mezcla.conteo) == 51 and min(mezcla.conteo) > 5


def test_modo_recompensa_victoria():
    env = PokerHeadsUpEnv(oponente=PoliticaAleatoria(np.random.default_rng(1)), modo_recompensa="victoria")
    env.reset(seed=3)
    rng = np.random.default_rng(0)
    vistos = set()
    for _ in range(200):
        obs, info = env.reset()
        if info["mano_terminada"]:
            vistos.add(info["recompensa"])
            continue
        while True:
            obs, r, terminado, _, info = env.step(int(rng.choice(np.flatnonzero(info["mascara"]))))
            if terminado:
                vistos.add(r)
                assert r == (1.0 if info["gano"] is True else (-1.0 if info["gano"] is False else 0.0))
                break
    assert vistos <= {-1.0, 0.0, 1.0} and {-1.0, 1.0} <= vistos


# ---------- estadística y cadenas ----------

def test_wilson_contiene_la_proporcion():
    lo, hi = wilson_ic(60, 100)
    assert lo < 0.6 < hi and 0.49 < lo and hi < 0.70
    assert wilson_ic(0, 0) == (0.0, 1.0)


def test_cadena_exige_transitividad_completa():
    # A>B, B>C, A>C  -> cadena de 3
    G = np.zeros((3, 3), dtype=int)
    for i, j in ((0, 1), (1, 2), (0, 2)):
        G[i, j], G[j, i] = 95, 5
    P = matriz_probabilidad(G, "mano")
    assert cadena_estricta_mas_larga(adyacencia(P, 0.9)) == [0, 1, 2]
    # ciclo A>B, B>C, C>A -> NO hay cadena de 3 (solo de 2)
    G = np.zeros((3, 3), dtype=int)
    for i, j in ((0, 1), (1, 2), (2, 0)):
        G[i, j], G[j, i] = 95, 5
    P = matriz_probabilidad(G, "mano")
    assert len(cadena_estricta_mas_larga(adyacencia(P, 0.9))) == 2
    tr = transitividad(adyacencia(P, 0.9))
    assert tr["caminos2"] == 3 and tr["caminos2_cerrados"] == 0


def test_cadena_modo_conservador_es_mas_estricto():
    G = np.zeros((2, 2), dtype=int)
    G[0, 1], G[1, 0] = 46, 4          # p=0.92 con solo 50 manos
    assert matriz_probabilidad(G, "mano")[0, 1] >= 0.9
    assert matriz_probabilidad(G, "mano", conservador=True)[0, 1] < 0.9
    G[0, 1], G[1, 0] = 4600, 400      # misma proporción, 100 veces más datos
    assert matriz_probabilidad(G, "mano", conservador=True)[0, 1] >= 0.9


def test_verificar_cadena_revisa_todos_los_pares():
    G = np.zeros((3, 3), dtype=int)
    for i, j in ((0, 1), (1, 2), (0, 2)):
        G[i, j], G[j, i] = 95, 5
    filas = verificar_cadena(G, [0, 1, 2], "mano", 3, 0.9)
    assert len(filas) == 3 and all(f["cumple_puntual"] for f in filas)


def test_simular_rachas_coincide_con_la_formula_si_los_rivales_son_iguales():
    rng = np.random.default_rng(0)
    sim = simular_rachas(PoliticaAleatoria(np.random.default_rng(1)), PoliticaAleatoria(np.random.default_rng(2)),
                         k=3, n_carreras=600, semilla=5)
    assert abs(sim["prob_racha_simulada"] - 0.5) < 0.09
    assert abs(prob_racha(0.5, 3) - 0.5) < 1e-9


def test_evaluar_partidas_alterna_posicion_y_es_justo_entre_iguales():
    r = evaluar_partidas(PoliticaAleatoria(np.random.default_rng(1)), PoliticaAleatoria(np.random.default_rng(2)),
                         n_partidas=300, semilla=3)
    assert abs(r["winrate_partidas"] - 0.5) < 0.1


# ---------- liga ----------

def test_liga_pequena_genera_todos_los_archivos():
    with tempfile.TemporaryDirectory() as tmp:
        correr_liga(cfg_mini(), tmp, imprimir=lambda *a: None)
        esperados = {"manifest.json", "enfrentamientos.csv", "iteraciones.csv", "cadenas.csv",
                     "matriz_mano.csv", "matriz_partida.csv", "matriz_racha.csv"}
        assert esperados <= set(os.listdir(tmp))
        assert os.path.exists(ruta_agente(tmp, 1)) and os.path.exists(ruta_agente(tmp, 2))
        assert os.path.exists(os.path.join(tmp, "entrenamiento", "agente_02", "metricas.csv"))
        assert any(f.startswith("checkpoint_ep") for f in os.listdir(os.path.join(tmp, "entrenamiento", "agente_01")))
        with open(os.path.join(tmp, "enfrentamientos.csv"), newline="") as f:
            filas = list(csv.DictReader(f))
        # agente 1 vs {Uniforme}; agente 2 vs {Uniforme, Agente 1}
        assert [(int(r["a"]), int(r["b"])) for r in filas] == [(1, 0), (2, 0), (2, 1)]
        # número FIJO de manos: el total se divide por igual entre los rivales
        assert [int(r["manos"]) for r in filas] == [800, 400, 400]
        assert [int(r["partidas"]) for r in filas] == [80, 40, 40]
        with open(os.path.join(tmp, "manifest.json")) as f:
            man = json.load(f)
        assert set(man["agentes"]) == {"1", "2"} and "numpy" in man["versiones"]
        G, Gp, E, _ = cargar_enfrentamientos(tmp, 3)
        assert G[2, 1] + G[1, 2] + E[2, 1] == 400 and Gp[2, 1] + Gp[1, 2] == 40


def test_agente_nuevo_parte_de_los_pesos_del_anterior():
    with tempfile.TemporaryDirectory() as tmp:
        # objetivo imposible + 0 manos de entrenamiento efectivas: el agente 2 debe ser copia del 1
        correr_liga(cfg_mini(n_agentes=1), tmp, imprimir=lambda *a: None)
        a1 = AgenteQ.cargar(ruta_agente(tmp, 1))
        cfg = cfg_mini(n_agentes=2, max_episodios=1, min_episodios=1, bloque=1, checkpoint_cada=10**9)
        from Clases.liga import entrenar_agente_liga
        entrenar_agente_liga(cfg, tmp, 2, imprimir=lambda *a: None)
        a2 = AgenteQ.cargar(ruta_agente(tmp, 2))
        diferentes = int((a1.Q != a2.Q).sum())
        assert diferentes <= 40   # una mano solo toca las (estado, acción) de sus pocas decisiones (de 10.560 celdas)


def test_criterio_cada_rival_no_se_cumple_si_no_se_alcanza():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = cfg_mini(n_agentes=1, objetivo=1.01)       # imposible
        correr_liga(cfg, tmp, imprimir=lambda *a: None)
        with open(os.path.join(tmp, "iteraciones.csv"), newline="") as f:
            fila = next(csv.DictReader(f))
        assert fila["alcanzo_objetivo"] == "0" and int(fila["episodios"]) == cfg.max_episodios


def test_parada_anticipada_cuando_se_alcanza_el_objetivo():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = cfg_mini(n_agentes=1, objetivo=0.0, max_episodios=20_000)
        correr_liga(cfg, tmp, imprimir=lambda *a: None)
        with open(os.path.join(tmp, "iteraciones.csv"), newline="") as f:
            fila = next(csv.DictReader(f))
        assert fila["alcanzo_objetivo"] == "1" and int(fila["episodios"]) == 2_000


def test_reentrenar_reproduce_la_misma_tabla_q():
    with tempfile.TemporaryDirectory() as tmp:
        correr_liga(cfg_mini(n_agentes=2), tmp, imprimir=lambda *a: None)
        for m in (1, 2):
            nuevo, original, igual = reentrenar_y_comparar(tmp, m, imprimir=lambda *a: None)
            assert igual and nuevo == original


def test_misma_semilla_misma_liga_y_semilla_distinta_otra():
    def hashes(semilla):
        with tempfile.TemporaryDirectory() as tmp:
            man = correr_liga(cfg_mini(semilla=semilla), tmp, imprimir=lambda *a: None)
            return [man["agentes"][k]["hash_q_sha256"] for k in ("1", "2")]
    assert hashes(5) == hashes(5)
    assert hashes(5) != hashes(6)


def test_reanudar_agrega_agentes_sin_tocar_los_anteriores():
    with tempfile.TemporaryDirectory() as tmp:
        man1 = correr_liga(cfg_mini(n_agentes=2), tmp, imprimir=lambda *a: None)
        try:
            correr_liga(cfg_mini(n_agentes=3), tmp, imprimir=lambda *a: None)
            assert False, "debía exigir reanudar=True"
        except FileExistsError:
            pass
        man2 = correr_liga(cfg_mini(n_agentes=3), tmp, reanudar=True, imprimir=lambda *a: None)
        assert set(man2["agentes"]) == {"1", "2", "3"}
        assert man2["agentes"]["1"]["hash_q_sha256"] == man1["agentes"]["1"]["hash_q_sha256"]
        assert man2["agentes"]["2"]["hash_q_sha256"] == man1["agentes"]["2"]["hash_q_sha256"]
        # cambiar un parámetro de entrenamiento al reanudar rompería la reproducibilidad
        try:
            correr_liga(cfg_mini(n_agentes=4, alpha=0.5), tmp, reanudar=True, imprimir=lambda *a: None)
            assert False
        except ValueError:
            pass


def test_cadenas_csv_tiene_modos_y_umbrales():
    with tempfile.TemporaryDirectory() as tmp:
        correr_liga(cfg_mini(n_agentes=2), tmp, imprimir=lambda *a: None)
        with open(os.path.join(tmp, "cadenas.csv"), newline="") as f:
            filas = list(csv.DictReader(f))
        assert {r["iteracion"] for r in filas} == {"1", "2"}
        assert {r["conservador"] for r in filas} == {"0", "1"}
        assert len(filas) == 2 * 2 * 4  # iteraciones x modos x umbrales


def test_los_tres_criterios_de_victoria_corren_y_son_coherentes():
    for criterio in ("racha", "mano", "partida"):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = cfg_mini(n_agentes=2, criterio_victoria=criterio)
            correr_liga(cfg, tmp, imprimir=lambda *a: None)
            with open(os.path.join(tmp, "iteraciones.csv"), newline="") as f:
                filas = list(csv.DictReader(f))
            assert len(filas) == 2
            with open(os.path.join(tmp, "cadenas.csv"), newline="") as f:
                assert {r["criterio"] for r in csv.DictReader(f)} == {criterio}
            vs_uni = float(filas[0]["winrate_vs_uniforme"])
            assert 0.0 <= vs_uni <= 1.0


def test_recompensa_auto_alinea_entrenamiento_con_el_criterio():
    assert cfg_mini(criterio_victoria="racha").modo_recompensa() == "victoria"
    assert cfg_mini(criterio_victoria="mano").modo_recompensa() == "victoria"
    assert cfg_mini(criterio_victoria="partida").modo_recompensa() == "fichas"
    assert cfg_mini(criterio_victoria="partida", recompensa="victoria").modo_recompensa() == "victoria"


def test_config_invalida_se_rechaza_y_se_serializa():
    for kw in (dict(criterio_victoria="otro"), dict(criterio_parada="otro"), dict(recompensa="otro")):
        try:
            ConfigLiga(**kw)
            assert False, kw
        except ValueError:
            pass
    cfg = cfg_mini(umbrales=(0.5, 0.95))
    assert ConfigLiga.from_dict(json.loads(json.dumps(cfg.to_dict()))) == cfg


def test_enfrentamientos_con_manos_fijas_por_par():
    with tempfile.TemporaryDirectory() as tmp:
        correr_liga(cfg_mini(n_agentes=2, manos_por_par=300, partidas_por_par=20), tmp, imprimir=lambda *a: None)
        with open(os.path.join(tmp, "enfrentamientos.csv"), newline="") as f:
            filas = list(csv.DictReader(f))
        assert [int(r["manos"]) for r in filas] == [300, 300, 300]
        assert [int(r["partidas"]) for r in filas] == [20, 20, 20]


def test_rival_uniforme_de_la_liga_es_realmente_uniforme():
    from Clases.liga import crear_uniforme
    p = crear_uniforme(cfg_mini(), 0)
    assert np.allclose(p.pesos, 1.0)


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