"""
Pruebas de la partida humano-vs-bot (lógica de la web, sin interfaz).

Desde la raíz del repo:   python test_partida_web.py
"""
import numpy as np

from Clases.agente_q import AgenteQ
from partida_web import PartidaContraBot, BOT, HUMANO
from Clases.simulacion import PoliticaAleatoria


def _contiene_lista(obj, cartas) -> bool:
    """¿Aparece en `obj` (recursivo) alguna lista cuyos elementos sean exactamente las cartas dadas?"""
    cartas = sorted(cartas)
    if isinstance(obj, (list, tuple)):
        if len(obj) == len(cartas) and all(isinstance(x, (int, np.integer)) for x in obj) and sorted(obj) == cartas:
            return True
        return any(_contiene_lista(x, cartas) for x in obj)
    if isinstance(obj, dict):
        return any(_contiene_lista(x, cartas) for x in obj.values())
    return False


def _humano_azar(p, rng):
    acc = p.acciones_humano()
    assert acc, "turno del humano sin acciones legales"
    p.actuar(int(rng.choice([a["id"] for a in acc])))


def _jugar_mano(p, rng):
    while not p.motor.esta_terminada():
        _humano_azar(p, rng)


def test_conserva_fichas_y_acumula_estadisticas():
    rng = np.random.default_rng(0)
    p = PartidaContraBot(PoliticaAleatoria.uniforme(np.random.default_rng(1)), semilla=2, reiniciar_fichas=True)
    for _ in range(400):
        _jugar_mano(p, rng)
        assert sum(j.fichas for j in p.jugadores) == 2000
        if _ < 399:
            p.nueva_mano()
    assert p.manos == 400 and len(p.netos) == 400
    assert abs(p.vista()["neto_total"] - sum(p.netos)) < 1e-9


def test_partida_completa_termina_con_campeon():
    rng = np.random.default_rng(3)
    p = PartidaContraBot(PoliticaAleatoria.uniforme(np.random.default_rng(4)), semilla=5, reiniciar_fichas=False)
    for _ in range(2000):
        _jugar_mano(p, rng)
        if p.partida_terminada:
            break
        p.nueva_mano()
    v = p.vista()
    assert v["partida_terminada"] and v["campeon"] in v["nombres"]
    assert sorted(v["fichas"]) == [0, 2000]
    try:
        p.nueva_mano()
        assert False
    except ValueError:
        pass


def test_no_filtra_cartas_del_bot():
    rng = np.random.default_rng(6)
    p = PartidaContraBot(PoliticaAleatoria.uniforme(np.random.default_rng(7)), semilla=8, reiniciar_fichas=True)
    reveladas = ocultas = 0
    for _ in range(300):
        cartas_bot = set(p.jugadores[BOT].mano)
        while not p.motor.esta_terminada():
            v = p.vista()
            assert v["mano_bot"] is None                       # nunca durante la mano
            assert not _contiene_lista(v, cartas_bot), "la vista contiene la mano del bot"
            _humano_azar(p, rng)
        v = p.vista()
        if v["resultado"]["por_retiro"]:
            assert v["mano_bot"] is None; ocultas += 1         # retiro: no se muestran
        else:
            assert sorted(v["mano_bot"]) == sorted(cartas_bot) and v["jugada_bot"]; reveladas += 1
        p.nueva_mano()
    assert reveladas > 20 and ocultas > 20
    # el historial no debe contener ids de cartas del bot en la mano en curso
    assert all("cartas" not in linea for linea in p.vista()["historial"])


def test_bot_agente_q_juega_sin_acciones_ilegales():
    ag = AgenteQ(rng=np.random.default_rng(0)); ag.epsilon = 0.0
    rng = np.random.default_rng(9)
    p = PartidaContraBot(ag, nombre_bot="Q", semilla=1, reiniciar_fichas=True)
    for _ in range(300):
        _jugar_mano(p, rng)
        p.nueva_mano()


def test_politica_ilegal_se_corrige():
    class Loca:
        usa_obs = False
        def __call__(self, obs, mascara): return 99
    p = PartidaContraBot(Loca(), semilla=1, reiniciar_fichas=True)
    rng = np.random.default_rng(1)
    for _ in range(50):
        _jugar_mano(p, rng)
        p.nueva_mano()


def test_turno_y_subida_personalizada():
    p = PartidaContraBot(PoliticaAleatoria.uniforme(np.random.default_rng(1)), semilla=3, reiniciar_fichas=True)
    for _ in range(100):
        if p.motor.esta_terminada():
            p.nueva_mano(); continue
        r = p.rango_subida()
        if r:
            lo, hi = r; assert 0 < lo <= hi
            p.subir(lo)
        else:
            p.actuar(p.acciones_humano()[0]["id"])
    # fuera de turno / mano terminada: error claro
    while not p.motor.esta_terminada():
        p.actuar(p.acciones_humano()[0]["id"])
    try:
        p.actuar(1); assert False
    except ValueError:
        pass


def test_alterna_dealer():
    rng = np.random.default_rng(0)
    p = PartidaContraBot(PoliticaAleatoria.uniforme(np.random.default_rng(1)), semilla=4, reiniciar_fichas=True)
    d = []
    for _ in range(6):
        d.append(p.vista()["dealer_idx"]); _jugar_mano(p, rng); p.nueva_mano()
    assert all(d[i] != d[i + 1] for i in range(5))


def main():
    pruebas = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    fallos = 0
    for prueba in pruebas:
        try:
            prueba(); print(f"OK   - {prueba.__name__}")
        except AssertionError as e:
            fallos += 1; print(f"FALLO - {prueba.__name__}: {e}")
        except Exception as e:
            fallos += 1; print(f"ERROR - {prueba.__name__}: {type(e).__name__}: {e}")
    print()
    print(f"Las {len(pruebas)} pruebas pasaron correctamente." if fallos == 0 else f"{fallos} de {len(pruebas)} pruebas fallaron.")


if __name__ == "__main__":
    main()
