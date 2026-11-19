"""Pruebas del motor de reglas. Se corren desde la carpeta del proyecto con:

    python3 -m unittest -v
"""

import copy
import io
import random
import unittest
from collections import Counter
from contextlib import redirect_stdout
import sys
from pathlib import Path

# Añadir la raíz del proyecto al sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.azulvitral import (
    JOKER,
    AzulVitralGame,
    Color,
    InvalidMoveError,
    Move,
    PatternStrip,
    load_strips,
)
from src.impresion import (
    game_text,
    print_center,
    print_factories,
    print_final,
    print_game,
    print_header,
    print_player,
    print_turn,
)
from src.partida import TwoPlayerMatch

NA, AM, RS, AZ, TR = Color.NARANJO, Color.AMARILLO, Color.ROSADO, Color.AZUL, Color.TRANSPARENTE
J = JOKER


def nueva_partida(seed=7):
    return AzulVitralGame(["Ana", "Beto"], seed=seed, first_player=0)


def tira(patron, puestas=(), retirada=False):
    """Tira con `patron` boca arriba y `puestas` ocupando los primeros espacios."""
    strip = PatternStrip(side_a=list(patron), side_b=[TR] * 5)
    strip.filled_spaces = list(puestas) + [None] * (5 - len(puestas))
    strip.is_removed = retirada
    return strip


def preparar(g, fabricas=(), centro=(), tiras=None, vidriero=1, jugador=0):
    """Deja la mesa y el tablero de un jugador como los necesita la prueba.

    Columnas y vidriero van de 1 a 8, como se ven en la mesa. Las tiras que
    no se indican quedan todas transparentes y vacías.
    """
    for i, factory in enumerate(g.factories):
        factory.tiles = list(fabricas[i]) if i < len(fabricas) else []
    g.center.tiles = list(centro)
    player = g.players[jugador]
    tiras = tiras or {}
    player.strips = [tiras.get(c, tira([TR] * 5)) for c in range(1, 9)]
    player.glazier = vidriero - 1
    return player


def foto(g):
    """Copia de todo el estado, para comparar antes y después."""
    estado = (
        g.players, g.factories, g.center, g.bag, g.tower, g.round_marker,
        g.round, g.current, g.round_starter, g.game_over, g.final_scores,
    )
    return copy.deepcopy(estado), g.rng.getstate()


def contar_piezas(g):
    total = len(g.bag) + len(g.tower) + len(g.center.tiles)
    total += sum(len(f.tiles) for f in g.factories)
    total += sum(1 for c in g.round_marker if c is not None)
    for p in g.players:
        total += sum(len(s.tiles()) for s in p.strips)
        total += sum(1 for w in p.windows for c in w if c is not None)
    return total


class DatosDeTirasTest(unittest.TestCase):
    def test_cuatro_tableros_de_ocho_tiras_con_dos_caras(self):
        tableros = load_strips()
        self.assertEqual(sorted(tableros), ["amarillo", "azul", "gris", "rojo"])
        for nombre, tiras in tableros.items():
            self.assertEqual(len(tiras), 8, nombre)
            for anverso, reverso in tiras:
                self.assertEqual(len(anverso), 5)
                self.assertEqual(len(reverso), 5)

    def test_una_sola_cara_con_dos_comodines_por_tablero(self):
        for nombre, tiras in load_strips().items():
            caras = [c for par in tiras for c in par if JOKER in c]
            self.assertEqual(len(caras), 1, nombre)
            self.assertEqual(caras[0].count(JOKER), 2, nombre)

    def test_los_cuatro_tableros_tienen_las_mismas_caras_sin_comodin(self):
        # En las capturas los cuatro tableros repiten las mismas 15 caras de
        # color; si una no calza, hubo un error al transcribir.
        def caras(tiras):
            return sorted(
                tuple(sorted(c.name for c in cara))
                for par in tiras for cara in par if JOKER not in cara
            )
        tableros = load_strips()
        referencia = caras(tableros["rojo"])
        self.assertEqual(len(referencia), 15)
        for nombre, tiras in tableros.items():
            self.assertEqual(caras(tiras), referencia, nombre)


class PreparacionTest(unittest.TestCase):
    def test_estado_inicial(self):
        g = nueva_partida()
        self.assertEqual(g.round, 1)
        self.assertEqual(len(g.bag), 74)
        self.assertTrue(all(len(f.tiles) == 4 for f in g.factories))
        self.assertEqual(g.center.tiles, [])
        self.assertTrue(g.center.has_first_player_token)
        self.assertEqual(g.get_current_player().name, "Ana")
        for p in g.players:
            self.assertEqual((p.glazier, p.score, p.broken_position), (0, 0, 1))
            self.assertEqual(len(p.strips), 8)
        self.assertEqual(Counter(g.round_marker[1:]), Counter(Color))
        self.assertEqual(contar_piezas(g), 100)

    def test_cada_jugador_recibe_las_tiras_de_su_tablero(self):
        g = AzulVitralGame(["Ana", "Beto"], boards=("gris", "azul"), seed=1)
        self.assertTrue(all(s.name.startswith("gris-") for s in g.players[0].strips))
        self.assertTrue(all(s.name.startswith("azul-") for s in g.players[1].strips))

    def test_la_cara_con_comodines_empieza_boca_abajo(self):
        for seed in range(40):
            g = AzulVitralGame(["Ana", "Beto"], seed=seed)
            for p in g.players:
                for s in p.strips:
                    self.assertNotIn(JOKER, s.pattern())

    def test_la_misma_semilla_repite_la_partida(self):
        a, b = nueva_partida(seed=5), nueva_partida(seed=5)
        self.assertEqual(foto(a), foto(b))


class ColocacionValidaTest(unittest.TestCase):
    def test_tomar_de_una_fabrica_manda_el_resto_al_centro(self):
        g = nueva_partida()
        p = preparar(g, fabricas=[[AZ, AZ, NA, AM]], tiras={1: tira([AZ] * 5)})
        rep = g.apply_move(Move.take(1, AZ, 1))
        self.assertEqual(g.factories[0].tiles, [])
        self.assertEqual(Counter(g.center.tiles), Counter([NA, AM]))
        self.assertEqual(p.strips[0].filled_spaces, [AZ, AZ, None, None, None])
        self.assertEqual((rep.placed, rep.broken), (2, 0))
        self.assertEqual(g.get_current_player().name, "Beto")

    def test_el_vidriero_se_mueve_a_la_tira_elegida(self):
        g = nueva_partida()
        p = preparar(g, fabricas=[[NA, NA, AM, AM]], tiras={4: tira([NA] * 5)})
        g.apply_move(Move.take(1, NA, 4))
        self.assertEqual(p.glazier, 3)

    def test_colocar_bajo_el_vidriero_no_lo_mueve(self):
        g = nueva_partida()
        p = preparar(g, fabricas=[[NA, AM]], tiras={3: tira([NA] * 5)}, vidriero=3)
        g.apply_move(Move.take(1, NA, 3))
        self.assertEqual(p.glazier, 2)

    def test_opcion_b_vuelve_a_la_tira_activa_mas_a_la_izquierda(self):
        g = nueva_partida()
        p = preparar(g, fabricas=[[NA]], vidriero=6,
                     tiras={1: tira([TR] * 5, retirada=True), 2: tira([TR] * 5, retirada=True)})
        g.apply_move(Move.reset_glazier())
        self.assertEqual(p.glazier, 2)


class ColocacionInvalidaTest(unittest.TestCase):
    def rechaza(self, g, move, motivo):
        antes = foto(g)
        with self.assertRaises(InvalidMoveError) as ctx:
            g.apply_move(move)
        self.assertIn(motivo, str(ctx.exception))
        self.assertEqual(foto(g), antes, "una jugada rechazada no puede cambiar el estado")

    def test_tira_a_la_izquierda_del_vidriero(self):
        g = nueva_partida()
        preparar(g, fabricas=[[AZ]], tiras={2: tira([AZ] * 5)}, vidriero=4)
        self.rechaza(g, Move.take(1, AZ, 2), "a la izquierda del vidriero")

    def test_tira_retirada(self):
        g = nueva_partida()
        preparar(g, fabricas=[[AZ]], tiras={3: tira([AZ] * 5, retirada=True)})
        self.rechaza(g, Move.take(1, AZ, 3), "ya se retiró")

    def test_color_que_no_esta_en_la_fabrica(self):
        g = nueva_partida()
        preparar(g, fabricas=[[AZ, AZ]])
        self.rechaza(g, Move.take(1, NA, 1), "no tiene piezas Naranjo")

    def test_fabrica_vacia_o_inexistente(self):
        g = nueva_partida()
        preparar(g, fabricas=[[AZ], []])
        self.rechaza(g, Move.take(2, AZ, 1), "está vacía")
        self.rechaza(g, Move.take(6, AZ, 1), "Origen inválido")
        self.rechaza(g, Move.take(0, AZ, 1), "Origen inválido")

    def test_color_que_no_esta_en_el_centro(self):
        g = nueva_partida()
        preparar(g, fabricas=[[AZ]], centro=[AM])
        self.rechaza(g, Move.take("centro", NA, 1), "En el centro no hay piezas Naranjo")

    def test_falta_el_color_o_la_tira(self):
        g = nueva_partida()
        preparar(g, fabricas=[[AZ]])
        self.rechaza(g, Move.take(1, None, 1), "Falta indicar el color")
        self.rechaza(g, Move.take(1, "AZUL", 1), "Color inválido")
        self.rechaza(g, Move.take(1, AZ, None), "Falta elegir la tira")
        self.rechaza(g, Move.take(1, AZ, 9), "no existe")

    def test_no_se_puede_romper_si_otra_tira_alcanzable_tiene_espacio(self):
        g = nueva_partida()
        preparar(g, fabricas=[[AZ, AZ]], tiras={1: tira([AM] * 5), 2: tira([AZ] * 5)})
        self.rechaza(g, Move.take(1, AZ, 1), "pero sí en la(s) tira(s) 2")

    def test_falta_la_pieza_para_la_ventana(self):
        g = nueva_partida()
        preparar(g, fabricas=[[AZ]], tiras={1: tira([AZ] * 5, [AZ] * 4)})
        self.rechaza(g, Move.take(1, AZ, 1), "falta elegir qué pieza va a la ventana")

    def test_pieza_para_la_ventana_que_no_esta_en_la_tira(self):
        g = nueva_partida()
        preparar(g, fabricas=[[AZ]], tiras={1: tira([AZ] * 5, [AZ] * 4)})
        self.rechaza(g, Move.take(1, AZ, 1, NA), "No hay ninguna pieza Naranjo")

    def test_pieza_para_la_ventana_sin_completar_la_tira(self):
        g = nueva_partida()
        preparar(g, fabricas=[[AZ]], tiras={1: tira([AZ] * 5)})
        self.rechaza(g, Move.take(1, AZ, 1, AZ), "no se completa")

    def test_opcion_b_con_el_vidriero_ya_a_la_izquierda(self):
        g = nueva_partida()
        preparar(g, fabricas=[[AZ]])
        self.rechaza(g, Move.reset_glazier(), "ya está en la tira más a la izquierda")

    def test_opcion_b_sin_tiras_activas_a_la_izquierda(self):
        # El vidriero quedó en la columna 1 después de retirar esa tira:
        # ya está a la izquierda de todas las que quedan.
        g = nueva_partida()
        preparar(g, fabricas=[[AZ]], tiras={1: tira([TR] * 5, retirada=True)}, vidriero=1)
        self.rechaza(g, Move.reset_glazier(), "ya está en la tira más a la izquierda")

    def test_opcion_b_no_lleva_piezas(self):
        g = nueva_partida()
        preparar(g, fabricas=[[AZ]], vidriero=3)
        self.rechaza(g, Move("B", source=1, color=AZ), "no lleva")

    def test_opcion_desconocida(self):
        g = nueva_partida()
        self.rechaza(g, Move("C"), "Opción desconocida")

    def test_jugar_con_la_partida_terminada(self):
        g = nueva_partida()
        g.game_over = True
        self.rechaza(g, Move.reset_glazier(), "ya terminó")


class ComodinesTest(unittest.TestCase):
    def test_primero_se_llenan_los_espacios_del_color(self):
        g = nueva_partida()
        p = preparar(g, fabricas=[[AZ, AZ, AM, AM]], tiras={1: tira([AZ, J, NA, J, AZ])})
        g.apply_move(Move.take(1, AZ, 1))
        self.assertEqual(p.strips[0].filled_spaces, [AZ, None, None, None, AZ])

    def test_el_comodin_recibe_lo_que_no_cabe_en_su_color(self):
        g = nueva_partida()
        p = preparar(g, fabricas=[[AZ, AZ, AZ, AM]], tiras={1: tira([AZ, J, NA, J, AZ])})
        rep = g.apply_move(Move.take(1, AZ, 1))
        self.assertEqual(p.strips[0].filled_spaces, [AZ, AZ, None, None, AZ])
        self.assertEqual(rep.broken, 0)

    def test_el_comodin_acepta_un_color_que_no_esta_en_la_tira(self):
        g = nueva_partida()
        p = preparar(g, fabricas=[[RS, RS, RS, AM]], tiras={1: tira([AZ, J, NA, J, AZ])})
        rep = g.apply_move(Move.take(1, RS, 1))
        self.assertEqual(p.strips[0].filled_spaces, [None, RS, None, RS, None])
        self.assertEqual(rep.broken, 1)


class PiezasRotasTest(unittest.TestCase):
    def test_las_que_no_caben_van_a_la_torre_y_avanzan_el_track(self):
        g = nueva_partida()
        p = preparar(g, fabricas=[[AZ] * 4, [NA]], tiras={1: tira([AZ, AZ, NA, NA, NA])})
        torre = len(g.tower)
        rep = g.apply_move(Move.take(1, AZ, 1))
        self.assertEqual((rep.placed, rep.broken), (2, 2))
        self.assertEqual(len(g.tower), torre + 2)
        self.assertEqual(p.broken_position, 3)

    def test_sin_espacio_en_ninguna_tira_alcanzable_se_rompe_todo(self):
        # La tira 1 tiene espacio pero está a la izquierda del vidriero.
        g = nueva_partida()
        p = preparar(g, fabricas=[[NA, NA, NA, AM]], tiras={1: tira([NA] * 5)}, vidriero=2)
        rep = g.apply_move(Move.take(1, NA, 5))
        self.assertEqual((rep.placed, rep.broken), (0, 3))
        self.assertEqual(p.glazier, 4)
        self.assertEqual(p.broken_position, 4)

    def test_sin_tiras_alcanzables_se_toma_sin_tira_y_se_rompe_todo(self):
        g = nueva_partida()
        p = preparar(g, fabricas=[[NA, NA, AM]], tiras={8: tira([NA] * 5, retirada=True)}, vidriero=8)
        antes = foto(g)
        with self.assertRaises(InvalidMoveError):
            g.apply_move(Move.take(1, NA, 8))
        self.assertEqual(foto(g), antes)
        rep = g.apply_move(Move.take(1, NA, None))
        self.assertEqual(rep.broken, 2)
        self.assertEqual(p.glazier, 7)
        self.assertEqual(p.broken_position, 3)

    def test_la_primera_toma_del_centro_lleva_la_loseta_y_un_cristal_roto(self):
        g = nueva_partida()
        ana = preparar(g, fabricas=[[NA] * 4], centro=[AZ, AM], tiras={1: tira([AZ] * 5)})
        beto = preparar(g, fabricas=[[NA] * 4], centro=[AZ, AM], tiras={1: tira([AM] * 5)}, jugador=1)
        rep = g.apply_move(Move.take("centro", AZ, 1))
        self.assertTrue(rep.took_first_player_token)
        self.assertTrue(ana.has_first_player_token)
        self.assertFalse(g.center.has_first_player_token)
        self.assertEqual(ana.broken_position, 2)
        g.apply_move(Move.take("centro", AM, 1))
        self.assertEqual(beto.broken_position, 1)


class CierreDeTiraTest(unittest.TestCase):
    def test_bonificacion_por_color_de_ronda_y_primer_cierre(self):
        g = nueva_partida()
        g.round_marker[0] = NA
        p = preparar(g, fabricas=[[NA, NA, AM]], tiras={1: tira([AZ, AZ, NA, NA, J], [AZ, AZ, NA])})
        torre = len(g.tower)
        rep = g.apply_move(Move.take(1, NA, 1, NA))
        # Tres naranjas (una sobre el comodín) = +3, más los 4 de la ventana 1.
        self.assertEqual((rep.color_bonus, rep.window_points), (3, 4))
        self.assertEqual(p.score, 7)
        self.assertEqual(p.windows[0], [NA, None])
        self.assertEqual(p.strips[0].current_side, 1)
        self.assertEqual(p.strips[0].filled_spaces, [None] * 5)
        self.assertFalse(p.strips[0].is_removed)
        self.assertEqual(len(g.tower), torre + 4)

    def test_segundo_cierre_va_abajo_y_retira_la_tira(self):
        g = nueva_partida()
        p = preparar(g, fabricas=[[AZ, AM]], tiras={2: tira([AZ] * 5, [AZ] * 4)})
        p.windows[1] = [AM, None]
        rep = g.apply_move(Move.take(1, AZ, 2, AZ))
        self.assertEqual(p.windows[1], [AM, AZ])
        self.assertTrue(p.strips[1].is_removed)
        self.assertTrue(rep.strip_removed)

    def test_puntuacion_en_cascada_hacia_la_derecha(self):
        g = nueva_partida()
        g.round_marker[0] = NA
        p = preparar(g, fabricas=[[AZ, AM]], tiras={
            2: tira([AZ] * 5, [AZ] * 4),
            6: tira([TR] * 5, retirada=True),
        })
        p.windows[0] = [TR, None]  # a la izquierda: no suma
        p.windows[2] = [AM, None]  # ventana 3: 3 puntos
        p.windows[5] = [RS, RS]    # ventana 6: 1 punto
        rep = g.apply_move(Move.take(1, AZ, 2, AZ))
        self.assertEqual(rep.window_points, 3 + 3 + 1)
        self.assertEqual(p.score, 7)


class DesbordeTest(unittest.TestCase):
    def romper(self, posicion, cantidad):
        g = nueva_partida()
        p = preparar(g, fabricas=[[AZ] * cantidad, [NA]])
        p.broken_position = posicion
        p.score = 30
        rep = g.apply_move(Move.take(1, AZ, 1))
        return p, rep

    def test_llegar_a_la_casilla_10_resta_18_y_vuelve_a_1(self):
        p, rep = self.romper(8, 2)
        self.assertEqual((p.broken_position, p.score, rep.overflows), (1, 12, 1))

    def test_las_piezas_que_sobran_siguen_avanzando_desde_la_1(self):
        p, rep = self.romper(8, 4)
        self.assertEqual((p.broken_position, p.score), (3, 12))

    def test_la_loseta_inicial_tambien_puede_desbordar(self):
        g = nueva_partida()
        p = preparar(g, fabricas=[[NA]], centro=[AZ], tiras={1: tira([AZ] * 5)})
        p.broken_position = 9
        p.score = 25
        g.apply_move(Move.take("centro", AZ, 1))
        self.assertEqual((p.broken_position, p.score), (1, 7))

    def test_dos_desbordes_en_una_jugada(self):
        g = nueva_partida()
        p = g.players[0]
        p.score = 40
        self.assertEqual(g._add_broken_glass(p, 18), 2)
        self.assertEqual((p.broken_position, p.score), (1, 4))

    def test_el_track_no_se_reinicia_al_cambiar_de_ronda(self):
        g = nueva_partida()
        p = preparar(g, fabricas=[[AZ]], tiras={1: tira([AZ] * 5)})
        p.broken_position = 5
        rep = g.apply_move(Move.take(1, AZ, 1))
        self.assertTrue(rep.round_ended)
        self.assertEqual(p.broken_position, 5)


class FinDeRondaTest(unittest.TestCase):
    def test_fin_de_ronda(self):
        g = nueva_partida()
        color_ronda_1 = g.round_marker[0]
        preparar(g, fabricas=[[AZ]], tiras={1: tira([AZ] * 5)})
        torre = len(g.tower)
        rep = g.apply_move(Move.take(1, AZ, 1))
        self.assertTrue(rep.round_ended)
        self.assertEqual(g.round, 2)
        self.assertIsNone(g.round_marker[0])
        self.assertEqual(g.tower[-1], color_ronda_1)
        self.assertEqual(len(g.tower), torre + 1)
        self.assertTrue(all(len(f.tiles) == 4 for f in g.factories))
        self.assertTrue(g.center.has_first_player_token)

    def test_empieza_la_ronda_quien_tomo_la_loseta(self):
        g = nueva_partida()
        preparar(g, fabricas=[[NA]], centro=[AZ], tiras={1: tira([NA] * 5)})
        preparar(g, fabricas=[[NA]], centro=[AZ], tiras={1: tira([AZ] * 5)}, jugador=1)
        g.apply_move(Move.take(1, NA, 1))
        g.apply_move(Move.take("centro", AZ, 1))
        self.assertEqual(g.round, 2)
        self.assertEqual(g.get_current_player().name, "Beto")
        self.assertFalse(any(p.has_first_player_token for p in g.players))

    def test_la_bolsa_se_rellena_con_la_torre_y_si_no_alcanza_quedan_incompletas(self):
        g = nueva_partida()
        preparar(g, fabricas=[[AZ]], tiras={1: tira([AZ] * 5)})
        g.bag = [AM, AM]
        g.tower = [NA] * 9
        g.apply_move(Move.take(1, AZ, 1))
        # 2 de la bolsa + 9 de la torre + la pieza del marcador = 12
        self.assertEqual([len(f.tiles) for f in g.factories], [4, 4, 4, 0, 0])
        self.assertEqual((g.bag, g.tower), ([], []))

    def test_la_partida_termina_despues_de_la_ronda_6(self):
        g = nueva_partida()
        g.round = 6
        g.round_marker = [None] * 5 + [AM]
        preparar(g, fabricas=[[AZ]], tiras={1: tira([AZ] * 5)})
        rep = g.apply_move(Move.take(1, AZ, 1))
        self.assertTrue(rep.game_over)
        self.assertTrue(g.game_over)
        self.assertEqual(set(g.final_scores), {"Ana", "Beto"})
        self.assertIsNotNone(g.validate_move(Move.reset_glazier()))


class PuntuacionFinalTest(unittest.TestCase):
    def test_piezas_sobrantes_suman_1_cada_3(self):
        g = nueva_partida()
        p = preparar(g, tiras={
            1: tira([AZ] * 5, [AZ, AZ]),
            2: tira([NA] * 5, [NA, NA]),
            3: tira([AM] * 5, [AM, AM, AM]),
        })
        g.end_game_scoring()
        detalle = g.final_scores["Ana"]
        self.assertEqual((detalle.leftover_tiles, detalle.leftover_points), (7, 2))

    def test_resta_la_casilla_del_cristal_roto(self):
        g = nueva_partida()
        p = preparar(g)
        p.score = 20
        p.broken_position = 6
        g.end_game_scoring()
        self.assertEqual(p.score, 12)
        self.assertEqual(g.final_scores["Ana"].score_before, 20)

    def test_ornamentos(self):
        g = nueva_partida()
        p = preparar(g)
        p.windows = [
            [AZ, AZ], [NA, NA],      # ornamento 1: 4 de 4
            [AM, AM], [RS, None],    # ornamento 2: 3 de 4
            [TR, None], [AZ, None],  # ornamento 3: 2 de 4
            [NA, None], [None, None],  # ornamento 4: 1 de 4
        ]
        g.end_game_scoring()
        self.assertEqual(g.final_scores["Ana"].ornament_points, [10, 6, 3, 0])
        self.assertEqual(p.score, 19)

    def test_todo_junto(self):
        g = nueva_partida()
        p = preparar(g, tiras={1: tira([AZ] * 5, [AZ] * 3)})
        p.score = 20
        p.broken_position = 2
        p.windows[6] = [NA, None]
        p.windows[7] = [AM, None]
        g.end_game_scoring()
        self.assertEqual(p.score, 20 + 1 - 1 + 3)


class GanadorTest(unittest.TestCase):
    def terminada(self, puntos, posiciones):
        g = nueva_partida()
        for p, pts, pos in zip(g.players, puntos, posiciones):
            p.score, p.broken_position = pts, pos
        g.game_over = True
        return g

    def test_gana_el_que_tiene_mas_puntos(self):
        g = self.terminada((40, 35), (1, 1))
        self.assertEqual(g.winner().name, "Ana")

    def test_empate_gana_el_de_la_casilla_mas_baja_del_cristal_roto(self):
        g = self.terminada((30, 30), (7, 2))
        self.assertEqual(g.winner().name, "Beto")

    def test_el_desempate_no_cuenta_los_desbordes(self):
        # Ana llegó dos veces a la casilla 10 y quedó en la 2; Beto nunca
        # desbordó y quedó en la 7. Solo cuenta la casilla: gana Ana.
        g = nueva_partida()
        ana, beto = g.players
        ana.score = 60
        g._add_broken_glass(ana, 19)
        beto.broken_position = 7
        ana.score = beto.score = 30
        g.game_over = True
        self.assertEqual(ana.broken_position, 2)
        self.assertEqual(g.winner().name, "Ana")
        self.assertEqual(g.ranking()[0][0].name, "Ana")

    def test_empate_tecnico(self):
        g = self.terminada((30, 30), (3, 3))
        self.assertIsNone(g.winner())

    def test_no_hay_ganador_antes_del_final(self):
        with self.assertRaises(RuntimeError):
            nueva_partida().winner()


class PuntajeNoNegativoTest(unittest.TestCase):
    def test_un_desborde_deja_el_puntaje_en_0(self):
        g = nueva_partida()
        p = preparar(g, fabricas=[[AZ], [NA]])
        p.score, p.broken_position = 10, 9
        g.apply_move(Move.take(1, AZ, 1))
        self.assertEqual((p.score, p.broken_position), (0, 1))

    def test_el_desborde_se_resta_en_el_turno_y_no_otra_vez_al_final(self):
        g = nueva_partida()
        p = preparar(g, fabricas=[[AZ], [NA]])
        p.score, p.broken_position = 30, 9
        g.apply_move(Move.take(1, AZ, 1))
        self.assertEqual(p.score, 12)
        g.end_game_scoring()
        self.assertEqual(p.score, 12)  # casilla 1 vale 0

    def test_completar_despues_de_un_desborde_suma_desde_0(self):
        # 5 puntos, se rompe 1 pieza en la casilla 9: queda en 0 y después
        # suma los 4 de la ventana 1. Sin el piso serían 5 - 18 + 4 = -9.
        g = nueva_partida()
        g.round_marker[0] = AM
        p = preparar(g, fabricas=[[AZ, AZ, AZ], [NA]], tiras={1: tira([AZ] * 5, [AZ] * 3)})
        p.score, p.broken_position = 5, 9
        rep = g.apply_move(Move.take(1, AZ, 1, AZ))
        self.assertEqual((rep.broken, rep.overflows, rep.window_points), (1, 1, 4))
        self.assertEqual(p.score, 4)

    def test_la_casilla_final_no_deja_el_puntaje_negativo(self):
        g = nueva_partida()
        p = preparar(g)
        p.score, p.broken_position = 3, 9
        g.end_game_scoring()
        self.assertEqual(p.score, 0)

    def test_el_piso_se_aplica_antes_de_sumar_los_ornamentos(self):
        # 3 - 8 queda en 0, y después +3 del ornamento: 3.
        g = nueva_partida()
        p = preparar(g)
        p.score, p.broken_position = 3, 6
        p.windows[0] = [AZ, None]
        p.windows[1] = [NA, None]
        g.end_game_scoring()
        self.assertEqual(p.score, 3)
        self.assertEqual(g.final_scores["Ana"].total, 3)


class CasosLimiteTest(unittest.TestCase):
    def test_desborde_por_la_loseta_y_piezas_rotas_en_el_mismo_turno(self):
        # Casilla 8: la loseta lleva a la 9, la primera pieza rota a la 10
        # (-18, vuelve a la 1) y la segunda a la 2.
        g = nueva_partida()
        p = preparar(g, fabricas=[[NA]], centro=[AZ, AZ])
        p.score, p.broken_position = 30, 8
        rep = g.apply_move(Move.take("centro", AZ, 1))
        self.assertEqual((rep.broken, rep.overflows), (2, 1))
        self.assertEqual((p.score, p.broken_position), (12, 2))

    def test_sin_espacio_en_las_alcanzables_vale_cualquier_alcanzable(self):
        g = nueva_partida()
        preparar(g, fabricas=[[NA, NA, AM]], tiras={1: tira([NA] * 5)}, vidriero=4)
        destinos = {m.strip for m in g.legal_moves() if m.source == 1 and m.color == NA}
        self.assertEqual(destinos, {4, 5, 6, 7, 8})

    def test_un_comodin_libre_cuenta_como_espacio_valido(self):
        g = nueva_partida()
        preparar(g, fabricas=[[RS]], tiras={2: tira([AZ, J, AZ, AZ, AZ])})
        antes = foto(g)
        with self.assertRaises(InvalidMoveError) as ctx:
            g.apply_move(Move.take(1, RS, 1))
        self.assertIn("pero sí en la(s) tira(s) 2", str(ctx.exception))
        self.assertEqual(foto(g), antes)

    def test_bolsa_y_torre_agotadas_llenan_las_que_alcanzan_y_se_sigue_jugando(self):
        g = nueva_partida()
        preparar(g, fabricas=[[AZ]], tiras={1: tira([AZ] * 5)})
        g.bag, g.tower = [AM, AM], [AZ, AZ, AZ]
        g.apply_move(Move.take(1, AZ, 1))
        # 2 de la bolsa + 3 de la torre + la pieza del marcador = 6
        self.assertEqual([len(f.tiles) for f in g.factories], [4, 2, 0, 0, 0])
        self.assertEqual((g.bag, g.tower), ([], []))
        self.assertEqual(g.round, 2)
        self.assertFalse(g.game_over)
        self.assertTrue(g.legal_moves())
        jugada = next(m for m in g.legal_moves() if m.source == 2)
        g.apply_move(jugada)
        self.assertEqual(g.round, 2)

    def test_sin_nada_en_la_bolsa_ni_la_torre_la_partida_llega_igual_al_final(self):
        # Desde la ronda 5 solo circula la pieza del marcador de la ronda
        # anterior: una fábrica con 1 pieza y las demás vacías.
        g = nueva_partida()
        g.round = 5
        g.round_marker = [None, None, None, None, AM, TR]
        preparar(g, fabricas=[[AZ]], tiras={1: tira([AZ] * 5), 2: tira([AM] * 5)})
        preparar(g, fabricas=[[AZ]], tiras={1: tira([TR] * 5)}, jugador=1)
        g.bag, g.tower = [], []
        g.apply_move(Move.take(1, AZ, 1))
        self.assertEqual(g.round, 6)
        self.assertEqual([f.tiles for f in g.factories], [[AM], [], [], [], []])
        g.apply_move(Move.take(1, AM, 2))
        self.assertTrue(g.game_over)

    def test_sacar_de_bolsa_y_torre_vacias_no_falla(self):
        g = nueva_partida()
        g.bag, g.tower = [], []
        self.assertEqual(g._draw_tiles_from_bag(4), [])


class PartidaTest(unittest.TestCase):
    def test_partidas_completas_con_jugadas_al_azar(self):
        for seed in range(12):
            azar = random.Random(seed)

            def elegir(game, player):
                return azar.choice(game.legal_moves())

            def revisar(report, game):
                self.assertEqual(contar_piezas(game), 100)
                self.assertTrue(all(p.score >= 0 for p in game.players))

            m = TwoPlayerMatch(["Ana", "Beto"], inputs={"Ana": elegir, "Beto": elegir},
                               seed=seed, on_turn=revisar)
            m.run()
            self.assertTrue(m.is_over)
            self.assertEqual(m.game.round, 6)
            self.assertEqual(m.game.round_marker, [None] * 6)
            self.assertEqual(set(m.game.final_scores), {"Ana", "Beto"})
            self.assertGreater(len(m.history), 30)

    def test_el_ejemplo_llega_al_final(self):
        import scripts.ejemplo_partida as ej
        m = TwoPlayerMatch(
            ["Ana", "Beto"],
            inputs={n: ej.jugador_fijo(n, lista) for n, lista in ej.JUGADAS.items()},
            first_player="Ana",
            seed=ej.SEMILLA,
        )
        self.assertEqual(m.run().name, "Beto")
        self.assertEqual([p.score for p in m.game.players], [39, 50])

    def test_los_turnos_se_alternan(self):
        m = TwoPlayerMatch(["Ana", "Beto"], seed=3, first_player="Beto")
        self.assertEqual(m.current_player.name, "Beto")
        m.play_turn(m.game.legal_moves()[0])
        self.assertEqual(m.current_player.name, "Ana")

    def test_una_jugada_invalida_no_cambia_nada(self):
        m = TwoPlayerMatch(["Ana", "Beto"], seed=3, first_player=0)
        antes = foto(m.game)
        with self.assertRaises(InvalidMoveError):
            m.play_turn(Move.reset_glazier())
        self.assertEqual(foto(m.game), antes)
        self.assertEqual(m.history, [])

    def test_on_invalid_vuelve_a_pedir_la_jugada(self):
        errores = []
        m = TwoPlayerMatch(["Ana", "Beto"], seed=3, first_player=0,
                           on_invalid=lambda player, move, err: errores.append(str(err)))
        jugadas = iter([Move.reset_glazier(), m.game.legal_moves()[0]])
        m.inputs = {"Ana": lambda game, player: next(jugadas)}
        m.ask_and_play()
        self.assertEqual(len(errores), 1)
        self.assertEqual(len(m.history), 1)
        self.assertEqual(m.current_player.name, "Beto")


class ConsolaTest(unittest.TestCase):
    def test_toda_jugada_valida_se_puede_escribir_y_leer(self):
        from scripts.jugar import describir, leer_jugada
        for seed in range(5):
            g = AzulVitralGame(["Ana", "Beto"], seed=seed)
            azar = random.Random(seed)
            while not g.game_over:
                jugadas = g.legal_moves()
                for m in jugadas:
                    self.assertEqual(leer_jugada(describir(m)), m)
                g.apply_move(azar.choice(jugadas))

    def test_formas_de_escribir(self):
        from scripts.jugar import leer_jugada
        self.assertEqual(leer_jugada("3 az 2"), Move.take(3, AZ, 2))
        self.assertEqual(leer_jugada("c Naranjo 4 na"), Move.take("centro", NA, 4, NA))
        self.assertEqual(leer_jugada("2 transp -"), Move.take(2, TR, None))
        self.assertEqual(leer_jugada("B"), Move.reset_glazier())

    def test_lo_que_no_se_entiende_se_rechaza(self):
        from scripts.jugar import leer_jugada
        for texto in ("", "3 azul", "x azul 2", "3 morado 2", "3 azul dos", "3 a 2"):
            with self.assertRaises(ValueError, msg=texto):
                leer_jugada(texto)


def estado_base():
    """Una partida en la ronda 3, escrita a mano como la escribiría un jugador."""
    return {
        "ronda": 3,
        "colores_ronda": ["AZUL", "ROSADO", "AMARILLO", "TRANSPARENTE"],
        "turno": "Beto",
        "loseta": "Ana",
        "fabricas": [["AZUL", "AZUL", "NARANJO", "ROSADO"], [], ["TRANSPARENTE"], [], []],
        "centro": ["AMARILLO", "AMARILLO"],
        "torre": [],
        "jugadores": [
            {"nombre": "Ana", "tablero": "rojo", "puntos": 12, "cristal_roto": 3, "vidriero": 2,
             "tiras": ["tr tr TR TR TR", "ROSADO na na na na", "-", "am am am am am",
                       "az am am am am", "rs rs tr tr tr", "na na na na na", "am rs rs rs rs"],
             "ventanas": [["TRANSPARENTE", None], [None, None], ["AZUL", "AZUL"],
                          [None, None], [None, None], [None, None], [None, None], [None, None]]},
            {"nombre": "Beto", "tablero": "amarillo", "puntos": 0, "cristal_roto": 9, "vidriero": 1,
             "tiras": ["NA ** rs *az am", "az am am am am", "az az az az az", "rs na na na na",
                       "am am am am am", "tr tr tr tr tr", "rs rs tr tr tr", "na na az az az"],
             "ventanas": [[None, None]] * 8},
        ],
    }


class EstadoTest(unittest.TestCase):
    def test_un_estado_escrito_a_mano_arma_la_partida(self):
        from tests.test_estado import partida_desde_estado
        g = partida_desde_estado(estado_base(), seed=1)
        ana, beto = g.players
        self.assertEqual((g.round, g.current_round_color(), g.get_current_player().name), (3, AZ, "Beto"))
        self.assertEqual(g.round_marker, [None, None, AZ, RS, AM, TR])
        self.assertTrue(ana.has_first_player_token)
        self.assertFalse(g.center.has_first_player_token)
        self.assertTrue(ana.strips[2].is_removed)
        self.assertEqual(ana.strips[0].filled_spaces, [None, None, TR, TR, TR])
        self.assertEqual(beto.strips[0].filled_spaces, [NA, None, None, AZ, None])
        self.assertEqual(beto.strips[0].pattern(), [NA, J, RS, J, AM])
        self.assertEqual((ana.glazier, beto.broken_position), (1, 9))
        self.assertEqual(contar_piezas(g), 100)
        self.assertTrue(g.legal_moves())

    def test_guardar_y_volver_a_cargar_no_cambia_nada(self):
        from tests.test_estado import estado_de_partida, partida_desde_estado
        for seed in range(6):
            g = AzulVitralGame(["Ana", "Beto"], seed=seed)
            azar = random.Random(seed)
            turno = 0
            while not g.game_over:
                if turno % 7 == 3:
                    estado = estado_de_partida(g)
                    copia = partida_desde_estado(estado, seed=seed)
                    self.assertEqual(estado_de_partida(copia), estado)
                    self.assertEqual(copia.legal_moves(), g.legal_moves())
                    self.assertEqual(Counter(copia.bag), Counter(g.bag))
                    self.assertEqual(copia.center.has_first_player_token, g.center.has_first_player_token)
                    self.assertEqual([q.has_first_player_token for q in copia.players],
                                     [q.has_first_player_token for q in g.players])
                    self.assertEqual(contar_piezas(copia), 100)
                g.apply_move(azar.choice(g.legal_moves()))
                turno += 1

    def test_una_partida_cargada_se_juega_hasta_el_final(self):
        from tests.test_estado import partida_desde_estado
        g = partida_desde_estado(estado_base(), seed=3)
        azar = random.Random(3)
        while not g.game_over:
            g.apply_move(azar.choice(g.legal_moves()))
            self.assertEqual(contar_piezas(g), 100)
        self.assertEqual(g.round_marker, [None] * 6)

    def test_estados_imposibles_se_rechazan_con_el_motivo(self):
        from tests.test_estado import partida_desde_estado

        def con(cambio):
            e = copy.deepcopy(estado_base())
            cambio(e)
            return e

        def tira(jugador, col, texto):
            def cambio(e):
                e["jugadores"][jugador]["tiras"][col - 1] = texto
            return cambio

        casos = [
            (tira(0, 2, "rs rs rs rs am"), "no es de ninguna tira"),
            (tira(0, 2, "tr tr tr tr tr"), "ya está en otra columna"),
            (tira(0, 2, "rs na na"), "5 espacios"),
            (tira(0, 2, "RS NA NA NA NA"), "está completa"),
            (tira(0, 2, "-"), "las dos cosas tienen que ir juntas"),
            (lambda e: e["jugadores"][0]["ventanas"].__setitem__(1, [None, "AZUL"]), "abajo sin una arriba"),
            (lambda e: e.__setitem__("colores_ronda", ["AZUL"]), "quedan 4 colores"),
            (lambda e: e.__setitem__("centro", ["AZUL"] * 20), "Azul a la vista"),
            (lambda e: (e.__setitem__("centro", []), e.__setitem__("fabricas", [[]] * 5)), "ya terminó"),
            (lambda e: e["fabricas"].__setitem__(1, ["AZUL"] * 5), "caben 4"),
            (lambda e: e.__setitem__("turno", "Carla"), "El turno"),
            (lambda e: e.__setitem__("loseta", "mesa"), "loseta"),
            (lambda e: e["jugadores"][1].__setitem__("tablero", "rojo"), "tablero distinto"),
            (lambda e: e["jugadores"][1].__setitem__("cristal_roto", 10), "de 1 a 9"),
            (lambda e: e["jugadores"][1].__setitem__("vidriero", 0), "de 1 a 8"),
            (lambda e: e["jugadores"][1].__setitem__("puntos", -2), "de 0 para arriba"),
        ]
        for cambio, motivo in casos:
            with self.assertRaises(ValueError, msg=motivo) as ctx:
                partida_desde_estado(con(cambio))
            self.assertIn(motivo, str(ctx.exception))

    def test_el_cuestionario_vuelve_a_preguntar_lo_que_no_sirve(self):
        from tests.test_estado import partida_desde_estado, preguntar_estado
        b = estado_base()
        ana, beto = b["jugadores"]
        respuestas = [
            "7", "3",                       # ronda: 7 no existe
            "az rs", "az rs am tr",         # faltan colores
            "Ana", "verde", "rojo", "12", "3", "2",
            "tr tr tr tr", *ana["tiras"],   # una tira con 4 espacios
            "3:az/az 1:tr",
            "Ana", "Beto", "rojo", "amarillo", "0", "9", "1", *beto["tiras"], "",
            "az az na rs", "-", "tr", "-", "-",
            "am am",
            "nadie", "Ana", "Beto", "",
        ]
        entradas = iter(respuestas)
        avisos = []
        estado = preguntar_estado(entrada=lambda _: next(entradas), salida=avisos.append)
        self.assertEqual(next(entradas, None), None)
        self.assertGreaterEqual(sum(1 for a in avisos if a.startswith("  ")), 7)
        g = partida_desde_estado(estado, seed=1)
        self.assertEqual(g.players[0].strips[0].filled_spaces, [None, None, TR, TR, TR])
        self.assertEqual(g.players[0].windows[2], [AZ, AZ])
        self.assertEqual(g.get_current_player().name, "Beto")

    def test_guardar_y_cargar_un_archivo(self):
        import os
        import tempfile
        from tests.test_estado import cargar_estado, guardar_estado
        with tempfile.TemporaryDirectory() as carpeta:
            ruta = os.path.join(carpeta, "partida.json")
            guardar_estado(estado_base(), ruta)
            self.assertEqual(cargar_estado(ruta), estado_base())


class MaquinaTest(unittest.TestCase):
    def test_la_maquina_siempre_elige_una_jugada_valida_y_termina_la_partida(self):
        from scripts.contra_maquina import jugador_al_azar
        with redirect_stdout(io.StringIO()):
            for seed in range(4):
                azar = jugador_al_azar(random.Random(seed))
                m = TwoPlayerMatch(["Ana", "Máquina"], inputs={"Ana": azar, "Máquina": azar}, seed=seed)
                m.run()
                self.assertTrue(m.is_over)

    def test_la_maquina_elige_con_la_misma_probabilidad_entre_todas(self):
        from scripts.contra_maquina import jugador_al_azar
        # Una partida de mentira que siempre ofrece las mismas jugadas, para
        # poder elegir muchas veces sin recalcularlas.
        g = nueva_partida()
        jugadas = g.legal_moves()

        class Fija:
            def legal_moves(self):
                return jugadas

        elegir = jugador_al_azar(random.Random(0))
        with redirect_stdout(io.StringIO()):
            vistas = Counter(elegir(Fija(), g.get_current_player()) for _ in range(len(jugadas) * 200))
        self.assertEqual(set(vistas), set(jugadas))
        esperado = 200
        self.assertTrue(all(abs(n - esperado) < 0.35 * esperado for n in vistas.values()), vistas)


class ImpresionTest(unittest.TestCase):
    def test_imprimir_no_cambia_el_estado(self):
        m = TwoPlayerMatch(["Ana", "Beto"], seed=11, first_player=0)
        azar = random.Random(1)
        while not m.is_over:
            antes = foto(m.game)
            with redirect_stdout(io.StringIO()):
                print_header(m.game)
                print_factories(m.game)
                print_center(m.game)
                for p in m.game.players:
                    print_player(m.game, p)
                print_game(m.game)
            self.assertEqual(foto(m.game), antes)
            rep = m.play_turn(azar.choice(m.game.legal_moves()))
            with redirect_stdout(io.StringIO()):
                print_turn(rep)
        antes = foto(m.game)
        with redirect_stdout(io.StringIO()):
            print_game(m.game)
            print_final(m.game)
        self.assertEqual(foto(m.game), antes)

    def test_el_estado_impreso_muestra_lo_pedido(self):
        g = nueva_partida()
        preparar(g, fabricas=[[AZ, AZ, NA, AM]], tiras={1: tira([AZ] * 5)})
        g.apply_move(Move.take(1, AZ, 1))
        texto = game_text(g)
        for parte in ("Fábricas", "Centro", "loseta inicial", "Ana", "Beto", "vidriero",
                      "ventana sup", "ventana inf", "cristal roto", "puntos", "[AZ]"):
            self.assertIn(parte, texto)


if __name__ == "__main__":
    unittest.main()
