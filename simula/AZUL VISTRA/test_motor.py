"""Pruebas del motor de reglas. Se corren desde la carpeta del proyecto con:

    python3 -m unittest -v
"""

import copy
import io
import random
import unittest
from collections import Counter
from contextlib import redirect_stdout

from azulvitral import (
    JOKER,
    AzulVitralGame,
    Color,
    InvalidMoveError,
    Move,
    PatternStrip,
    load_strips,
)
from impresion import (
    game_text,
    print_center,
    print_factories,
    print_final,
    print_game,
    print_header,
    print_player,
    print_turn,
)
from partida import TwoPlayerMatch

RO, AM, RS, AZ, TR = Color.ROJO, Color.AMARILLO, Color.ROSADO, Color.AZUL, Color.TRANSPARENTE
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
        p = preparar(g, fabricas=[[AZ, AZ, RO, AM]], tiras={1: tira([AZ] * 5)})
        rep = g.apply_move(Move.take(1, AZ, 1))
        self.assertEqual(g.factories[0].tiles, [])
        self.assertEqual(Counter(g.center.tiles), Counter([RO, AM]))
        self.assertEqual(p.strips[0].filled_spaces, [AZ, AZ, None, None, None])
        self.assertEqual((rep.placed, rep.broken), (2, 0))
        self.assertEqual(g.get_current_player().name, "Beto")

    def test_el_vidriero_se_mueve_a_la_tira_elegida(self):
        g = nueva_partida()
        p = preparar(g, fabricas=[[RO, RO, AM, AM]], tiras={4: tira([RO] * 5)})
        g.apply_move(Move.take(1, RO, 4))
        self.assertEqual(p.glazier, 3)

    def test_colocar_bajo_el_vidriero_no_lo_mueve(self):
        g = nueva_partida()
        p = preparar(g, fabricas=[[RO, AM]], tiras={3: tira([RO] * 5)}, vidriero=3)
        g.apply_move(Move.take(1, RO, 3))
        self.assertEqual(p.glazier, 2)

    def test_opcion_b_vuelve_a_la_tira_activa_mas_a_la_izquierda(self):
        g = nueva_partida()
        p = preparar(g, fabricas=[[RO]], vidriero=6,
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
        self.rechaza(g, Move.take(1, RO, 1), "no tiene piezas Rojo")

    def test_fabrica_vacia_o_inexistente(self):
        g = nueva_partida()
        preparar(g, fabricas=[[AZ], []])
        self.rechaza(g, Move.take(2, AZ, 1), "está vacía")
        self.rechaza(g, Move.take(6, AZ, 1), "Origen inválido")
        self.rechaza(g, Move.take(0, AZ, 1), "Origen inválido")

    def test_color_que_no_esta_en_el_centro(self):
        g = nueva_partida()
        preparar(g, fabricas=[[AZ]], centro=[AM])
        self.rechaza(g, Move.take("centro", RO, 1), "En el centro no hay piezas Rojo")

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
        self.rechaza(g, Move.take(1, AZ, 1, RO), "No hay ninguna pieza Rojo")

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
        p = preparar(g, fabricas=[[AZ, AZ, AM, AM]], tiras={1: tira([AZ, J, RO, J, AZ])})
        g.apply_move(Move.take(1, AZ, 1))
        self.assertEqual(p.strips[0].filled_spaces, [AZ, None, None, None, AZ])

    def test_el_comodin_recibe_lo_que_no_cabe_en_su_color(self):
        g = nueva_partida()
        p = preparar(g, fabricas=[[AZ, AZ, AZ, AM]], tiras={1: tira([AZ, J, RO, J, AZ])})
        rep = g.apply_move(Move.take(1, AZ, 1))
        self.assertEqual(p.strips[0].filled_spaces, [AZ, AZ, None, None, AZ])
        self.assertEqual(rep.broken, 0)

    def test_el_comodin_acepta_un_color_que_no_esta_en_la_tira(self):
        g = nueva_partida()
        p = preparar(g, fabricas=[[RS, RS, RS, AM]], tiras={1: tira([AZ, J, RO, J, AZ])})
        rep = g.apply_move(Move.take(1, RS, 1))
        self.assertEqual(p.strips[0].filled_spaces, [None, RS, None, RS, None])
        self.assertEqual(rep.broken, 1)


class PiezasRotasTest(unittest.TestCase):
    def test_las_que_no_caben_van_a_la_torre_y_avanzan_el_track(self):
        g = nueva_partida()
        p = preparar(g, fabricas=[[AZ] * 4, [RO]], tiras={1: tira([AZ, AZ, RO, RO, RO])})
        torre = len(g.tower)
        rep = g.apply_move(Move.take(1, AZ, 1))
        self.assertEqual((rep.placed, rep.broken), (2, 2))
        self.assertEqual(len(g.tower), torre + 2)
        self.assertEqual(p.broken_position, 3)

    def test_sin_espacio_en_ninguna_tira_alcanzable_se_rompe_todo(self):
        # La tira 1 tiene espacio pero está a la izquierda del vidriero.
        g = nueva_partida()
        p = preparar(g, fabricas=[[RO, RO, RO, AM]], tiras={1: tira([RO] * 5)}, vidriero=2)
        rep = g.apply_move(Move.take(1, RO, 5))
        self.assertEqual((rep.placed, rep.broken), (0, 3))
        self.assertEqual(p.glazier, 4)
        self.assertEqual(p.broken_position, 4)

    def test_sin_tiras_alcanzables_se_toma_sin_tira_y_se_rompe_todo(self):
        g = nueva_partida()
        p = preparar(g, fabricas=[[RO, RO, AM]], tiras={8: tira([RO] * 5, retirada=True)}, vidriero=8)
        antes = foto(g)
        with self.assertRaises(InvalidMoveError):
            g.apply_move(Move.take(1, RO, 8))
        self.assertEqual(foto(g), antes)
        rep = g.apply_move(Move.take(1, RO, None))
        self.assertEqual(rep.broken, 2)
        self.assertEqual(p.glazier, 7)
        self.assertEqual(p.broken_position, 3)

    def test_la_primera_toma_del_centro_lleva_la_loseta_y_un_cristal_roto(self):
        g = nueva_partida()
        ana = preparar(g, fabricas=[[RO] * 4], centro=[AZ, AM], tiras={1: tira([AZ] * 5)})
        beto = preparar(g, fabricas=[[RO] * 4], centro=[AZ, AM], tiras={1: tira([AM] * 5)}, jugador=1)
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
        g.round_marker[0] = RO
        p = preparar(g, fabricas=[[RO, RO, AM]], tiras={1: tira([AZ, AZ, RO, RO, J], [AZ, AZ, RO])})
        torre = len(g.tower)
        rep = g.apply_move(Move.take(1, RO, 1, RO))
        # Tres rojas (una sobre el comodín) = +3, más los 4 de la ventana 1.
        self.assertEqual((rep.color_bonus, rep.window_points), (3, 4))
        self.assertEqual(p.score, 7)
        self.assertEqual(p.windows[0], [RO, None])
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
        g.round_marker[0] = RO
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
        p = preparar(g, fabricas=[[AZ] * cantidad, [RO]])
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
        self.assertEqual(p.overflow_points_lost, 18)

    def test_la_loseta_inicial_tambien_puede_desbordar(self):
        g = nueva_partida()
        p = preparar(g, fabricas=[[RO]], centro=[AZ], tiras={1: tira([AZ] * 5)})
        p.broken_position = 9
        g.apply_move(Move.take("centro", AZ, 1))
        self.assertEqual((p.broken_position, p.score), (1, -18))

    def test_dos_desbordes_en_una_jugada(self):
        g = nueva_partida()
        p = g.players[0]
        self.assertEqual(g._add_broken_glass(p, 18), 2)
        self.assertEqual((p.broken_position, p.score, p.overflow_points_lost), (1, -36, 36))

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
        preparar(g, fabricas=[[RO]], centro=[AZ], tiras={1: tira([RO] * 5)})
        preparar(g, fabricas=[[RO]], centro=[AZ], tiras={1: tira([AZ] * 5)}, jugador=1)
        g.apply_move(Move.take(1, RO, 1))
        g.apply_move(Move.take("centro", AZ, 1))
        self.assertEqual(g.round, 2)
        self.assertEqual(g.get_current_player().name, "Beto")
        self.assertFalse(any(p.has_first_player_token for p in g.players))

    def test_si_nadie_toma_del_centro_vuelve_a_empezar_el_mismo(self):
        g = AzulVitralGame(["Ana", "Beto"], seed=7, first_player=1)
        preparar(g, fabricas=[[AZ], [RO]], tiras={1: tira([RO] * 5)})
        preparar(g, fabricas=[[AZ], [RO]], tiras={1: tira([AZ] * 5)}, jugador=1)
        g.apply_move(Move.take(1, AZ, 1))
        g.apply_move(Move.take(2, RO, 1))
        self.assertEqual(g.round, 2)
        self.assertEqual(g.get_current_player().name, "Beto")

    def test_la_bolsa_se_rellena_con_la_torre_y_si_no_alcanza_quedan_incompletas(self):
        g = nueva_partida()
        preparar(g, fabricas=[[AZ]], tiras={1: tira([AZ] * 5)})
        g.bag = [AM, AM]
        g.tower = [RO] * 9
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
            2: tira([RO] * 5, [RO, RO]),
            3: tira([AM] * 5, [AM, AM, AM]),
        })
        g.end_game_scoring()
        detalle = g.final_scores["Ana"]
        self.assertEqual((detalle.leftover_tiles, detalle.leftover_points), (7, 2))

    def test_resta_la_casilla_del_cristal_roto(self):
        g = nueva_partida()
        p = preparar(g)
        p.broken_position = 6
        g.end_game_scoring()
        self.assertEqual(p.score, -8)

    def test_ornamentos(self):
        g = nueva_partida()
        p = preparar(g)
        p.windows = [
            [AZ, AZ], [RO, RO],      # ornamento 1: 4 de 4
            [AM, AM], [RS, None],    # ornamento 2: 3 de 4
            [TR, None], [AZ, None],  # ornamento 3: 2 de 4
            [RO, None], [None, None],  # ornamento 4: 1 de 4
        ]
        g.end_game_scoring()
        self.assertEqual(g.final_scores["Ana"].ornament_points, [10, 6, 3, 0])
        self.assertEqual(p.score, 19)

    def test_todo_junto(self):
        g = nueva_partida()
        # Asumimos que preparar(...) devuelve el jugador actual (p)
        p = preparar(g, tiras={1: tira([AZ] * 5, [AZ] * 3)})

        p.score = 20
        p.broken_position = 2  # Casilla 2 -> penalización de -1 pto
        p.windows[6] = [RO, None]  # 1 cristal en ventana 6
        p.windows[7] = [AM, None]  # 1 cristal en ventana 7

        # Ejecuta el cálculo y aplicación de puntuación final de la partida
        g.end_game_scoring()

    # Desglose del cálculo esperado:
    #   20 (score inicial)
    # +  1 (3 cristales sobrantes en la tira 1 -> 3 // 3 = 1 pto)
    # + (-1) (penalización de la casilla 2 de cristales rotos)
    # +  3 (adorno entre ventanas 6 y 7 rodeado por 2 cristales)
    # = 23
        self.assertEqual(p.score, 20 + 1 + p.broken_penalty() + 3)

class GanadorTest(unittest.TestCase):
    def terminada(self, puntos, posiciones, desbordes):
        g = nueva_partida()
        for p, pts, pos, perdido in zip(g.players, puntos, posiciones, desbordes):
            p.score, p.broken_position, p.overflow_points_lost = pts, pos, perdido
        g.game_over = True
        return g

    def test_gana_el_que_tiene_mas_puntos(self):
        g = self.terminada((40, 35), (1, 1), (0, 0))
        self.assertEqual(g.winner().name, "Ana")

    def test_empate_gana_el_que_perdio_menos_en_el_cristal_roto(self):
        # Ana está en la casilla 2 (-1) pero ya perdió 18 en un desborde: 19.
        # Beto está en la 7 (-10) sin desbordes: 10. Gana Beto.
        g = self.terminada((30, 30), (2, 7), (18, 0))
        self.assertEqual(g.winner().name, "Beto")

    def test_empate_tecnico(self):
        g = self.terminada((30, 30), (3, 3), (0, 0))
        self.assertIsNone(g.winner())

    def test_no_hay_ganador_antes_del_final(self):
        with self.assertRaises(RuntimeError):
            nueva_partida().winner()


class PartidaTest(unittest.TestCase):
    def test_partidas_completas_con_jugadas_al_azar(self):
        for seed in range(12):
            azar = random.Random(seed)

            def elegir(game, player):
                return azar.choice(game.legal_moves())

            def revisar(report, game):
                self.assertEqual(contar_piezas(game), 100)

            m = TwoPlayerMatch(["Ana", "Beto"], inputs={"Ana": elegir, "Beto": elegir},
                               seed=seed, on_turn=revisar)
            m.run()
            self.assertTrue(m.is_over)
            self.assertEqual(m.game.round, 6)
            self.assertEqual(m.game.round_marker, [None] * 6)
            self.assertEqual(set(m.game.final_scores), {"Ana", "Beto"})
            self.assertGreater(len(m.history), 30)

    def test_el_ejemplo_llega_al_final(self):
        import ejemplo_partida as ej
        m = TwoPlayerMatch(
            ["Ana", "Beto"],
            inputs={n: ej.jugador_fijo(n, lista) for n, lista in ej.JUGADAS.items()},
            first_player="Ana",
            seed=ej.SEMILLA,
        )
        self.assertEqual(m.run().name, "Beto")
        self.assertEqual([p.score for p in m.game.players], [39, 48])

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
        preparar(g, fabricas=[[AZ, AZ, RO, AM]], tiras={1: tira([AZ] * 5)})
        g.apply_move(Move.take(1, AZ, 1))
        texto = game_text(g)
        for parte in ("Fábricas", "Centro", "loseta inicial", "Ana", "Beto", "vidriero",
                      "ventana sup", "ventana inf", "cristal roto", "puntos", "[AZ]"):
            self.assertIn(parte, texto)


if __name__ == "__main__":
    unittest.main()
