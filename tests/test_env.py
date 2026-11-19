import random
import sys
import unittest
from pathlib import Path
import numpy as np
from pettingzoo.test import api_test
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.azulvitral import AzulVitralGame, Color, Move
from src.env_azul_v2 import (
    NUM_ACTIONS,
    OBS_DIM,
    AzulGymWrapperV2,
    AzulSintraEnvV2,
    build_observation_v2,
    decode_action,
    encode_move,
    heuristic_move,
)
from src.env_azul_v3 import AzulGymWrapperV3, AzulSintraEnvV3


class CodificacionAccionesTest(unittest.TestCase):
    def test_biyeccion_completa_1621_acciones(self):
        for idx in range(NUM_ACTIONS):
            move = decode_action(idx)
            self.assertIsInstance(move, Move)
            self.assertEqual(encode_move(move), idx, f"Fallo de biyección en índice {idx}")

    def test_codificacion_opcion_b(self):
        reset_move = Move.reset_glazier()
        self.assertEqual(encode_move(reset_move), 0)
        self.assertEqual(decode_action(0), reset_move)

    def test_todas_las_jugadas_legales_son_codificables(self):
        for seed in range(6):
            game = AzulVitralGame(["Ana", "Beto"], seed=seed)
            rng = random.Random(seed)
            while not game.game_over:
                legales = game.legal_moves()
                indices = set()
                for move in legales:
                    idx = encode_move(move)
                    self.assertGreaterEqual(idx, 0)
                    self.assertLess(idx, NUM_ACTIONS)
                    self.assertEqual(decode_action(idx), move)
                    indices.add(idx)
                self.assertEqual(len(indices), len(legales))
                game.apply_move(rng.choice(legales))


class ObservacionV2Test(unittest.TestCase):
    def test_dimension_tipo_y_rango_observacion(self):
        env = AzulSintraEnvV2()
        env.reset(seed=10)
        space = env.observation_space("player_0")["observation"]
        rng = random.Random(10)

        while not env.game.game_over:
            for ag in env.possible_agents:
                obs_dict = env.observe(ag)
                obs = obs_dict["observation"]
                self.assertEqual(obs.shape, (OBS_DIM,))
                self.assertEqual(obs.dtype, np.float32)
                self.assertFalse(np.isnan(obs).any())
                self.assertFalse(np.isinf(obs).any())
                self.assertTrue(space.contains(obs))
            act = encode_move(rng.choice(env.game.legal_moves()))
            env.step(act)

    def test_perspectiva_egocentrica(self):
        game = AzulVitralGame(["Ana", "Beto"], seed=5, first_player=0)
        game.players[0].score = 24
        game.players[1].score = 11
        game.players[0].glazier = 2
        game.players[1].glazier = 6

        obs_p0 = build_observation_v2(game, 0)
        obs_p1 = build_observation_v2(game, 1)

        # El último elemento indica si el agente observado es player_0 (1.0) o player_1 (0.0)
        self.assertEqual(obs_p0[-1], 1.0)
        self.assertEqual(obs_p1[-1], 0.0)
        self.assertFalse(np.array_equal(obs_p0, obs_p1))

        # En el bloque egocéntrico (índice 69 = primer elemento de jugador propio), aparece su puntaje
        self.assertAlmostEqual(obs_p0[69], 0.24, places=5)
        self.assertAlmostEqual(obs_p1[69], 0.11, places=5)


class PettingZooAECEnvV2Test(unittest.TestCase):
    def test_pettingzoo_api_oficial(self):
        env = AzulSintraEnvV2()
        api_test(env, num_cycles=250, verbose_progress=False)

    def test_mascara_coincide_exactamente_con_legal_moves(self):
        env = AzulSintraEnvV2()
        for seed in range(5):
            env.reset(seed=seed)
            rng = random.Random(seed)
            while not env.game.game_over:
                activo = env.agent_selection
                inactivo = "player_1" if activo == "player_0" else "player_0"

                mask_activo = env.action_mask(activo)
                mask_inactivo = env.action_mask(inactivo)

                indices_mascara = set(np.where(mask_activo == 1)[0].tolist())
                indices_legales = {encode_move(m) for m in env.game.legal_moves()}

                self.assertEqual(indices_mascara, indices_legales)
                self.assertEqual(int(mask_inactivo.sum()), 0)

                env.step(rng.choice(list(indices_legales)))

            # Al terminar la partida, ambas máscaras deben estar en cero
            for ag in env.possible_agents:
                self.assertEqual(int(env.action_mask(ag).sum()), 0)

    def test_recompensas_terminales_y_fin_de_episodio(self):
        env = AzulSintraEnvV2()
        env.reset(seed=2026)
        rng = random.Random(2026)

        while not env.game.game_over:
            mask = env.action_mask(env.agent_selection)
            validos = np.where(mask == 1)[0]
            env.step(int(rng.choice(validos)))

        self.assertTrue(all(env.terminations.values()))
        winner = env.game.winner()
        if winner is not None:
            w_idx = env.game.players.index(winner)
            l_idx = 1 - w_idx
            self.assertGreater(env.rewards[f"player_{w_idx}"], 30.0)
            self.assertLess(env.rewards[f"player_{l_idx}"], -30.0)


class GymnasiumWrapperV2Test(unittest.TestCase):
    def test_interfaz_gymnasium_y_episodios_completos(self):
        env = AzulGymWrapperV2()
        for seed in range(15):
            obs, info = env.reset(seed=seed)
            self.assertIsInstance(info, dict)
            self.assertTrue(env.observation_space.contains(obs))

            rng = random.Random(seed)
            terminated = False
            truncated = False
            pasos = 0

            while not (terminated or truncated):
                mask = env.action_masks()
                self.assertEqual(mask.dtype, bool)
                self.assertEqual(mask.shape, (NUM_ACTIONS,))
                validos = np.where(mask)[0]
                self.assertGreater(len(validos), 0)

                obs, reward, terminated, truncated, info = env.step(int(rng.choice(validos)))
                self.assertTrue(env.observation_space.contains(obs))
                self.assertIsInstance(reward, float)
                self.assertIsInstance(terminated, bool)
                self.assertIsInstance(truncated, bool)
                self.assertIsInstance(info, dict)
                pasos += 1

            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertEqual(env.aec_env.game.round, 6)
            self.assertGreaterEqual(pasos, 12)

    def test_reproducibilidad_con_semilla(self):
        env1 = AzulGymWrapperV2()
        env2 = AzulGymWrapperV2()

        obs1, _ = env1.reset(seed=777)
        obs2, _ = env2.reset(seed=777)

        np.testing.assert_array_equal(obs1, obs2)
        np.testing.assert_array_equal(env1.action_masks(), env2.action_masks())

    def test_integracion_con_action_masker(self):
        try:
            from sb3_contrib.common.wrappers import ActionMasker
        except ImportError:
            self.skipTest("sb3_contrib no está instalado en este entorno.")

        base_env = AzulGymWrapperV2()
        masked_env = ActionMasker(base_env, lambda e: e.action_masks())
        obs, _ = masked_env.reset(seed=42)
        self.assertEqual(obs.shape, (OBS_DIM,))

        rng = random.Random(42)
        done = False
        while not done:
            validos = np.where(masked_env.action_masks())[0]
            obs, _, terminated, truncated, _ = masked_env.step(int(rng.choice(validos)))
            done = terminated or truncated


class HeuristicaTacticaTest(unittest.TestCase):
    def test_heuristica_siempre_legal_y_supera_azar(self):
        rng = random.Random(99)
        victorias_h = 0
        pts_h, pts_r = [], []

        for seed in range(20):
            game = AzulVitralGame(["Heuristica", "Azar"], seed=seed, first_player=seed % 2)
            while not game.game_over:
                legales = game.legal_moves()
                if game.current == 0:
                    move = heuristic_move(game, rng)
                    self.assertIn(move, legales)
                else:
                    move = rng.choice(legales)
                game.apply_move(move)

            pts_h.append(game.players[0].score)
            pts_r.append(game.players[1].score)
            if game.winner() == game.players[0]:
                victorias_h += 1

        self.assertGreaterEqual(victorias_h, 18)
        self.assertGreater(np.mean(pts_h), np.mean(pts_r) + 25.0)


class EnvAzulV3Test(unittest.TestCase):
    def test_pettingzoo_api_oficial_v3(self):
        env = AzulSintraEnvV3()
        api_test(env, num_cycles=250, verbose_progress=False)

    def test_mascara_y_observacion_v3(self):
        env = AzulSintraEnvV3()
        for seed in range(5):
            env.reset(seed=seed)
            rng = random.Random(seed)
            while not env.game.game_over:
                activo = env.agent_selection
                inactivo = "player_1" if activo == "player_0" else "player_0"

                obs_dict = env.observe(activo)
                obs = obs_dict["observation"]
                self.assertEqual(obs.shape, (OBS_DIM,))
                self.assertEqual(obs.dtype, np.float32)
                self.assertFalse(np.isnan(obs).any())
                self.assertTrue(env.observation_space(activo)["observation"].contains(obs))

                mask_activo = env.action_mask(activo)
                mask_inactivo = env.action_mask(inactivo)
                indices_mascara = set(np.where(mask_activo == 1)[0].tolist())
                indices_legales = {encode_move(m) for m in env.game.legal_moves()}

                self.assertEqual(indices_mascara, indices_legales)
                self.assertEqual(int(mask_inactivo.sum()), 0)

                env.step(rng.choice(list(indices_legales)))

            for ag in env.possible_agents:
                self.assertEqual(int(env.action_mask(ag).sum()), 0)

    def test_shaping_cristal_roto_y_recompensas_terminales_v3(self):
        env = AzulSintraEnvV3()
        env.reset(seed=2026)
        rng = random.Random(2026)

        while not env.game.game_over:
            mask = env.action_mask(env.agent_selection)
            validos = np.where(mask == 1)[0]
            env.step(int(rng.choice(validos)))

        self.assertTrue(all(env.terminations.values()))
        winner = env.game.winner()
        if winner is not None:
            w_idx = env.game.players.index(winner)
            l_idx = 1 - w_idx
            self.assertGreater(env.rewards[f"player_{w_idx}"], 35.0)
            self.assertLess(env.rewards[f"player_{l_idx}"], -35.0)

    def test_gymnasium_wrapper_v3_modos_y_episodios(self):
        for modo in ("random", "heuristic", "curriculum"):
            env = AzulGymWrapperV3(opponent_mode=modo)
            obs, info = env.reset(seed=123)
            self.assertIn("opponent", info)
            self.assertIn("seat", info)
            self.assertEqual(obs.shape, (OBS_DIM,))
            self.assertTrue(env.observation_space.contains(obs))

            rng = random.Random(123)
            terminated = False
            truncated = False
            pasos = 0

            while not (terminated or truncated):
                mask = env.action_masks()
                self.assertEqual(mask.dtype, bool)
                self.assertEqual(mask.shape, (NUM_ACTIONS,))
                validos = np.where(mask)[0]
                self.assertGreater(len(validos), 0)

                obs, reward, terminated, truncated, info = env.step(int(rng.choice(validos)))
                self.assertTrue(env.observation_space.contains(obs))
                self.assertIsInstance(reward, float)
                self.assertIsInstance(terminated, bool)
                self.assertIsInstance(truncated, bool)
                pasos += 1

            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertEqual(env.aec_env.game.round, 6)
            self.assertGreaterEqual(pasos, 12)

    def test_reproducibilidad_v3_e_integracion_action_masker(self):
        env1 = AzulGymWrapperV3(opponent_mode="random")
        env2 = AzulGymWrapperV3(opponent_mode="random")

        obs1, _ = env1.reset(seed=888)
        obs2, _ = env2.reset(seed=888)
        np.testing.assert_array_equal(obs1, obs2)
        np.testing.assert_array_equal(env1.action_masks(), env2.action_masks())

        try:
            from sb3_contrib.common.wrappers import ActionMasker
        except ImportError:
            self.skipTest("sb3_contrib no está instalado en este entorno.")

        masked_env = ActionMasker(AzulGymWrapperV3(opponent_mode="heuristic"), lambda e: e.action_masks())
        obs, _ = masked_env.reset(seed=42)
        self.assertEqual(obs.shape, (OBS_DIM,))
        rng = random.Random(42)
        done = False
        while not done:
            validos = np.where(masked_env.action_masks())[0]
            obs, _, terminated, truncated, _ = masked_env.step(int(rng.choice(validos)))
            done = terminated or truncated


if __name__ == "__main__":
    unittest.main()
