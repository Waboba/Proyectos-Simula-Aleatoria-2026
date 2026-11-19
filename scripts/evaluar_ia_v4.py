"""Evaluación experimental determinista de IA v4 (models_checkpoint/ppo_azul_sintra_v4.zip)
frente a Aleatorio, IA v1, IA v2, IA v3 y Heurística Táctica.

Alterna estrictamente el asiento (0 y 1) en 50% / 50% de las partidas para
eliminar el sesgo de jugador inicial.
"""

from __future__ import annotations

import argparse
import importlib
import math
import random
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.azulvitral import AzulVitralGame, Move
from src.env_azul_v2 import NUM_ACTIONS, decode_action, encode_move, heuristic_move
from src.env_azul_v3 import AzulSintraEnvV3
from src.env_azul_v4 import (
    AgenteHibridoV4,
    aplicar_jugada_segura,
    cargar_modelo_maskable_ppo,
    partida_terminada,
)

CHECKPOINT_DIR = PROJECT_ROOT / "models_checkpoint"
DEFAULT_MODEL_V4 = "models_checkpoint/ppo_azul_sintra_v4.zip"


def _politica_ppo_reactiva(modelo) -> Callable[[AzulVitralGame, int, random.Random], Move]:
    obs_dim = int(modelo.observation_space.shape[0])
    act_dim = int(modelo.action_space.n)

    # Modelos v2 / v3 (288 observaciones, 1621 acciones)
    if obs_dim == 288:
        helper_v3 = AzulSintraEnvV3()

        def _elegir_288(game: AzulVitralGame, player_idx: int, rng: random.Random) -> Move:
            helper_v3.game = game
            helper_v3.agents = ["player_0", "player_1"]
            helper_v3.terminations = {"player_0": False, "player_1": False}
            ag = f"player_{player_idx}"
            helper_v3.agent_selection = ag

            obs_dict = helper_v3.observe(ag)
            act, _ = modelo.predict(
                obs_dict["observation"],
                action_masks=obs_dict["action_mask"].astype(bool),
                deterministic=True,
            )
            return decode_action(int(act))

        return _elegir_288

    # Modelo v1 (253 observaciones en src/env_azul.py)
    mod_v1 = importlib.import_module("src.env_azul")
    env_cls_v1 = getattr(mod_v1, "AzulSintraEnv", None)
    decode_v1 = getattr(mod_v1, "decode_action", decode_action)
    encode_v1 = getattr(mod_v1, "encode_move", encode_move)
    helper_v1 = env_cls_v1() if env_cls_v1 is not None else None

    def _elegir_v1(game: AzulVitralGame, player_idx: int, rng: random.Random) -> Move:
        ag = f"player_{player_idx}"
        if helper_v1 is not None:
            helper_v1.game = game
            helper_v1.agents = ["player_0", "player_1"]
            helper_v1.terminations = {"player_0": False, "player_1": False}
            helper_v1.agent_selection = ag
            raw = helper_v1.observe(ag)
            if isinstance(raw, dict):
                obs = np.asarray(raw["observation"], dtype=np.float32)
                mask = np.asarray(raw["action_mask"], dtype=bool)
            else:
                obs = np.asarray(raw, dtype=np.float32)
                mask = np.zeros(act_dim, dtype=bool)
                for mv in game.legal_moves():
                    idx = encode_v1(mv)
                    if 0 <= idx < act_dim:
                        mask[idx] = True
        else:
            helper_fb = AzulSintraEnvV3()
            helper_fb.game = game
            helper_fb.agents = ["player_0", "player_1"]
            helper_fb.terminations = {"player_0": False, "player_1": False}
            helper_fb.agent_selection = ag
            raw = helper_fb.observe(ag)
            obs = np.asarray(raw["observation"], dtype=np.float32)[:obs_dim]
            mask = np.asarray(raw["action_mask"], dtype=bool)[:act_dim]

        act, _ = modelo.predict(obs, action_masks=mask, deterministic=True)
        jugada = decode_v1(int(act))
        legales = game.legal_moves()
        return jugada if jugada in legales else legales[0]

    return _elegir_v1


def _construir_rivales() -> Dict[str, Callable[[AzulVitralGame, int, random.Random], Move]]:
    rivales: Dict[str, Callable[[AzulVitralGame, int, random.Random], Move]] = {
        "Aleatorio": lambda g, p, rng: rng.choice(g.legal_moves()),
        "Heurística Táctica": lambda g, p, rng: heuristic_move(g, rng),
    }

    for etiqueta, archivo in [
        ("IA v1", "ppo_azul_sintra_v1.zip"),
        ("IA v2", "ppo_azul_sintra_v2.zip"),
        ("IA v3", "ppo_azul_sintra_v3.zip"),
    ]:
        ruta = CHECKPOINT_DIR / archivo
        if ruta.exists():
            modelo_rival = cargar_modelo_maskable_ppo(ruta, device="cpu")
            rivales[etiqueta] = _politica_ppo_reactiva(modelo_rival)

    return rivales


def _simular_serie_balanceada(
    politica_ia: Callable[[AzulVitralGame, int], Move],
    politica_rival: Callable[[AzulVitralGame, int, random.Random], Move],
    num_partidas: int = 100,
    seed_base: int = 10_000,
) -> Tuple[float, float, float, float, float]:
    victorias_ia = 0
    victorias_rival = 0
    puntos_ia: List[float] = []
    puntos_rival: List[float] = []

    for i in range(num_partidas):
        ia_seat = i % 2
        seed_partida = seed_base + (i // 2) * 17
        rng_partida = random.Random(seed_partida + 999)

        nombres = ["IA_v4", "Rival"] if ia_seat == 0 else ["Rival", "IA_v4"]
        game = AzulVitralGame(player_names=nombres, seed=seed_partida)

        while not partida_terminada(game):
            actual = game.get_current_player()
            turno_idx = next(idx for idx, p in enumerate(game.players) if p is actual)
            if turno_idx == ia_seat:
                mv = politica_ia(game, turno_idx)
            else:
                mv = politica_rival(game, turno_idx, rng_partida)
            aplicar_jugada_segura(game, turno_idx, mv)

        s_ia = float(game.players[ia_seat].score)
        s_rv = float(game.players[1 - ia_seat].score)
        puntos_ia.append(s_ia)
        puntos_rival.append(s_rv)

        if s_ia > s_rv:
            victorias_ia += 1
        elif s_rv > s_ia:
            victorias_rival += 1

    return (
        100.0 * victorias_ia / num_partidas,
        100.0 * victorias_rival / num_partidas,
        float(np.mean(puntos_ia)),
        float(np.std(puntos_ia)),
        float(np.mean(puntos_rival)),
    )


def _elo_diferencial(win_rate: float) -> float:
    p = min(max(win_rate / 100.0, 1e-4), 1.0 - 1e-4)
    return -400.0 * math.log10((1.0 / p) - 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evalúa IA v4 en series balanceadas contra todos los rivales."
    )
    parser.add_argument(
        "--modelo",
        type=str,
        default=DEFAULT_MODEL_V4,
        help="Ruta al checkpoint de IA v4 (por defecto models_checkpoint/ppo_azul_sintra_v4.zip).",
    )
    parser.add_argument(
        "--partidas",
        type=int,
        default=100,
        help="Número de partidas balanceadas por enfrentamiento.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=6,
        help="Parámetro Top-K para Lookahead 1-ply (default: 6).",
    )
    args = parser.parse_args()

    agente_v4 = AgenteHibridoV4(ruta_modelo=args.modelo, k_lookahead=args.k)
    rivales = _construir_rivales()

    print("=" * 84)
    print(f"EVALUACIÓN DETERMINISTA DE IA v4 (K={args.k}) | N={args.partidas} partidas/rival")
    print(f"Checkpoint: {agente_v4.ruta_modelo}")
    print("=" * 84)
    print(
        f"{'Rival':<22} | {'Vic. IA':>8} | {'Vic. Rival':>10} | "
        f"{'Puntaje IA':>15} | {'Puntaje Rival':>13} | {'ΔElo':>7}"
    )
    print("-" * 84)

    for nombre_rival, fn_rival in rivales.items():
        wr_ia, wr_rv, mean_ia, std_ia, mean_rv = _simular_serie_balanceada(
            politica_ia=agente_v4.elegir_jugada,
            politica_rival=fn_rival,
            num_partidas=args.partidas,
        )
        delta_elo = _elo_diferencial(wr_ia)
        print(
            f"{nombre_rival:<22} | {wr_ia:7.1f}% | {wr_rv:9.1f}% | "
            f"{mean_ia:6.2f} ± {std_ia:5.2f} | {mean_rv:13.2f} | {delta_elo:+7.0f}"
        )
    print("=" * 84)


if __name__ == "__main__":
    main()
