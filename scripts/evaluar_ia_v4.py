"""
Evaluación experimental determinista de IA v4 frente a:
  - Aleatorio
  - IA v1, IA v2, IA v3 (si sus checkpoints están presentes)
  - Heurística Táctica

Alterna estrictamente el asiento (player_0 / player_1) y el orden de inicio
en 50% / 50% de las partidas para eliminar el sesgo de primer jugador.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

from src.azulvitral import AzulVitralGame, Move
from src.env_azul_v2 import build_observation_v2, encode_move, heuristic_move
from src.env_azul_v4 import (
    AgenteHibridoV4,
    NUM_ACTIONS,
    cargar_modelo_maskable_ppo,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "models_checkpoint"
DEFAULT_MODEL_V4 = "models_checkpoint/ppo_azul_sintra_v4.zip"


def _politica_ppo_reactiva(modelo) -> Callable[[AzulVitralGame, int], Move]:
    def _elegir(game: AzulVitralGame, player_idx: int) -> Move:
        legal_moves = game.get_legal_moves(player_idx)
        if not legal_moves:
            return Move.reset_glazier()
        mask = np.zeros(NUM_ACTIONS, dtype=bool)
        idx_map = {}
        for mv in legal_moves:
            idx = encode_move(mv)
            mask[idx] = True
            idx_map[idx] = mv
        obs = build_observation_v2(game, player_idx).astype(np.float32)
        act, _ = modelo.predict(obs, action_masks=mask, deterministic=True)
        return idx_map.get(int(act), legal_moves[0])

    return _elegir


def _construir_rivales() -> Dict[str, Callable[[AzulVitralGame, int], Move]]:
    rivales: Dict[str, Callable[[AzulVitralGame, int], Move]] = {
        "Aleatorio": lambda g, p: random.Random(
            g.round_num * 997 + p * 31 + len(g.center)
        ).choice(g.get_legal_moves(p)),
        "Heurística Táctica": heuristic_move,
    }

    for etiqueta, nombre_archivo in [
        ("IA v1", "ppo_azul_sintra_v1.zip"),
        ("IA v2", "ppo_azul_sintra_v2.zip"),
        ("IA v3", "ppo_azul_sintra_v3.zip"),
    ]:
        ruta = CHECKPOINT_DIR / nombre_archivo
        if ruta.exists():
            modelo_rival = cargar_modelo_maskable_ppo(ruta, device="cpu")
            rivales[etiqueta] = _politica_ppo_reactiva(modelo_rival)

    return rivales


def _simular_serie_balanceada(
    politica_ia: Callable[[AzulVitralGame, int], Move],
    politica_rival: Callable[[AzulVitralGame, int], Move],
    num_partidas: int = 100,
    seed_base: int = 10_000,
) -> Tuple[float, float, float, float, float]:
    victorias_ia = 0
    victorias_rival = 0
    puntos_ia: List[float] = []
    puntos_rival: List[float] = []

    for i in range(num_partidas):
        # La mitad de las partidas la IA ocupa asiento 0 y la otra mitad asiento 1
        ia_seat = i % 2
        seed_partida = seed_base + (i // 2) * 17
        game = AzulVitralGame(num_players=2, seed=seed_partida)

        while not game.is_game_over():
            turno_idx = game.current_player_idx
            if turno_idx == ia_seat:
                mv = politica_ia(game, turno_idx)
            else:
                mv = politica_rival(game, turno_idx)
            game.apply_move(turno_idx, mv)

        s_ia = float(game.players[ia_seat].score)
        s_rv = float(game.players[1 - ia_seat].score)
        puntos_ia.append(s_ia)
        puntos_rival.append(s_rv)

        if s_ia > s_rv:
            victorias_ia += 1
        elif s_rv > s_ia:
            victorias_rival += 1

    win_rate_ia = 100.0 * victorias_ia / num_partidas
    win_rate_rv = 100.0 * victorias_rival / num_partidas
    return (
        win_rate_ia,
        win_rate_rv,
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
        help="Número de partidas balanceadas por enfrentamiento (debe ser par).",
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

    print("=" * 82)
    print(f"EVALUACIÓN DETERMINISTA DE IA v4 (K={args.k}) | N={args.partidas} partidas/rival")
    print(f"Checkpoint: {agente_v4.ruta_modelo}")
    print("=" * 82)
    print(
        f"{'Rival':<22} | {'Vic. IA':>8} | {'Vic. Rival':>10} | "
        f"{'Puntaje IA':>15} | {'Puntaje Rival':>13} | {'ΔElo':>7}"
    )
    print("-" * 82)

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
    print("=" * 82)


if __name__ == "__main__":
    main()
