"""Evaluación determinista de IA v3 (100 partidas por rival, alternando salida).

Ejecutar desde la raíz del proyecto:
    python -m scripts.evaluar_ia_v3
"""

from pathlib import Path
import random
import statistics
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sb3_contrib import MaskablePPO

from src.azulvitral import AzulVitralGame
from src.env_azul_v2 import NUM_ACTIONS, build_observation_v2, decode_action, encode_move, heuristic_move
from src.env_azul_v3 import AzulGymWrapperV3

MODEL_V3_PATH = PROJECT_ROOT / "models_checkpoint" / "ppo_azul_sintra_v3.zip"
MODEL_V2_PATH = PROJECT_ROOT / "models_checkpoint" / "ppo_azul_sintra_v2.zip"
MODEL_V1_PATH = PROJECT_ROOT / "models_checkpoint" / "ppo_azul_sintra_v1.zip"


def evaluar_contra(modelo_v3: MaskablePPO, rival_tipo: str, modelo_rival=None, obs_fn_v1=None, n_partidas: int = 100):
    victorias_v3 = 0
    victorias_rival = 0
    empates = 0
    pts_v3 = []
    pts_rival = []

    for i in range(n_partidas):
        rng = random.Random(90_000 + i)
        seat_v3 = i % 2
        first_player = (i // 2) % 2
        game = AzulVitralGame(["P0", "P1"], first_player=first_player, seed=50_000 + i)

        while not game.game_over:
            cur = game.current
            legales = game.legal_moves()
            mask = np.zeros(NUM_ACTIONS, dtype=bool)
            for m in legales:
                mask[encode_move(m)] = True

            if cur == seat_v3:
                obs = build_observation_v2(game, cur).astype(np.float32)
                act, _ = modelo_v3.predict(obs, action_masks=mask, deterministic=True)
                game.apply_move(decode_action(int(act)))
            else:
                if rival_tipo == "random":
                    game.apply_move(rng.choice(legales))
                elif rival_tipo == "heuristic":
                    game.apply_move(heuristic_move(game, rng))
                elif rival_tipo == "v2":
                    obs = build_observation_v2(game, cur).astype(np.float32)
                    act, _ = modelo_rival.predict(obs, action_masks=mask, deterministic=True)
                    game.apply_move(decode_action(int(act)))
                elif rival_tipo == "v1":
                    obs = obs_fn_v1(game, cur)
                    act, _ = modelo_rival.predict(obs, action_masks=mask, deterministic=True)
                    game.apply_move(decode_action(int(act)))

        s_v3 = game.players[seat_v3].score
        s_riv = game.players[1 - seat_v3].score
        pts_v3.append(s_v3)
        pts_rival.append(s_riv)

        w = game.winner()
        if w is None:
            empates += 1
        elif game.players.index(w) == seat_v3:
            victorias_v3 += 1
        else:
            victorias_rival += 1

    print(f"\n--- IA v3 vs. {rival_tipo.upper()} ({n_partidas} partidas) ---")
    print(f"Victorias IA v3 : {victorias_v3} ({victorias_v3 / n_partidas * 100:.1f}%)")
    print(f"Victorias Rival : {victorias_rival} ({victorias_rival / n_partidas * 100:.1f}%)")
    print(f"Empates         : {empates} ({empates / n_partidas * 100:.1f}%)")
    print(f"Puntaje IA v3   : {statistics.mean(pts_v3):.2f} ± {statistics.pstdev(pts_v3):.2f}")
    print(f"Puntaje Rival   : {statistics.mean(pts_rival):.2f} ± {statistics.pstdev(pts_rival):.2f}")


def main():
    if not MODEL_V3_PATH.exists():
        print(f"No se encontró {MODEL_V3_PATH}. Ejecuta primero: python -m scripts.train_rl_v3")
        return

    modelo_v3 = MaskablePPO.load(str(MODEL_V3_PATH), device="cpu")
    evaluar_contra(modelo_v3, "random", n_partidas=100)

    if MODEL_V1_PATH.exists():
        modelo_v1 = MaskablePPO.load(str(MODEL_V1_PATH), device="cpu")
        obs_fn_v1 = AzulGymWrapperV3._resolve_v1_obs_fn()
        evaluar_contra(modelo_v3, "v1", modelo_rival=modelo_v1, obs_fn_v1=obs_fn_v1, n_partidas=100)

    if MODEL_V2_PATH.exists():
        modelo_v2 = MaskablePPO.load(str(MODEL_V2_PATH), device="cpu")
        evaluar_contra(modelo_v3, "v2", modelo_rival=modelo_v2, n_partidas=100)

    evaluar_contra(modelo_v3, "heuristic", n_partidas=100)


if __name__ == "__main__":
    main()
