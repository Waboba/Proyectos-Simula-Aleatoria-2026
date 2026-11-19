import random
import sys
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.azulvitral import AzulVitralGame
from src.env_azul_v2 import AzulSintraEnvV2, decode_action, heuristic_move


def crear_agente_v2(modelo):
    helper = AzulSintraEnvV2()

    def elegir(game: AzulVitralGame):
        helper.game = game
        ag = f"player_{game.current}"
        helper.agent_selection = ag
        obs_dict = helper.observe(ag)
        act, _ = modelo.predict(
            obs_dict["observation"],
            action_masks=obs_dict["action_mask"].astype(bool),
            deterministic=True,
        )
        return decode_action(int(act))

    return elegir


def crear_agente_v1():
    v1_path = PROJECT_ROOT / "models_checkpoint" / "ppo_azul_sintra_v1.zip"
    if not v1_path.exists():
        return None
    import src.env_azul as env_v1_mod

    modelo_v1 = MaskablePPO.load(str(v1_path), device="cpu")
    helper = env_v1_mod.AzulSintraEnv()
    decode_v1 = getattr(
        helper,
        "decode_action",
        getattr(env_v1_mod, "decode_action", None),
    )

    def elegir(game: AzulVitralGame):
        helper.game = game
        ag = f"player_{game.current}"
        helper.agent_selection = ag
        obs_dict = helper.observe(ag)
        act, _ = modelo_v1.predict(
            obs_dict["observation"],
            action_masks=obs_dict["action_mask"].astype(bool),
            deterministic=True,
        )
        return decode_v1(int(act))

    return elegir


def evaluar_contra(nombre_rival: str, agente_ia, agente_rival, n_partidas: int = 100):
    victorias_ia = 0
    victorias_rival = 0
    empates = 0
    pts_ia = []
    pts_rival = []

    for i in range(n_partidas):
        empieza_ia = i % 2 == 0
        game = AzulVitralGame(
            ["IA_v2", nombre_rival],
            first_player=0 if empieza_ia else 1,
            seed=10_000 + i,
        )
        while not game.game_over:
            move = agente_ia(game) if game.current == 0 else agente_rival(game)
            game.apply_move(move)

        pts_ia.append(game.players[0].score)
        pts_rival.append(game.players[1].score)
        ganador = game.winner()
        if ganador is None:
            empates += 1
        elif ganador == game.players[0]:
            victorias_ia += 1
        else:
            victorias_rival += 1

    print(f"\n--- IA v2 vs. {nombre_rival} ({n_partidas} partidas) ---")
    print(f"Victorias IA v2 : {victorias_ia} ({victorias_ia / n_partidas * 100:.1f}%)")
    print(f"Victorias Rival : {victorias_rival} ({victorias_rival / n_partidas * 100:.1f}%)")
    print(f"Empates         : {empates} ({empates / n_partidas * 100:.1f}%)")
    print(f"Puntaje IA v2   : {np.mean(pts_ia):.2f} ± {np.std(pts_ia):.2f}")
    print(f"Puntaje Rival   : {np.mean(pts_rival):.2f} ± {np.std(pts_rival):.2f}")


def main():
    v2_path = PROJECT_ROOT / "models_checkpoint" / "ppo_azul_sintra_v2.zip"
    if not v2_path.exists():
        print(f"No se encontró {v2_path}. Ejecuta primero train_rl_v2.py.")
        return

    modelo_v2 = MaskablePPO.load(str(v2_path), device="cpu")
    agente_v2 = crear_agente_v2(modelo_v2)

    rng_azar = random.Random(42)
    evaluar_contra("Aleatorio", agente_v2, lambda g: rng_azar.choice(g.legal_moves()))

    agente_v1 = crear_agente_v1()
    if agente_v1 is not None:
        evaluar_contra("Modelo v1", agente_v2, agente_v1)

    rng_heur = random.Random(42)
    evaluar_contra("Heurística Táctica", agente_v2, lambda g: heuristic_move(g, rng_heur))


if __name__ == "__main__":
    main()
