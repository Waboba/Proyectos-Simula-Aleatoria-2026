"""Evalúa el modelo MaskablePPO entrenado frente a oponentes de referencia.

Ejecutar desde la raíz del proyecto:
    python -m scripts.evaluar_ia
"""

import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sb3_contrib import MaskablePPO

from src.azulvitral import AzulVitralGame, Move, Player
from src.env_azul import AzulSintraEnv
from src.partida import TwoPlayerMatch


def crear_agente_ia(model: MaskablePPO):
    """Devuelve una función de entrada compatible con TwoPlayerMatch."""
    env_helper = AzulSintraEnv()

    def elegir_jugada(game: AzulVitralGame, player: Player) -> Move:
        env_helper.game = game
        player_idx = game.players.index(player)
        agent_name = f"player_{player_idx}"
        env_helper.agent_selection = agent_name

        obs_dict = env_helper.observe(agent_name)
        action, _ = model.predict(
            obs_dict["observation"],
            action_masks=obs_dict["action_mask"].astype(bool),
            deterministic=True,
        )
        return env_helper.decode_action(int(action))

    return elegir_jugada


def crear_agente_aleatorio(rng: random.Random):
    """Baseline aleatorio uniforme entre jugadas válidas."""

    def elegir_jugada(game: AzulVitralGame, player: Player) -> Move:
        return rng.choice(game.legal_moves())

    return elegir_jugada


def evaluar(num_partidas: int = 100):
    model_path = PROJECT_ROOT / "models_checkpoint" / "ppo_azul_sintra_v1.zip"
    if not model_path.exists():
        print(f"No se encontró el modelo en {model_path}.")
        print("Ejecuta primero: python -m scripts.train_rl")
        return

    model = MaskablePPO.load(str(model_path))
    agente_ia = crear_agente_ia(model)

    victorias_ia = 0
    victorias_rival = 0
    empates = 0
    puntos_ia = []
    puntos_rival = []

    print(f"Evaluando IA en {num_partidas} partidas contra Agente Aleatorio...")

    for i in range(num_partidas):
        rng = random.Random(1000 + i)
        agente_azar = crear_agente_aleatorio(rng)

        # Alternar quién empieza en cada partida
        first = "IA_PPO" if i % 2 == 0 else "Rival_Azar"
        match = TwoPlayerMatch(
            names=["IA_PPO", "Rival_Azar"],
            inputs={"IA_PPO": agente_ia, "Rival_Azar": agente_azar},
            first_player=first,
            seed=1000 + i,
        )
        ganador = match.run()

        p_ia = next(p for p in match.game.players if p.name == "IA_PPO")
        p_riv = next(p for p in match.game.players if p.name == "Rival_Azar")

        puntos_ia.append(p_ia.score)
        puntos_rival.append(p_riv.score)

        if ganador is None:
            empates += 1
        elif ganador.name == "IA_PPO":
            victorias_ia += 1
        else:
            victorias_rival += 1

    print("\n=== Resultados de Evaluación ===")
    print(f"Partidas jugadas : {num_partidas}")
    print(f"Victorias IA     : {victorias_ia} ({victorias_ia / num_partidas * 100:.1f}%)")
    print(f"Victorias Rival  : {victorias_rival} ({victorias_rival / num_partidas * 100:.1f}%)")
    print(f"Empates          : {empates} ({empates / num_partidas * 100:.1f}%)")
    print(f"Puntaje medio IA : {np.mean(puntos_ia):.2f} ± {np.std(puntos_ia):.2f}")
    print(f"Puntaje Rival    : {np.mean(puntos_rival):.2f} ± {np.std(puntos_rival):.2f}")


if __name__ == "__main__":
    evaluar(100)
