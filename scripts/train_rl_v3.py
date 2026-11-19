"""Entrenamiento en 2 fases para IA v3 (Azul: Vitrales de Sintra).

Fase 1 (~3 min en CPU): Preentrenamiento supervisado (Behavioral Cloning)
  sobre 30,000 estados jugados por heuristic_move para partir con ~60+ pts
  y disciplina estricta de cristal roto.
Fase 2 (~40 min en CPU): Ajuste fino con MaskablePPO (400,000 pasos)
  sobre AzulGymWrapperV3 (50% Heurística, 30% v2, 10% v1, 10% Azar).

Ejecutar desde la raíz del proyecto:
    python -m scripts.train_rl_v3
"""

from pathlib import Path
import random
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv
import torch

from src.azulvitral import AzulVitralGame
from src.env_azul_v2 import NUM_ACTIONS, OBS_DIM, build_observation_v2, encode_move, heuristic_move
from src.env_azul_v3 import AzulGymWrapperV3

MODEL_V3_BC_PATH = PROJECT_ROOT / "models_checkpoint" / "ppo_azul_sintra_v3_bc.zip"
MODEL_V3_PATH = PROJECT_ROOT / "models_checkpoint" / "ppo_azul_sintra_v3.zip"
TB_LOG_DIR = PROJECT_ROOT / "tensorboard_logs"


def mask_fn(env: gym.Env) -> np.ndarray:
    return env.action_masks()


def make_env(seed_offset: int):
    def _init():
        env = AzulGymWrapperV3(opponent_mode="curriculum", curriculum=(0.50, 0.30, 0.10, 0.10))
        env.reset(seed=3000 + seed_offset)
        return ActionMasker(env, mask_fn)

    return _init


def recolectar_dataset_heuristico(n_samples: int = 30_000, seed: int = 2026):
    """Genera pares (observación_288, máscara_1621, acción_heurística) sin sesgo de asiento."""
    rng = random.Random(seed)
    obs_list = np.empty((n_samples, OBS_DIM), dtype=np.float32)
    mask_list = np.zeros((n_samples, NUM_ACTIONS), dtype=bool)
    act_list = np.empty(n_samples, dtype=np.int64)

    count = 0
    while count < n_samples:
        game = AzulVitralGame(["H0", "H1"], first_player=rng.choice([0, 1]), seed=rng.randint(1, 10_000_000))
        rival_azar = rng.random() < 0.25  # 25% contra azar para cubrir tableros con alta oferta

        while not game.game_over and count < n_samples:
            p_idx = game.current
            legales = game.legal_moves()
            best_move = heuristic_move(game, rng)
            best_idx = encode_move(best_move)

            # Registra la decisión experta (en ambos asientos o en asiento 0 si rival es azar)
            if not rival_azar or p_idx == 0:
                obs_list[count] = build_observation_v2(game, p_idx)
                for m in legales:
                    mask_list[count, encode_move(m)] = True
                act_list[count] = best_idx
                count += 1

            if rival_azar and p_idx == 1:
                game.apply_move(rng.choice(legales))
            else:
                game.apply_move(best_move)

    return obs_list, mask_list, act_list


def preentrenar_imitacion(
    model: MaskablePPO,
    obs_arr: np.ndarray,
    masks_arr: np.ndarray,
    acts_arr: np.ndarray,
    epochs: int = 12,
    batch_size: int = 256,
    lr: float = 1e-3,
):
    """Entrena la red de política por máxima verosimilitud sobre jugadas heurísticas."""
    policy = model.policy
    policy.set_training_mode(True)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    n = len(acts_arr)
    rng_np = np.random.default_rng(42)

    for epoch in range(1, epochs + 1):
        perm = rng_np.permutation(n)
        total_loss = 0.0
        correct = 0

        for start in range(0, n, batch_size):
            idx = perm[start: start + batch_size]
            obs_b = torch.as_tensor(obs_arr[idx], dtype=torch.float32, device=model.device)
            mask_b = torch.as_tensor(masks_arr[idx], dtype=torch.bool, device=model.device)
            act_b = torch.as_tensor(acts_arr[idx], dtype=torch.long, device=model.device)

            dist = policy.get_distribution(obs_b, action_masks=mask_b)
            loss = -dist.log_prob(act_b).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()

            with torch.no_grad():
                preds = dist.get_actions(deterministic=True)
                correct += int((preds == act_b).sum().item())
            total_loss += float(loss.item()) * len(idx)

        print(
            f"  [Fase 1 - BC] Época {epoch:02d}/{epochs:02d} | "
            f"Loss: {total_loss / n:.4f} | Precisión top-1: {correct / n * 100:.1f}%"
        )

    policy.set_training_mode(False)


def main():
    MODEL_V3_PATH.parent.mkdir(parents=True, exist_ok=True)
    TB_LOG_DIR.mkdir(parents=True, exist_ok=True)

    print("=== FASE 1: Preentrenamiento por Imitación (Behavioral Cloning) ===")
    t0 = time.time()
    obs_arr, masks_arr, acts_arr = recolectar_dataset_heuristico(n_samples=30_000, seed=2026)
    print(f"Dataset recolectado ({len(acts_arr)} turnos) en {time.time() - t0:.1f} s.")

    vec_env = DummyVecEnv([make_env(i) for i in range(4)])
    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        policy_kwargs={"net_arch": [256, 256]},
        tensorboard_log=str(TB_LOG_DIR),
        verbose=1,
        seed=2026,
        device="cpu",
    )

    preentrenar_imitacion(model, obs_arr, masks_arr, acts_arr, epochs=12, batch_size=256, lr=1e-3)
    model.save(str(MODEL_V3_BC_PATH))
    print(f"Checkpoint inicial guardado en: {MODEL_V3_BC_PATH}\n")

    print("=== FASE 2: Ajuste Fino con MaskablePPO + Currículo Competitivo (400,000 pasos) ===")
    model.learn(
        total_timesteps=400_000,
        tb_log_name="MaskablePPO_v3",
        reset_num_timesteps=True,
    )
    model.save(str(MODEL_V3_PATH))
    vec_env.close()
    print(f"\n¡Entrenamiento v3 finalizado! Modelo guardado en: {MODEL_V3_PATH}")


if __name__ == "__main__":
    main()
