"""Entrenamiento para IA v4 (Azul: Vitrales de Sintra).

Mejoras respecto a train_rl_v3.py:
1. Usa `AzulGymWrapperV4`, donde `costo_cristal_roto` lee correctamente
   `player.broken_position` (casillas 1..10) y `BROKEN_GLASS_PENALTIES`,
   activando la penalización telescópica real turno a turno.
2. Transfer Learning desde `ppo_azul_sintra_v3.zip` (si existe) o preentrenamiento
   por Behavioral Cloning en caso contrario, seguido de 400,000 pasos de
   `MaskablePPO` frente a un currículo más exigente:
   50% Heurística Táctica | 30% IA v3 | 15% IA v2 | 5% Aleatorio.

Ejecutar desde la raíz del proyecto:
    python -m scripts.train_rl_v4
"""

from pathlib import Path
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

from scripts.train_rl_v3 import preentrenar_imitacion, recolectar_dataset_heuristico
from src.env_azul_v4 import AzulGymWrapperV4

MODEL_V3_PATH = PROJECT_ROOT / "models_checkpoint" / "ppo_azul_sintra_v3.zip"
MODEL_V4_PATH = PROJECT_ROOT / "models_checkpoint" / "ppo_azul_sintra_v4.zip"
TB_LOG_DIR = PROJECT_ROOT / "tensorboard_logs"


def mask_fn(env: gym.Env) -> np.ndarray:
    return env.action_masks()


def make_env(seed_offset: int):
    def _init():
        env = AzulGymWrapperV4(
            opponent_mode="curriculum",
            curriculum=(0.50, 0.30, 0.15, 0.05),
        )
        env.reset(seed=4000 + seed_offset)
        return ActionMasker(env, mask_fn)

    return _init


def main():
    MODEL_V4_PATH.parent.mkdir(parents=True, exist_ok=True)
    TB_LOG_DIR.mkdir(parents=True, exist_ok=True)

    vec_env = DummyVecEnv([make_env(i) for i in range(4)])

    if MODEL_V3_PATH.exists():
        print(f"=== Cargando pesos base desde {MODEL_V3_PATH.name} (Transfer Learning) ===")
        model = MaskablePPO.load(
            str(MODEL_V3_PATH),
            env=vec_env,
            device="cpu",
            learning_rate=8e-5,
            ent_coef=0.008,
            tensorboard_log=str(TB_LOG_DIR),
        )
    else:
        print("=== No se encontró v3.zip; ejecutando Fase 1 (Behavioral Cloning) desde cero ===")
        t0 = time.time()
        obs_arr, masks_arr, acts_arr = recolectar_dataset_heuristico(n_samples=30_000, seed=2026)
        print(f"Dataset recolectado ({len(acts_arr)} turnos) en {time.time() - t0:.1f} s.")

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
            seed=4040,
            device="cpu",
        )
        preentrenar_imitacion(model, obs_arr, masks_arr, acts_arr, epochs=12, batch_size=256, lr=1e-3)

    print("=== Entrenando IA v4 con MaskablePPO + Cristal Roto Telescópico Exacto (400,000 pasos) ===")
    print("Currículo: 50% Heurística | 30% IA v3 | 15% IA v2 | 5% Aleatorio")
    model.learn(
        total_timesteps=400_000,
        tb_log_name="MaskablePPO_v4",
        reset_num_timesteps=True,
    )
    model.save(str(MODEL_V4_PATH))
    vec_env.close()
    print(f"\n¡Entrenamiento v4 finalizado! Modelo guardado en: {MODEL_V4_PATH}")


if __name__ == "__main__":
    main()
