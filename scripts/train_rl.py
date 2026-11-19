"""Entrenamiento de agente RL para Azul: Vitrales de Sintra usando MaskablePPO.

Ejecutar desde la raíz del proyecto:
    python -m scripts.train_rl
"""

import os
import sys
from pathlib import Path

# Añadir la raíz del proyecto al sys.path antes de importar módulos locales
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pettingzoo.test import api_test
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

from src.env_azul import AzulGymWrapper, AzulSintraEnv


def mask_fn(env: AzulGymWrapper):
    """Extrae la máscara de acciones válidas para ActionMasker."""
    return env.action_masks()


def make_env(seed: int = 42):
    def _init():
        env = AzulGymWrapper(seed=seed)
        return ActionMasker(env, mask_fn)

    return _init


def validar_entornos():
    """Verifica conformidad con PettingZoo AEC y Gymnasium antes de entrenar."""
    print("1. Validando entorno PettingZoo (api_test)...")
    pz_env = AzulSintraEnv()
    api_test(pz_env, num_cycles=500, verbose_progress=False)

    print("2. Validando entorno Gymnasium (check_env)...")
    gym_env = AzulGymWrapper(seed=123)
    check_env(gym_env, warn=True)
    print("   -> ¡Ambos entornos validados sin errores!\n")


def train():
    print("=== Entrenamiento de IA para Azul: Vitrales de Sintra ===")
    validar_entornos()

    # Crear 4 entornos paralelos en DummyVecEnv
    num_envs = 4
    env = DummyVecEnv([make_env(seed=100 + i) for i in range(num_envs)])

    checkpoint_dir = PROJECT_ROOT / "models_checkpoint"
    tb_log_dir = PROJECT_ROOT / "tensorboard_logs"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(tb_log_dir, exist_ok=True)

    model = MaskablePPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=str(tb_log_dir),
    )

    total_timesteps = 200_000
    print(f"Iniciando entrenamiento por {total_timesteps:,} pasos...")
    model.learn(total_timesteps=total_timesteps)

    save_path = checkpoint_dir / "ppo_azul_sintra_v1"
    model.save(str(save_path))
    print(f"\n¡Entrenamiento finalizado! Modelo guardado en: {save_path}.zip")


if __name__ == "__main__":
    train()
