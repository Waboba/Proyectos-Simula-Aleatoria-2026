import os
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv

from src.env_azul_v2 import AzulGymWrapperV2, heuristic_move


def _crear_oponente_v1():
    """Carga ppo_azul_sintra_v1.zip usando su propio entorno y decodificador (v1)."""
    v1_path = PROJECT_ROOT / "models_checkpoint" / "ppo_azul_sintra_v1.zip"
    if not v1_path.exists():
        return None
    try:
        import src.env_azul as env_v1_mod

        modelo_v1 = MaskablePPO.load(str(v1_path), device="cpu")
        helper_v1 = env_v1_mod.AzulSintraEnv()
        decode_v1 = getattr(
            helper_v1,
            "decode_action",
            getattr(env_v1_mod, "decode_action", None),
        )

        def jugar_v1(game):
            helper_v1.game = game
            ag = f"player_{game.current}"
            helper_v1.agent_selection = ag
            obs_dict = helper_v1.observe(ag)
            act, _ = modelo_v1.predict(
                obs_dict["observation"],
                action_masks=obs_dict["action_mask"].astype(bool),
                deterministic=False,
            )
            return decode_v1(int(act))

        return jugar_v1
    except Exception as err:
        print(f"Aviso: No se pudo cargar v1 para Self-Play ({err}); se usará heurística.")
        return None


def make_env(seed: int, oponente_v1=None):
    def _init():
        rng = random.Random(seed)
        if oponente_v1 is not None:
            pool = [
                lambda g: heuristic_move(g, rng),
                oponente_v1,
                lambda g: rng.choice(g.legal_moves()),
            ]
            weights = [0.45, 0.35, 0.20]
        else:
            pool = [
                lambda g: heuristic_move(g, rng),
                lambda g: rng.choice(g.legal_moves()),
            ]
            weights = [0.75, 0.25]

        env = AzulGymWrapperV2(opponent_pool=pool, opponent_weights=weights)
        env = ActionMasker(env, lambda e: e.action_masks())
        env.reset(seed=seed)
        return env

    return _init


def train():
    print("=== Entrenando Azul: Vitrales de Sintra (v2 — Self-Play + Heurística) ===")
    os.makedirs("models_checkpoint", exist_ok=True)
    os.makedirs("tensorboard_logs", exist_ok=True)

    oponente_v1 = _crear_oponente_v1()
    if oponente_v1 is not None:
        print("Currículo activo: 45% Heurística táctica | 35% Self-Play (v1) | 20% Aleatorio")
    else:
        print("Currículo activo: 75% Heurística táctica | 25% Aleatorio")

    num_envs = 4
    env = DummyVecEnv(
        [make_env(seed=100 + i, oponente_v1=oponente_v1) for i in range(num_envs)]
    )

    model = MaskablePPO(
        policy=MaskableActorCriticPolicy,
        env=env,
        learning_rate=2.5e-4,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        verbose=1,
        tensorboard_log="./tensorboard_logs/",
    )

    total_timesteps = 500_000
    print(f"Iniciando entrenamiento durante {total_timesteps:,} pasos...")
    model.learn(total_timesteps=total_timesteps, tb_log_name="MaskablePPO_v2")

    save_path = "models_checkpoint/ppo_azul_sintra_v2"
    model.save(save_path)
    print(f"¡Entrenamiento v2 finalizado! Modelo guardado en: {save_path}.zip")


if __name__ == "__main__":
    train()
