"""Entorno RL v3 para Azul: Vitrales de Sintra.

Mejoras respecto a v2:
1. Descuento inmediato y telescópico del cristal roto en cada turno
   (resuelve la asignación diferida de crédito al cierre de la ronda 6).
2. Currículo competitivo que incluye a IA v2 (30%), Heurística Táctica (50%),
   IA v1 (10%) y Aleatorio (10%).
"""

import importlib
from pathlib import Path
import random

import gymnasium as gym
import numpy as np
from pettingzoo import AECEnv

from src.azulvitral import AzulVitralGame, Move
from src.env_azul_v2 import (
    NUM_ACTIONS,
    OBS_DIM,
    build_observation_v2,
    decode_action,
    encode_move,
    heuristic_move,
)

# Penalización acumulada por casilla de cristal roto (casilla 3 -> 2, casilla 9 -> 15)
PENALIZACION_CRISTAL = (0, 1, 1, 2, 4, 6, 8, 11, 14, 15, 18, 22, 26)


def costo_cristal_roto(player) -> float:
    """Extrae la casilla actual de cristal roto del jugador y devuelve su penalización."""
    casilla = 0
    for attr in ("broken", "broken_glass", "broken_pos", "cristal_roto", "rotos", "glass_broken"):
        if hasattr(player, attr):
            val = getattr(player, attr)
            if isinstance(val, (int, float)):
                casilla = int(val)
                break
            if isinstance(val, (list, tuple)):
                casilla = len(val)
                break
    idx = max(0, min(casilla, len(PENALIZACION_CRISTAL) - 1))
    return float(PENALIZACION_CRISTAL[idx])


class AzulSintraEnvV3(AECEnv):
    """Entorno multi-agente PettingZoo AEC v3 con penalización densa de cristal roto."""

    metadata = {"render_modes": ["ansi"], "name": "azul_sintra_v3"}

    def __init__(self, render_mode: str | None = None):
        super().__init__()
        self.render_mode = render_mode
        self.possible_agents = ["player_0", "player_1"]
        self.agents: list[str] = []

        obs_box = gym.spaces.Box(low=-10.0, high=10.0, shape=(OBS_DIM,), dtype=np.float32)
        mask_box = gym.spaces.Box(low=0, high=1, shape=(NUM_ACTIONS,), dtype=np.int8)
        self.observation_spaces = {
            agent: gym.spaces.Dict({"observation": obs_box, "action_mask": mask_box})
            for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: gym.spaces.Discrete(NUM_ACTIONS)
            for agent in self.possible_agents
        }

        self.game: AzulVitralGame | None = None
        self.agent_selection = "player_0"
        self.rewards = {a: 0.0 for a in self.possible_agents}
        self._cumulative_rewards = {a: 0.0 for a in self.possible_agents}
        self.terminations = {a: False for a in self.possible_agents}
        self.truncations = {a: False for a in self.possible_agents}
        self.infos = {a: {} for a in self.possible_agents}

    def observation_space(self, agent: str) -> gym.spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> gym.spaces.Space:
        return self.action_spaces[agent]

    def reset(self, seed: int | None = None, options: dict | None = None):
        self.agents = self.possible_agents[:]
        first_player = int(options["first_player"]) if (options and "first_player" in options) else 0

        self.game = AzulVitralGame(["Player 0", "Player 1"], first_player=first_player, seed=seed)
        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        self.agent_selection = f"player_{self.game.current}"

    def action_mask(self, agent: str) -> np.ndarray:
        mask = np.zeros(NUM_ACTIONS, dtype=np.int8)
        if (
            self.game is None
            or self.game.game_over
            or agent != self.agent_selection
            or self.terminations.get(agent, True)
        ):
            return mask
        for move in self.game.legal_moves():
            mask[encode_move(move)] = 1
        return mask

    def observe(self, agent: str) -> dict[str, np.ndarray]:
        p_idx = 0 if agent == "player_0" else 1
        obs = build_observation_v2(self.game, p_idx).astype(np.float32)
        return {"observation": obs, "action_mask": self.action_mask(agent)}

    def _shaping_bonus(
        self,
        move: Move,
        prev_score: int,
        new_score: int,
        delta_costo_roto: float,
    ) -> float:
        """Señal densa que castiga inmediatamente romper piezas y premia limpieza táctica."""
        if delta_costo_roto > 0.0:
            return -0.40 * delta_costo_roto
        if move == Move.reset_glazier():
            return -0.08
        if new_score > prev_score:
            return 0.25
        return 0.15

    def step(self, action):
        agent = self.agent_selection

        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            if self.agents:
                self.agent_selection = self.agents[0]
            return

        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards[agent] = 0.0

        p_idx = 0 if agent == "player_0" else 1
        prev_scores = [self.game.players[0].score, self.game.players[1].score]
        prev_rotos = [
            costo_cristal_roto(self.game.players[0]),
            costo_cristal_roto(self.game.players[1]),
        ]

        move = decode_action(int(action))
        self.game.apply_move(move)

        new_scores = [self.game.players[0].score, self.game.players[1].score]
        new_rotos = [
            costo_cristal_roto(self.game.players[0]),
            costo_cristal_roto(self.game.players[1]),
        ]

        delta_roto_0 = new_rotos[0] - prev_rotos[0]
        delta_roto_1 = new_rotos[1] - prev_rotos[1]

        # Suma telescópica: cobra el cristal roto al instante en rondas 1..6
        # y evita duplicarlo cuando el motor lo descuenta al cerrar game_over.
        if not self.game.game_over:
            self.rewards["player_0"] = float(new_scores[0] - prev_scores[0]) - delta_roto_0
            self.rewards["player_1"] = float(new_scores[1] - prev_scores[1]) - delta_roto_1
        else:
            self.rewards["player_0"] = float(new_scores[0] - prev_scores[0]) + prev_rotos[0]
            self.rewards["player_1"] = float(new_scores[1] - prev_scores[1]) + prev_rotos[1]

        if p_idx == 0:
            self.rewards["player_0"] += self._shaping_bonus(
                move, prev_scores[0], new_scores[0], delta_roto_0
            )
        else:
            self.rewards["player_1"] += self._shaping_bonus(
                move, prev_scores[1], new_scores[1], delta_roto_1
            )

        if self.game.game_over:
            self.terminations = {a: True for a in self.agents}
            winner = self.game.winner()
            if winner is not None:
                w_idx = self.game.players.index(winner)
                self.rewards[f"player_{w_idx}"] += 60.0
                self.rewards[f"player_{1 - w_idx}"] -= 60.0
            self.agent_selection = self.agents[0]
        else:
            self.agent_selection = f"player_{self.game.current}"

        self._accumulate_rewards()

    def render(self):
        return None

    def close(self):
        pass


class AzulGymWrapperV3(gym.Env):
    """Envoltorio Gymnasium v3 con currículo (50% Heurística, 30% v2, 10% v1, 10% Azar)."""

    metadata = {"render_modes": ["ansi"]}
    _cached_models: dict[str, object] = {}
    _load_attempted: set[str] = set()
    _cached_v1_obs_fn = None

    def __init__(
        self,
        opponent_mode: str = "curriculum",
        curriculum: tuple[float, float, float, float] = (0.50, 0.30, 0.10, 0.10),
    ):
        super().__init__()
        self.aec_env = AzulSintraEnvV3()
        self.opponent_mode = opponent_mode
        self.curriculum = curriculum

        self.observation_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(NUM_ACTIONS)

        self.rng = random.Random()
        self.agent_idx = 0
        self.agent_name = "player_0"
        self.opp_name = "player_1"
        self.current_opponent = "heuristic"

    @classmethod
    def _get_model(cls, tag: str, filename: str):
        if tag in cls._load_attempted:
            return cls._cached_models.get(tag)
        cls._load_attempted.add(tag)
        path = Path(__file__).resolve().parent.parent / "models_checkpoint" / filename
        if not path.exists():
            return None
        try:
            from sb3_contrib import MaskablePPO
            cls._cached_models[tag] = MaskablePPO.load(str(path), device="cpu")
        except Exception:
            cls._cached_models[tag] = None
        return cls._cached_models.get(tag)

    @classmethod
    def _resolve_v1_obs_fn(cls):
        if cls._cached_v1_obs_fn is not None:
            return cls._cached_v1_obs_fn
        mod = importlib.import_module("src.env_azul")
        for fn_name in ("build_observation", "get_observation", "encode_observation", "build_obs"):
            fn = getattr(mod, fn_name, None)
            if callable(fn):
                cls._cached_v1_obs_fn = fn
                return cls._cached_v1_obs_fn
        for obj in vars(mod).values():
            if isinstance(obj, type) and issubclass(obj, AECEnv) and obj is not AECEnv:
                env_v1 = obj()

                def _obs_from_aec(game, opp_idx: int) -> np.ndarray:
                    env_v1.game = game
                    env_v1.agents = ["player_0", "player_1"]
                    env_v1.terminations = {"player_0": False, "player_1": False}
                    env_v1.agent_selection = f"player_{opp_idx}"
                    out = env_v1.observe(f"player_{opp_idx}")
                    obs = out["observation"] if isinstance(out, dict) else out
                    return np.asarray(obs, dtype=np.float32)

                cls._cached_v1_obs_fn = _obs_from_aec
                return cls._cached_v1_obs_fn
        return None

    def _choose_opponent(self) -> str:
        if self.opponent_mode != "curriculum":
            return self.opponent_mode
        p_heur, p_v2, p_v1, _ = self.curriculum
        r = self.rng.random()
        if r < p_heur:
            return "heuristic"
        if r < p_heur + p_v2 and self._get_model("v2", "ppo_azul_sintra_v2.zip") is not None:
            return "v2"
        if r < p_heur + p_v2 + p_v1 and self._get_model("v1", "ppo_azul_sintra_v1.zip") is not None:
            return "v1"
        return "random"

    def _select_opponent_action(self) -> int:
        game = self.aec_env.game
        legales = game.legal_moves()
        opp_idx = 1 - self.agent_idx
        mask = self.aec_env.action_mask(self.opp_name).astype(bool)

        if self.current_opponent == "heuristic":
            return encode_move(heuristic_move(game, self.rng))

        if self.current_opponent == "v2":
            model_v2 = self._get_model("v2", "ppo_azul_sintra_v2.zip")
            if model_v2 is not None:
                obs_v2 = build_observation_v2(game, opp_idx).astype(np.float32)
                action, _ = model_v2.predict(obs_v2, action_masks=mask, deterministic=False)
                return int(action)

        if self.current_opponent == "v1":
            model_v1 = self._get_model("v1", "ppo_azul_sintra_v1.zip")
            obs_fn = self._resolve_v1_obs_fn()
            if model_v1 is not None and obs_fn is not None:
                obs_v1 = obs_fn(game, opp_idx)
                action, _ = model_v1.predict(obs_v1, action_masks=mask, deterministic=False)
                return int(action)

        return encode_move(self.rng.choice(legales))

    def action_masks(self) -> np.ndarray:
        return self.aec_env.action_mask(self.agent_name).astype(bool)

    def _get_obs(self) -> np.ndarray:
        return build_observation_v2(self.aec_env.game, self.agent_idx).astype(np.float32)

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.rng = random.Random(seed)
        self.agent_idx = self.rng.choice([0, 1])
        self.agent_name = f"player_{self.agent_idx}"
        self.opp_name = f"player_{1 - self.agent_idx}"
        self.current_opponent = self._choose_opponent()

        first_player = self.rng.choice([0, 1])
        self.aec_env.reset(seed=seed, options={"first_player": first_player})

        while not self.aec_env.game.game_over and self.aec_env.agent_selection != self.agent_name:
            self.aec_env.step(self._select_opponent_action())

        return self._get_obs(), {"opponent": self.current_opponent, "seat": self.agent_idx}

    def step(self, action: int):
        self.aec_env.step(int(action))
        total_reward = float(self.aec_env.rewards.get(self.agent_name, 0.0))

        while not self.aec_env.game.game_over and self.aec_env.agent_selection != self.agent_name:
            self.aec_env.step(self._select_opponent_action())
            total_reward += float(self.aec_env.rewards.get(self.agent_name, 0.0))

        terminated = bool(self.aec_env.game.game_over)
        return self._get_obs(), total_reward, terminated, False, {"opponent": self.current_opponent}
