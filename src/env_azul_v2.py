import functools
import random
from typing import Callable, List, Optional
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pettingzoo import AECEnv

from src.azulvitral import (
    CENTER,
    JOKER,
    NUM_ROUNDS,
    NUM_STRIPS,
    ORNAMENT_BONUS,
    STRIP_SIZE,
    WINDOW_POINTS,
    AzulVitralGame,
    Color,
    Move,
)

COLORS = list(Color)
COLOR_TO_IDX = {c: i for i, c in enumerate(COLORS)}
NUM_ACTIONS = 1621
OBS_DIM = 288


def encode_move(move: Move) -> int:
    """Codifica un objeto Move en un entero discreto [0, 1620]."""
    if move.option == "B":
        return 0
    src_i = 5 if move.source == CENTER else (move.source - 1)
    col_i = COLOR_TO_IDX[move.color]
    strip_i = 8 if move.strip is None else (move.strip - 1)
    win_i = 0 if move.window_color is None else (COLOR_TO_IDX[move.window_color] + 1)
    return 1 + (((src_i * 5 + col_i) * 9 + strip_i) * 6 + win_i)


def decode_action(action_idx: int) -> Move:
    """Decodifica un índice entero [0, 1620] en su objeto Move biyectivo."""
    if action_idx == 0:
        return Move.reset_glazier()
    rem = action_idx - 1
    win_i = rem % 6
    rem //= 6
    strip_i = rem % 9
    rem //= 9
    col_i = rem % 5
    src_i = rem // 5
    source = CENTER if src_i == 5 else (src_i + 1)
    color = COLORS[col_i]
    strip = None if strip_i == 8 else (strip_i + 1)
    win_color = None if win_i == 0 else COLORS[win_i - 1]
    return Move.take(source, color, strip, win_color)


def build_observation_v2(game: AzulVitralGame, player_idx: int) -> np.ndarray:
    """Vector egocéntrico normalizado de 288 dimensiones (incluye las 6 rondas)."""
    obs: List[float] = []

    # 1. Fábricas (5 x 5 = 25)
    for factory in game.factories:
        for c in COLORS:
            obs.append(factory.tiles.count(c) / 4.0)

    # 2. Centro y loseta inicial (5 + 1 = 6)
    for c in COLORS:
        obs.append(game.center.tiles.count(c) / 20.0)
    obs.append(1.0 if game.center.has_first_player_token else 0.0)

    # 3. Ronda actual y marcador completo de las 6 rondas (1 + 30 = 31)
    obs.append(game.round / float(NUM_ROUNDS))
    for r_col in game.round_marker:
        for c in COLORS:
            obs.append(1.0 if r_col == c else 0.0)

    # 4. Suministro restante en bolsa y torre (5 + 2 = 7)
    for c in COLORS:
        obs.append(game.bag.count(c) / 20.0)
    obs.append(len(game.bag) / 100.0)
    obs.append(len(game.tower) / 100.0)

    # 5. Estado egocéntrico de ambos jugadores (2 x 109 = 218)
    for p_idx in (player_idx, 1 - player_idx):
        p = game.players[p_idx]
        obs.append(min(p.score, 150) / 100.0)
        obs.append(p.glazier / 7.0)
        obs.append(p.broken_position / 10.0)
        obs.append(1.0 if p.has_first_player_token else 0.0)
        obs.append(1.0 if p.can_reset_glazier() else 0.0)

        for strip in p.strips:
            obs.append(1.0 if strip.is_removed else 0.0)
            pat = strip.pattern()
            for s in range(STRIP_SIZE):
                obs.append(1.0 if strip.filled_spaces[s] is not None else 0.0)
                if strip.is_removed:
                    obs.append(0.0)
                elif pat[s] == JOKER:
                    obs.append(1.0)
                else:
                    obs.append((COLOR_TO_IDX[pat[s]] + 1) / 6.0)

        for w in p.windows:
            obs.append(1.0 if w[0] is not None else 0.0)
            obs.append(1.0 if w[1] is not None else 0.0)

    # 6. Indicador de jugador inicial (1)
    obs.append(1.0 if player_idx == 0 else 0.0)

    return np.array(obs, dtype=np.float32)


def heuristic_move(game: AzulVitralGame, rng: Optional[random.Random] = None) -> Move:
    """Política heurística táctica rápida (~63 pts promedio contra aleatorio)."""
    rng = rng or random
    legales = game.legal_moves()
    if len(legales) == 1:
        return legales[0]

    player = game.get_current_player()
    round_color = game.current_round_color()
    best_score = -1e9
    best_moves: List[Move] = []

    for m in legales:
        if m.option == "B":
            sc = 0.35 * player.glazier - 0.9
        else:
            if m.source == CENTER:
                taken = game.center.tiles.count(m.color)
                first_penalty = 0.8 if game.center.has_first_player_token else 0.0
            else:
                taken = game.factories[m.source - 1].tiles.count(m.color)
                first_penalty = 0.0

            if m.strip is None:
                sc = -2.2 * taken - first_penalty
            else:
                strip_idx = m.strip - 1
                strip = player.strips[strip_idx]
                free = len(strip.free_slots_for(m.color))
                placed = min(taken, free)
                broken = taken - placed
                glazier_shift = max(0, strip_idx - player.glazier)
                completes = placed > 0 and placed == strip.empty_count()

                sc = (
                    1.2 * placed
                    - 1.6 * broken
                    - first_penalty
                    - 0.25 * glazier_shift
                    + 0.15 * len(strip.tiles())
                )
                if completes:
                    cascade = WINDOW_POINTS[strip_idx] + sum(
                        WINDOW_POINTS[j]
                        for j in range(strip_idx + 1, NUM_STRIPS)
                        if any(player.windows[j])
                    )
                    bonus = sum(1 for t in strip.filled_spaces if t == round_color) + (
                        placed if m.color == round_color else 0
                    )
                    k = strip_idx // 2
                    orn_filled = (
                        sum(
                            1
                            for s in (player.windows[2 * k] + player.windows[2 * k + 1])
                            if s is not None
                        )
                        + 1
                    )
                    orn_gain = ORNAMENT_BONUS[orn_filled] - ORNAMENT_BONUS[orn_filled - 1]
                    sc += cascade + bonus + 0.6 * orn_gain

        if sc > best_score + 1e-6:
            best_score = sc
            best_moves = [m]
        elif abs(sc - best_score) <= 1e-6:
            best_moves.append(m)

    return rng.choice(best_moves)

class AzulSintraEnvV2(AECEnv):
    """Entorno multi-agente PettingZoo AEC para Azul: Vitrales de Sintra (v2)."""

    metadata = {"render_modes": ["ansi"], "name": "azul_sintra_v2"}

    def __init__(self, render_mode: str | None = None):
        super().__init__()
        self.render_mode = render_mode
        self.possible_agents = ["player_0", "player_1"]
        self.agents: list[str] = []

        obs_box = gym.spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(OBS_DIM,),
            dtype=np.float32,
        )
        mask_box = gym.spaces.Box(
            low=0,
            high=1,
            shape=(NUM_ACTIONS,),
            dtype=np.int8,
        )
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
        first_player = 0
        if options and "first_player" in options:
            first_player = int(options["first_player"])

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
        return {
            "observation": obs,
            "action_mask": self.action_mask(agent),
        }

    PENALIZACION_CRISTAL = (0, 1, 1, 2, 4, 6, 8, 11, 14, 15, 18, 22, 26)

    @classmethod
    def _costo_cristal_roto(cls, player) -> float:
        """Obtiene la penalización acumulada actual de cristal roto del jugador."""
        casilla = int(getattr(player, "broken", getattr(player, "broken_glass", 0)))
        idx = max(0, min(casilla, len(cls.PENALIZACION_CRISTAL) - 1))
        return float(cls.PENALIZACION_CRISTAL[idx])

    def _shaping_bonus(
        self,
        move: Move,
        prev_score: int,
        new_score: int,
        delta_costo_roto: float,
    ) -> float:
        """Señal densa que castiga inmediatamente el avance en cristal roto."""
        if delta_costo_roto > 0.0:
            # Penaliza proporcionalmente el daño real cometido en este turno
            return -0.35 * delta_costo_roto
        if move == Move.reset_glazier():
            return -0.10
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
            self._costo_cristal_roto(self.game.players[0]),
            self._costo_cristal_roto(self.game.players[1]),
        ]

        move = decode_action(int(action))
        self.game.apply_move(move)

        new_scores = [self.game.players[0].score, self.game.players[1].score]
        new_rotos = [
            self._costo_cristal_roto(self.game.players[0]),
            self._costo_cristal_roto(self.game.players[1]),
        ]

        # Incremento inmediato de penalización por cristal roto en este turno
        delta_roto_0 = new_rotos[0] - prev_rotos[0]
        delta_roto_1 = new_rotos[1] - prev_rotos[1]

        # Antes de game_over se aplica el costo potencial inmediato;
        # al cerrar game_over el motor ya descontó el cristal roto en new_scores,
        # por lo que se evita descontarlo dos veces.
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


class AzulGymWrapperV2(gym.Env):
    """Envoltorio Gymnasium de un jugador sobre AzulSintraEnvV2 con currículo de oponentes."""

    metadata = {"render_modes": ["ansi"]}
    _cached_v1_model = None
    _v1_load_attempted = False
    _cached_v1_obs_fn = None

    def __init__(
        self,
        opponent_mode: str = "curriculum",
        curriculum: tuple[float, float, float] = (0.45, 0.35, 0.20),
    ):
        super().__init__()
        self.aec_env = AzulSintraEnvV2()
        self.opponent_mode = opponent_mode
        self.curriculum = curriculum

        self.observation_space = gym.spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(OBS_DIM,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(NUM_ACTIONS)

        self.rng = random.Random()
        self.agent_idx = 0
        self.agent_name = "player_0"
        self.opp_name = "player_1"
        self.current_opponent = "heuristic"

    @classmethod
    def _get_v1_model(cls):
        if cls._v1_load_attempted:
            return cls._cached_v1_model
        cls._v1_load_attempted = True
        model_v1_path = Path(__file__).resolve().parent.parent / "models_checkpoint" / "ppo_azul_sintra_v1.zip"
        if not model_v1_path.exists():
            return None
        try:
            from sb3_contrib import MaskablePPO
            cls._cached_v1_model = MaskablePPO.load(str(model_v1_path), device="cpu")
        except Exception:
            cls._cached_v1_model = None
        return cls._cached_v1_model

    def _choose_opponent(self) -> str:
        if self.opponent_mode != "curriculum":
            return self.opponent_mode
        r = self.rng.random()
        p_heur, p_v1, _ = self.curriculum
        if r < p_heur:
            return "heuristic"
        if r < p_heur + p_v1 and self._get_v1_model() is not None:
            return "v1"
        return "random"

    def _resolve_v1_obs_fn(cls):
        if cls._cached_v1_obs_fn is not None:
            return cls._cached_v1_obs_fn

        import importlib

        mod = importlib.import_module("src.env_azul")
        for fn_name in (
            "build_observation",
            "get_observation",
            "encode_observation",
            "extract_observation",
            "build_obs",
            "make_observation",
        ):
            fn = getattr(mod, fn_name, None)
            if callable(fn):
                cls._cached_v1_obs_fn = fn
                return cls._cached_v1_obs_fn

        aec_cls = getattr(mod, "AzulSintraEnv", None)
        if aec_cls is None:
            for obj in vars(mod).values():
                if isinstance(obj, type) and issubclass(obj, AECEnv) and obj is not AECEnv:
                    aec_cls = obj
                    break

        if aec_cls is not None:
            env_v1 = aec_cls()

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

    def _select_opponent_action(self) -> int:
        game = self.aec_env.game
        legales = game.legal_moves()

        if self.current_opponent == "heuristic":
            return encode_move(heuristic_move(game, self.rng))

        if self.current_opponent == "v1":
            model_v1 = self._get_v1_model()
            obs_fn = self._resolve_v1_obs_fn()
            if model_v1 is not None and obs_fn is not None:
                opp_idx = 1 - self.agent_idx
                obs_v1 = obs_fn(game, opp_idx)
                mask = self.aec_env.action_mask(self.opp_name).astype(bool)
                action, _ = model_v1.predict(obs_v1, action_masks=mask, deterministic=False)
                return int(action)

        return encode_move(self.rng.choice(legales))

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
            opp_action = self._select_opponent_action()
            self.aec_env.step(opp_action)

        info = {"opponent": self.current_opponent, "seat": self.agent_idx}
        return self._get_obs(), info

    def step(self, action: int):
        self.aec_env.step(int(action))
        total_reward = float(self.aec_env.rewards.get(self.agent_name, 0.0))

        while not self.aec_env.game.game_over and self.aec_env.agent_selection != self.agent_name:
            opp_action = self._select_opponent_action()
            self.aec_env.step(opp_action)
            total_reward += float(self.aec_env.rewards.get(self.agent_name, 0.0))

        terminated = bool(self.aec_env.game.game_over)
        truncated = False
        info = {"opponent": self.current_opponent, "seat": self.agent_idx}
        return self._get_obs(), total_reward, terminated, truncated, info
