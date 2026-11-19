"""Exporta una partida completa a JSON para animarla en visualizador_azul.html.

Ejecutar desde la raíz del proyecto:
    python -m scripts.exportar_partida_web
    python -m scripts.exportar_partida_web --rival heuristica --seed 42
    python -m scripts.exportar_partida_web --rival v3 --seed 7
    python -m scripts.exportar_partida_web --rival v2 --k 6 --salida partida_historial.json
"""

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.azulvitral import (
    JOKER,
    NUM_STRIPS,
    WINDOW_POINTS,
    AzulVitralGame,
    Color,
    Move,
    Player,
    TurnReport,
)
from src.env_azul_v2 import AzulSintraEnvV2, decode_action, heuristic_move
from src.env_azul_v3 import AzulSintraEnvV3
from src.env_azul_v4 import select_action_v4
from src.impresion import result_text, turn_text

MODEL_V4_PATH = PROJECT_ROOT / "models_checkpoint" / "ppo_azul_sintra_v4.zip"
MODEL_V3_PATH = PROJECT_ROOT / "models_checkpoint" / "ppo_azul_sintra_v3.zip"
MODEL_V2_PATH = PROJECT_ROOT / "models_checkpoint" / "ppo_azul_sintra_v2.zip"
MODEL_V1_PATH = PROJECT_ROOT / "models_checkpoint" / "ppo_azul_sintra_v1.zip"


def _color_name(c) -> str | None:
    if c is None:
        return None
    if c == JOKER:
        return "COMODIN"
    return c.name if isinstance(c, Color) else str(c)


def _etiqueta_emergente(report: TurnReport, player_before: dict) -> str | None:
    """Detecta patrones estratégicos relevantes para destacar en la presentación."""
    if report.move.option == "B":
        return "Reinicio táctico del vidriero (Opción B)"
    if report.completed and report.window_points >= 5:
        return f"Cascada de ventanas (+{report.window_points} pts)"
    if report.completed and report.color_bonus >= 2:
        return f"Sinergia color de ronda (+{report.color_bonus} pts)"
    if report.strip_removed:
        return "Doble ventana completada (tira retirada)"
    if report.taken >= 3 and report.broken == 0 and not report.took_first_player_token:
        return f"Toma limpia de {report.taken} piezas (0 cristal roto)"
    if report.broken >= 3:
        return f"Sobrecarga de cristal roto (-{report.broken} casillas)"
    return None


def _serializar_jugador(p: Player) -> dict:
    tiras = []
    for idx, s in enumerate(p.strips):
        tiras.append(
            {
                "columna": idx + 1,
                "cara": "anv" if s.current_side == 0 else "rev",
                "retirada": bool(s.is_removed),
                "patron": [_color_name(sp) for sp in s.pattern()],
                "piezas": [_color_name(tile) for tile in s.filled_spaces],
            }
        )
    ventanas = []
    for idx, w in enumerate(p.windows):
        ventanas.append(
            {
                "columna": idx + 1,
                "superior": _color_name(w[0]),
                "inferior": _color_name(w[1]),
                "valor": WINDOW_POINTS[idx],
            }
        )
    ornamentos = []
    for k in range(NUM_STRIPS // 2):
        filled = sum(1 for sp in p.windows[2 * k] + p.windows[2 * k + 1] if sp is not None)
        ornamentos.append(filled)

    return {
        "nombre": p.name,
        "tablero": p.board,
        "puntos": int(p.score),
        "cristal_roto": int(p.broken_position),
        "penalizacion_roto": int(p.broken_penalty()),
        "vidriero": int(p.glazier),
        "tiene_loseta_inicial": bool(p.has_first_player_token),
        "tiras": tiras,
        "ventanas": ventanas,
        "ornamentos": ornamentos,
    }


def _serializar_estado(
    game: AzulVitralGame,
    turno: int,
    accion: dict | None = None,
) -> dict:
    final_data = None
    if game.game_over:
        detalles = {}
        for p in game.players:
            fs = game.final_scores[p.name]
            detalles[p.name] = {
                "puntos_ronda_6": fs.score_before,
                "piezas_sobrantes": fs.leftover_tiles,
                "puntos_sobrantes": fs.leftover_points,
                "penalizacion_cristal": fs.broken_penalty,
                "puntos_ornamentos": list(fs.ornament_points),
                "total": fs.total,
            }
        final_data = {
            "resumen": result_text(game),
            "ganador": game.winner().name if game.winner() else "Empate",
            "detalles": detalles,
        }

    return {
        "turno": turno,
        "ronda": int(game.round),
        "color_ronda": _color_name(game.current_round_color()),
        "marcador_ronda": [_color_name(c) for c in game.round_marker],
        "bolsa_count": len(game.bag),
        "torre_count": len(game.tower),
        "jugador_activo": int(game.current),
        "fabricas": [[_color_name(c) for c in f.tiles] for f in game.factories],
        "centro": {
            "piezas": [_color_name(c) for c in game.center.tiles],
            "tiene_loseta_inicial": bool(game.center.has_first_player_token),
        },
        "jugadores": [_serializar_jugador(p) for p in game.players],
        "accion": accion,
        "final": final_data,
    }


def _crear_politica_rival(rival: str, rng: random.Random):
    from sb3_contrib import MaskablePPO

    if rival == "azar":
        return "Rival Aleatorio", lambda g, idx: rng.choice(g.legal_moves())

    if rival == "heuristica":
        return "Heurística Táctica", lambda g, idx: heuristic_move(g, rng)

    if rival == "v1":
        import src.env_azul as env_v1_mod

        modelo = MaskablePPO.load(str(MODEL_V1_PATH), device="cpu")
        helper = env_v1_mod.AzulSintraEnv()
        decode_v1 = getattr(helper, "decode_action", getattr(env_v1_mod, "decode_action", None))

        def policy_v1(g: AzulVitralGame, idx: int) -> Move:
            helper.game = g
            ag = f"player_{idx}"
            helper.agent_selection = ag
            obs = helper.observe(ag)
            act, _ = modelo.predict(obs["observation"], action_masks=obs["action_mask"].astype(bool), deterministic=True)
            return decode_v1(int(act))

        return "IA v1 (PPO Base)", policy_v1

    if rival == "v2":
        modelo = MaskablePPO.load(str(MODEL_V2_PATH), device="cpu")
        helper = AzulSintraEnvV2()

        def policy_v2(g: AzulVitralGame, idx: int) -> Move:
            helper.game = g
            ag = f"player_{idx}"
            helper.agent_selection = ag
            obs = helper.observe(ag)
            act, _ = modelo.predict(obs["observation"], action_masks=obs["action_mask"].astype(bool), deterministic=True)
            return decode_action(int(act))

        return "IA v2 (Self-Play)", policy_v2

    if rival == "v3":
        modelo = MaskablePPO.load(str(MODEL_V3_PATH), device="cpu")
        helper = AzulSintraEnvV3()

        def policy_v3(g: AzulVitralGame, idx: int) -> Move:
            helper.game = g
            helper.agents = ["player_0", "player_1"]
            helper.terminations = {"player_0": False, "player_1": False}
            ag = f"player_{idx}"
            helper.agent_selection = ag
            obs = helper.observe(ag)
            act, _ = modelo.predict(obs["observation"], action_masks=obs["action_mask"].astype(bool), deterministic=True)
            return decode_action(int(act))

        return "IA v3 (BC + PPO)", policy_v3

    raise ValueError(f"Rival no reconocido: {rival}")


def exportar_partida(
    rival: str = "heuristica",
    seed: int = 42,
    top_k: int = 6,
    first_player: int = 0,
    salida: Path = Path("partida_historial.json"),
) -> dict:
    from sb3_contrib import MaskablePPO

    if not MODEL_V4_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró {MODEL_V4_PATH}. Entrena primero con scripts/train_rl_v4.py."
        )

    rng = random.Random(seed)
    modelo_v4 = MaskablePPO.load(str(MODEL_V4_PATH), device="cpu")
    nombre_v4 = f"IA v4 (Lookahead K={top_k})" if top_k > 1 else "IA v4 (Reactivo K=1)"
    nombre_rival, policy_rival = _crear_politica_rival(rival, rng)

    game = AzulVitralGame(
        [nombre_v4, nombre_rival],
        boards=("rojo", "amarillo"),
        first_player=first_player,
        seed=seed,
    )

    frames = [_serializar_estado(game, turno=0, accion=None)]
    turno = 0

    while not game.game_over:
        turno += 1
        actor_idx = game.current
        prev_player = _serializar_jugador(game.players[actor_idx])

        if actor_idx == 0:
            move = select_action_v4(modelo_v4, game, actor_idx, top_k=top_k)
        else:
            move = policy_rival(game, actor_idx)

        report = game.apply_move(move)
        accion_info = {
            "jugador_idx": actor_idx,
            "jugador_nombre": report.player,
            "opcion": move.option,
            "origen": move.source,
            "color": _color_name(move.color),
            "tira": move.strip,
            "color_ventana": _color_name(move.window_color),
            "tomadas": int(report.taken),
            "colocadas": int(report.placed),
            "rotas": int(report.broken),
            "tomo_loseta_inicial": bool(report.took_first_player_token),
            "completo_tira": bool(report.completed),
            "bono_color": int(report.color_bonus),
            "puntos_ventana": int(report.window_points),
            "tira_retirada": bool(report.strip_removed),
            "desbordes": int(report.overflows),
            "fin_ronda": bool(report.round_ended),
            "fin_partida": bool(report.game_over),
            "descripcion": turn_text(report),
            "etiqueta_emergente": _etiqueta_emergente(report, prev_player),
        }
        frames.append(_serializar_estado(game, turno=turno, accion=accion_info))

    historial = {
        "meta": {
            "titulo": "Azul: Vitrales de Sintra — Demostración RL",
            "agente_principal": nombre_v4,
            "rival": nombre_rival,
            "seed": seed,
            "total_turnos": turno,
            "ganador": game.winner().name if game.winner() else "Empate",
            "marcador_final": [game.players[0].score, game.players[1].score],
        },
        "frames": frames,
    }

    salida.parent.mkdir(parents=True, exist_ok=True)
    with open(salida, "w", encoding="utf-8") as fh:
        json.dump(historial, fh, ensure_ascii=False, indent=2)

    print(f"Partida exportada en: {salida}")
    print(f"  Enfrentamiento : {nombre_v4} vs. {nombre_rival} (seed={seed})")
    print(f"  Total de turnos: {turno}")
    print(f"  Resultado      : {result_text(game)}")
    return historial


def main():
    parser = argparse.ArgumentParser(description="Exporta una partida RL a JSON para D3.js")
    parser.add_argument(
        "--rival",
        choices=["heuristica", "v3", "v2", "v1", "azar"],
        default="heuristica",
        help="Oponente contra el cual juega IA v4 (default: heuristica)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria (default: 42)")
    parser.add_argument("--k", type=int, default=6, help="Top-K lookahead de IA v4 (default: 6)")
    parser.add_argument(
        "--first",
        type=int,
        choices=[0, 1],
        default=0,
        help="Jugador inicial: 0 = IA v4, 1 = Rival (default: 0)",
    )
    parser.add_argument(
        "--salida",
        type=Path,
        default=Path("partida_historial.json"),
        help="Archivo JSON de salida (default: partida_historial.json)",
    )
    args = parser.parse_args()
    exportar_partida(
        rival=args.rival,
        seed=args.seed,
        top_k=args.k,
        first_player=args.first,
        salida=args.salida,
    )


if __name__ == "__main__":
    main()
