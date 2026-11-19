"""Partida en consola contra la IA v4 (MaskablePPO + Lookahead K=6).

Ejecutar desde la raíz del proyecto:
    python -m scripts.contra_maquina_ia_v4
    python -m scripts.contra_maquina_ia_v4 Ana
    python -m scripts.contra_maquina_ia_v4 Ana 7
    python -m scripts.contra_maquina_ia_v4 --modelo models_checkpoint/ppo_azul_sintra_v4.zip --k 6
    python -m scripts.contra_maquina_ia_v4 --demo-heuristica --json partida_historial.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.jugar import avisar, describir, mostrar_turno, pedir_por_consola
from src.azulvitral import AzulVitralGame, Move, Player
from src.env_azul_v2 import decode_action, heuristic_move
from src.env_azul_v3 import AzulSintraEnvV3
from src.env_azul_v4 import (
    AgenteHibridoV4,
    aplicar_jugada_segura,
    cargar_modelo_maskable_ppo,
    partida_terminada,
)
from src.impresion import print_final, print_game

MAQUINA = "Máquina_v4"
MODEL_V4_PATH = PROJECT_ROOT / "models_checkpoint" / "ppo_azul_sintra_v4.zip"
MODEL_V3_PATH = PROJECT_ROOT / "models_checkpoint" / "ppo_azul_sintra_v3.zip"
MODEL_V2_PATH = PROJECT_ROOT / "models_checkpoint" / "ppo_azul_sintra_v2.zip"
MODEL_V1_PATH = PROJECT_ROOT / "models_checkpoint" / "ppo_azul_sintra_v1.zip"


def jugador_al_azar(azar: random.Random):
    """Rival que elige uniformemente al azar entre las jugadas válidas."""

    def elegir(game: AzulVitralGame, player: Player) -> Move:
        jugadas = game.legal_moves()
        jugada = azar.choice(jugadas)
        print(
            f"\n{player.name} elige entre {len(jugadas)} jugadas válidas: "
            f"{describir(jugada)}"
        )
        return jugada

    return elegir


def jugador_heuristico(azar: random.Random):
    """Rival controlado por la política heurística táctica."""

    def elegir(game: AzulVitralGame, player: Player) -> Move:
        jugadas = game.legal_moves()
        jugada = heuristic_move(game, azar)
        print(
            f"\n{player.name} (Heurística) elige entre {len(jugadas)} jugadas válidas: "
            f"{describir(jugada)}"
        )
        return jugada

    return elegir


def jugador_ia_reactiva(modelo, etiqueta: str = "IA"):
    """Rival controlado por un checkpoint MaskablePPO en modo reactivo (K=1)."""
    helper = AzulSintraEnvV3()

    def elegir(game: AzulVitralGame, player: Player) -> Move:
        helper.game = game
        helper.agents = ["player_0", "player_1"]
        helper.terminations = {"player_0": False, "player_1": False}
        player_idx = next(i for i, p in enumerate(game.players) if p is player)
        ag = f"player_{player_idx}"
        helper.agent_selection = ag

        obs_dict = helper.observe(ag)
        act, _ = modelo.predict(
            obs_dict["observation"],
            action_masks=obs_dict["action_mask"].astype(bool),
            deterministic=True,
        )
        jugada = decode_action(int(act))
        jugadas = game.legal_moves()
        print(
            f"\n{player.name} ({etiqueta}) elige entre {len(jugadas)} jugadas válidas: "
            f"{describir(jugada)}"
        )
        return jugada

    return elegir


def jugador_ia_v4(agente: AgenteHibridoV4):
    """Rival controlado por IA v4 (MaskablePPO + 1-ply Lookahead K)."""

    def elegir(game: AzulVitralGame, player: Player) -> Move:
        player_idx = next(i for i, p in enumerate(game.players) if p is player)
        jugadas = game.legal_moves()
        t0 = time.perf_counter()
        jugada = agente.elegir_jugada(game, player_idx)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        print(
            f"\n{player.name} (IA v4 K={agente.k_lookahead}, {dt_ms:.1f} ms) "
            f"elige entre {len(jugadas)} jugadas válidas: {describir(jugada)}"
        )
        return jugada

    return elegir


def _serializar_turno(
    game: AzulVitralGame, turno: int, actor: Player, jugada: Move
) -> Dict[str, Any]:
    return {
        "turno": turno,
        "ronda": int(getattr(game, "round_number", getattr(game, "round", 1))),
        "jugador": actor.name,
        "accion": describir(jugada),
        "jugadores": [
            {
                "nombre": p.name,
                "score": int(p.score),
                "broken_position": int(getattr(p, "broken_position", 1)),
            }
            for p in game.players
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Partida en consola contra IA v4 (models_checkpoint/ppo_azul_sintra_v4.zip)."
    )
    parser.add_argument(
        "posicionales",
        nargs="*",
        help="Opcional: [nombre_jugador] [semilla]",
    )
    parser.add_argument(
        "--modelo",
        type=str,
        default=str(MODEL_V4_PATH),
        help="Ruta al checkpoint .zip de IA v4 (por defecto models_checkpoint/ppo_azul_sintra_v4.zip).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=6,
        help="Candidatas Top-K para Lookahead 1-ply (1 = reactivo, 6 = híbrido).",
    )
    parser.add_argument(
        "--ia-empieza",
        action="store_true",
        help="Hace que IA v4 juegue en el primer asiento.",
    )
    parser.add_argument(
        "--demo-heuristica",
        action="store_true",
        help="Simula una partida automática entre IA v4 y la Heurística Táctica.",
    )
    parser.add_argument("--v3", action="store_true", help="Enfrentar al modelo v3.")
    parser.add_argument("--v2", action="store_true", help="Enfrentar al modelo v2.")
    parser.add_argument("--v1", action="store_true", help="Enfrentar al modelo v1.")
    parser.add_argument("--heuristica", action="store_true", help="Enfrentar a la heurística.")
    parser.add_argument("--azar", action="store_true", help="Enfrentar al rival aleatorio.")
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Exporta el historial de la partida a un archivo JSON.",
    )
    args = parser.parse_args()

    nombre_humano = "Jugador"
    semilla: Optional[int] = None
    if len(args.posicionales) >= 1:
        nombre_humano = args.posicionales[0]
    if len(args.posicionales) >= 2 and args.posicionales[1].isdigit():
        semilla = int(args.posicionales[1])

    azar = random.Random(semilla)

    if args.demo_heuristica:
        agente_v4 = AgenteHibridoV4(ruta_modelo=args.modelo, k_lookahead=args.k)
        nombres = [f"{MAQUINA}_K{args.k}", "Heurística"] if args.ia_empieza else ["Heurística", f"{MAQUINA}_K{args.k}"]
        politicas = (
            [jugador_ia_v4(agente_v4), jugador_heuristico(azar)]
            if args.ia_empieza
            else [jugador_heuristico(azar), jugador_ia_v4(agente_v4)]
        )
    else:
        if args.azar:
            nombre_rival = "Azar"
            fn_rival = jugador_al_azar(azar)
        elif args.heuristica:
            nombre_rival = "Heurística"
            fn_rival = jugador_heuristico(azar)
        elif args.v3:
            nombre_rival = "Máquina_v3"
            fn_rival = jugador_ia_reactiva(cargar_modelo_maskable_ppo(MODEL_V3_PATH), "IA v3")
        elif args.v2:
            nombre_rival = "Máquina_v2"
            fn_rival = jugador_ia_reactiva(cargar_modelo_maskable_ppo(MODEL_V2_PATH), "IA v2")
        elif args.v1:
            nombre_rival = "Máquina_v1"
            fn_rival = jugador_ia_reactiva(cargar_modelo_maskable_ppo(MODEL_V1_PATH), "IA v1")
        else:
            agente_v4 = AgenteHibridoV4(ruta_modelo=args.modelo, k_lookahead=args.k)
            nombre_rival = f"{MAQUINA}_K{args.k}"
            fn_rival = jugador_ia_v4(agente_v4)

        if args.ia_empieza:
            nombres = [nombre_rival, nombre_humano]
            politicas = [fn_rival, pedir_por_consola]
        else:
            nombres = [nombre_humano, nombre_rival]
            politicas = [pedir_por_consola, fn_rival]

    game = AzulVitralGame(player_names=nombres, seed=semilla)
    historial: List[Dict[str, Any]] = []
    turno = 0

    print(f"Partida iniciada: {nombres[0]} vs {nombres[1]} (modelo: {args.modelo})")

    while not partida_terminada(game):
        actual = game.get_current_player()
        actual_idx = next(i for i, p in enumerate(game.players) if p is actual)

        if not args.demo_heuristica:
            print_game(game)
            print(f"--- Turno de {actual.name} ---")

        jugada = politicas[actual_idx](game, actual)
        aplicar_jugada_segura(game, actual_idx, jugada)

        turno += 1
        historial.append(_serializar_turno(game, turno, actual, jugada))

    print_final(game)

    if args.json:
        out_path = Path(args.json)
        if not out_path.is_absolute():
            out_path = PROJECT_ROOT / out_path
        payload = {
            "meta": {
                "jugador_0": nombres[0],
                "jugador_1": nombres[1],
                "modelo": str(args.modelo),
                "k_lookahead": args.k,
                "seed": semilla,
                "puntaje_final": [int(game.players[0].score), int(game.players[1].score)],
            },
            "turnos": historial,
        }
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[OK] Historial exportado a: {out_path}")


if __name__ == "__main__":
    main()
