"""
Partida interactiva contra IA v4 (MaskablePPO + Lookahead K=6) o simulación
de demostración frente a la Heurística Táctica con exportación JSON opcional.

Uso:
  python -m scripts.contra_maquina_ia_v4
  python -m scripts.contra_maquina_ia_v4 --k 6 --ia-empieza
  python -m scripts.contra_maquina_ia_v4 --demo-heuristica --json partida_historial.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.azulvitral import AzulVitralGame, Move
from src.env_azul_v2 import heuristic_move
from src.env_azul_v4 import AgenteHibridoV4

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELO = "models_checkpoint/ppo_azul_sintra_v4.zip"


def _serializar_estado_turno(
    game: AzulVitralGame,
    turno: int,
    actor_idx: int,
    actor_nombre: str,
    accion_desc: str,
) -> Dict[str, Any]:
    jugadores_data = []
    for idx, p in enumerate(game.players):
        jugadores_data.append(
            {
                "id": idx,
                "nombre": actor_nombre if idx == actor_idx else f"Jugador {idx}",
                "score": int(p.score),
                "glazier_pos": int(getattr(p, "glazier_pos", 0)),
                "broken_position": int(getattr(p, "broken_position", 1)),
            }
        )
    return {
        "turno": turno,
        "ronda": int(getattr(game, "round_num", 1)),
        "jugador_idx": actor_idx,
        "jugador": actor_nombre,
        "accion": accion_desc,
        "jugadores": jugadores_data,
    }


def _mostrar_tablero(game: AzulVitralGame, nombres: List[str]) -> None:
    print("\n" + "=" * 68)
    ronda = getattr(game, "round_num", "?")
    print(f"RONDA {ronda} | Marcador -> {nombres[0]}: {game.players[0].score} pts | {nombres[1]}: {game.players[1].score} pts")
    print("=" * 68)

    for i, fab in enumerate(game.factories):
        print(f"  Fábrica {i + 1}: {list(fab)}")
    print(f"  Centro   : {list(game.center)} (Ficha inicial disponible: {getattr(game, 'first_player_token_in_center', False)})")

    for idx, p in enumerate(game.players):
        print(
            f"\n[{nombres[idx]}] Puntos: {p.score} | Vidriero en tira: {getattr(p, 'glazier_pos', 0) + 1} "
            f"| Cristal roto pos: {getattr(p, 'broken_position', 1)}"
        )
        for s_idx, strip in enumerate(p.pattern_strips):
            if strip is not None:
                print(f"   Tira {s_idx + 1}: {strip}")


def _pedir_jugada_humana(game: AzulVitralGame, player_idx: int) -> Move:
    legal_moves = game.get_legal_moves(player_idx)
    print("\nJugadas legales disponibles:")
    for i, mv in enumerate(legal_moves):
        print(f"  [{i}] {mv}")

    while True:
        entrada = input(f"\nElige el índice de tu jugada (0..{len(legal_moves) - 1}): ").strip()
        if entrada.isdigit():
            idx = int(entrada)
            if 0 <= idx < len(legal_moves):
                return legal_moves[idx]
        print("Entrada inválida. Ingresa un número dentro del rango indicado.")


def jugar_contra_ia(
    ruta_modelo: str = DEFAULT_MODELO,
    k_lookahead: int = 6,
    ia_empieza: bool = False,
    demo_heuristica: bool = False,
    ruta_json: Optional[str] = None,
    seed: int = 42,
) -> None:
    agente = AgenteHibridoV4(ruta_modelo=ruta_modelo, k_lookahead=k_lookahead)
    game = AzulVitralGame(num_players=2, seed=seed)

    ia_idx = 0 if ia_empieza else 1
    rival_idx = 1 - ia_idx
    nombres = ["", ""]
    nombres[ia_idx] = f"IA v4 (K={k_lookahead})"
    nombres[rival_idx] = "Heurística Táctica" if demo_heuristica else "Humano"

    historial: List[Dict[str, Any]] = []
    turno = 0

    print(f"\nIniciando partida: {nombres[0]} vs {nombres[1]}")
    print(f"Checkpoint cargado: {agente.ruta_modelo}")

    while not game.is_game_over():
        actual = game.current_player_idx
        if not demo_heuristica:
            _mostrar_tablero(game, nombres)

        t0 = time.perf_counter()
        if actual == ia_idx:
            mv = agente.elegir_jugada(game, actual)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            print(f"\n>> {nombres[actual]} juega: {mv} ({dt_ms:.1f} ms)")
        else:
            if demo_heuristica:
                mv = heuristic_move(game, actual)
                print(f">> {nombres[actual]} juega: {mv}")
            else:
                mv = _pedir_jugada_humana(game, actual)

        game.apply_move(actual, mv)
        turno += 1
        historial.append(
            _serializar_estado_turno(
                game=game,
                turno=turno,
                actor_idx=actual,
                actor_nombre=nombres[actual],
                accion_desc=str(mv),
            )
        )

    s0, s1 = game.players[0].score, game.players[1].score
    print("\n" + "#" * 68)
    print("PARTIDA FINALIZADA")
    print(f"  {nombres[0]}: {s0} puntos")
    print(f"  {nombres[1]}: {s1} puntos")
    if s0 > s1:
        print(f"  Ganador: {nombres[0]}")
    elif s1 > s0:
        print(f"  Ganador: {nombres[1]}")
    else:
        print("  Resultado: Empate")
    print("#" * 68)

    if ruta_json:
        out_path = Path(ruta_json)
        if not out_path.is_absolute():
            out_path = PROJECT_ROOT / out_path
        payload = {
            "meta": {
                "jugador_0": nombres[0],
                "jugador_1": nombres[1],
                "k_lookahead": k_lookahead,
                "seed": seed,
                "puntaje_final": [int(s0), int(s1)],
            },
            "turnos": historial,
        }
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[OK] Historial exportado a: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Juega contra IA v4 o ejecuta una demostración vs Heurística."
    )
    parser.add_argument(
        "--modelo",
        type=str,
        default=DEFAULT_MODELO,
        help="Ruta al checkpoint .zip de IA v4 (por defecto models_checkpoint/ppo_azul_sintra_v4.zip).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=6,
        help="Número de jugadas candidatas Top-K para Lookahead 1-ply (1 = reactivo, 6 = híbrido).",
    )
    parser.add_argument(
        "--ia-empieza",
        action="store_true",
        help="Asigna el asiento 0 (jugador inicial) a IA v4.",
    )
    parser.add_argument(
        "--demo-heuristica",
        action="store_true",
        help="Ejecuta una partida automática entre IA v4 y la Heurística Táctica.",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Ruta de salida JSON para visualizar la partida en visualizador_azul.html.",
    )
    args = parser.parse_args()

    jugar_contra_ia(
        ruta_modelo=args.modelo,
        k_lookahead=args.k,
        ia_empieza=args.ia_empieza,
        demo_heuristica=args.demo_heuristica,
        ruta_json=args.json,
    )


if __name__ == "__main__":
    main()
