"""Ejemplo: una partida completa entre Ana y Beto con jugadas fijas.

    python3 ejemplo_partida.py             muestra la mesa al empezar cada ronda
    python3 ejemplo_partida.py --detalle   la muestra después de cada turno

Las jugadas las armó generar_ejemplo.py para la semilla 2026. Con otra
semilla las fábricas salen distintas y la primera jugada que no calce se
rechaza con su motivo.
"""

import sys
from pathlib import Path
# Añadir la raíz del proyecto (Azul-Vitral) al sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from src.azulvitral import Color, Move
from src.impresion import print_final, print_game, print_turn
from src.partida import TwoPlayerMatch

SEMILLA = 2026

# Opción A: (origen, color, tira), o (origen, color, tira, pieza para la ventana)
# si la jugada completa la tira. El origen es una fábrica (1 a 5) o "centro".
# Opción B: "B".
JUGADAS = {
    "Ana": [
        # ronda 1
        (3, "AZUL", 1), (2, "NARANJO", 1), (5, "AZUL", 1), ("centro", "NARANJO", 1, "NARANJO"),
        ("centro", "TRANSPARENTE", 5),
        # ronda 2
        (4, "ROSADO", 5), (5, "AZUL", 7), (3, "NARANJO", 8), "B", ("centro", "AZUL", 6),
        # ronda 3
        (2, "ROSADO", 7, "ROSADO"), ("centro", "AZUL", 7), (3, "AZUL", 7), "B", ("centro", "AMARILLO", 3),
        # ronda 4
        (2, "AMARILLO", 3, "AMARILLO"), (3, "ROSADO", 5, "ROSADO"), (5, "AMARILLO", 6),
        ("centro", "AMARILLO", 6, "AMARILLO"), ("centro", "AZUL", 6),
        # ronda 5
        (1, "AZUL", 7, "AZUL"), (4, "NARANJO", 8), "B", (2, "NARANJO", 1), ("centro", "NARANJO", 1, "NARANJO"),
        ("centro", "TRANSPARENTE", 4),
        # ronda 6
        (3, "ROSADO", 8, "NARANJO"), (2, "TRANSPARENTE", 8), (5, "TRANSPARENTE", 8),
        ("centro", "TRANSPARENTE", 8), "B", ("centro", "ROSADO", 2),
    ],
    "Beto": [
        # ronda 1
        (1, "ROSADO", 2), (4, "ROSADO", 2), ("centro", "AMARILLO", 2), ("centro", "ROSADO", 2, "AMARILLO"),
        ("centro", "AZUL", 7),
        # ronda 2
        (2, "TRANSPARENTE", 7, "AZUL"), ("centro", "AMARILLO", 7), (1, "NARANJO", 7),
        ("centro", "NARANJO", 8), "B", ("centro", "TRANSPARENTE", 6),
        # ronda 3
        (5, "AMARILLO", 6, "AMARILLO"), (1, "ROSADO", 7, "NARANJO"), ("centro", "NARANJO", 8),
        "B", (4, "ROSADO", 3), ("centro", "TRANSPARENTE", 6),
        # ronda 4
        (1, "TRANSPARENTE", 6), (4, "TRANSPARENTE", 6), "B", ("centro", "NARANJO", 1),
        ("centro", "ROSADO", 4),
        # ronda 5
        (3, "TRANSPARENTE", 6, "TRANSPARENTE"), (5, "NARANJO", 8), "B", ("centro", "AMARILLO", 1, "NARANJO"),
        ("centro", "ROSADO", 3, "ROSADO"), ("centro", "AZUL", 4),
        # ronda 6
        (1, "ROSADO", 4, "ROSADO"), (4, "NARANJO", 8, "NARANJO"), ("centro", "AZUL", 8),
        ("centro", "NARANJO", 8), "B", ("centro", "AMARILLO", 1, "AMARILLO"),
    ],
}


def to_move(jugada):
    if jugada == "B":
        return Move.reset_glazier()
    origen, color, tira, *ventana = jugada
    return Move.take(origen, Color[color], tira, Color[ventana[0]] if ventana else None)


def jugador_fijo(nombre, jugadas):
    """Entrada de un jugador que va entregando sus jugadas en orden."""
    pendientes = iter(jugadas)

    def siguiente(game, player):
        jugada = next(pendientes, None)
        if jugada is None:
            raise RuntimeError(f"Se acabaron las jugadas de {nombre}.")
        return to_move(jugada)

    return siguiente


def main():
    detalle = "--detalle" in sys.argv[1:]

    def al_jugar(report, game):
        print_turn(report)
        if detalle or (report.round_ended and not report.game_over):
            print()
            print_game(game)
            print()

    partida = TwoPlayerMatch(
        ["Ana", "Beto"],
        inputs={nombre: jugador_fijo(nombre, lista) for nombre, lista in JUGADAS.items()},
        boards=("rojo", "amarillo"),
        first_player="Ana",
        seed=SEMILLA,
        on_turn=al_jugar,
    )
    print_game(partida.game)
    print()
    partida.run()
    print()
    print_game(partida.game)
    print()
    print_final(partida.game)


if __name__ == "__main__":
    main()
