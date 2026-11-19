"""Arma las jugadas fijas que usa ejemplo_partida.py.

Juega una partida con una estrategia simple para los dos jugadores y
escribe la lista de jugadas en el formato del ejemplo:

    python3 generar_ejemplo.py          (semilla 2026, la del ejemplo)
    python3 generar_ejemplo.py 2030     (otra semilla)

La estrategia no busca jugar bien, solo que la partida muestre de todo:
tiras completas, ventanas en cascada, piezas rotas y la opción B.
"""

import copy
import sys
from pathlib import Path

# Añadir la raíz del proyecto (Azul-Vitral) al sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.azulvitral import CENTER, AzulVitralGame, Move

SEMILLA = 2026


def valor(game: AzulVitralGame, move: Move) -> float:
    """Qué tan buena parece una jugada, probándola en una copia de la partida."""
    yo = game.current
    antes = game.players[yo]
    prueba = copy.deepcopy(game)
    rep = prueba.apply_move(move)
    despues = prueba.players[yo]

    if move.option == "B":
        # Volver a la izquierda conviene más mientras más a la derecha esté.
        return 0.25 * antes.glazier - 1.2

    v = (despues.score - antes.score) - 0.6 * rep.broken + 0.35 * rep.placed
    if move.strip is not None:
        v -= 0.25 * max(0, (move.strip - 1) - antes.glazier)  # mover el vidriero cuesta
        tira = despues.strips[move.strip - 1]
        if not tira.is_removed and not rep.completed:
            v += 0.15 * len(tira.tiles())  # dejar una tira avanzada vale algo
    return v


def jugar(semilla: int):
    game = AzulVitralGame(["Ana", "Beto"], boards=("rojo", "amarillo"),
                          first_player="Ana", seed=semilla)
    registro = []
    while not game.game_over:
        jugadas = game.legal_moves()
        # Si dos jugadas valen lo mismo, gana la de la tira más a la izquierda.
        mejor = max(jugadas, key=lambda m: (valor(game, m), -(m.strip or 0)))
        registro.append((game.round, game.get_current_player().name, mejor))
        game.apply_move(mejor)
    return game, registro


def como_texto(move: Move) -> str:
    if move.option == "B":
        return '"B"'
    origen = '"centro"' if move.source == CENTER else str(move.source)
    partes = [origen, f'"{move.color.name}"', str(move.strip)]
    if move.window_color is not None:
        partes.append(f'"{move.window_color.name}"')
    return "(" + ", ".join(partes) + ")"


def main():
    semilla = int(sys.argv[1]) if len(sys.argv) > 1 else SEMILLA
    game, registro = jugar(semilla)
    print("JUGADAS = {")
    for jugador in ("Ana", "Beto"):
        print(f'    "{jugador}": [')
        ronda, linea = None, []
        for numero, nombre, move in registro:
            if nombre != jugador:
                continue
            if numero != ronda:
                if linea:
                    print("        " + ", ".join(linea) + ",")
                    linea = []
                print(f"        # ronda {numero}")
                ronda = numero
            linea.append(como_texto(move))
            if len(", ".join(linea)) > 70:
                print("        " + ", ".join(linea) + ",")
                linea = []
        if linea:
            print("        " + ", ".join(linea) + ",")
        print("    ],")
    print("}")
    a, b = game.players
    print(f"# {len(registro)} turnos. Final: {a.name} {a.score}, {b.name} {b.score}", file=sys.stderr)


if __name__ == "__main__":
    main()
