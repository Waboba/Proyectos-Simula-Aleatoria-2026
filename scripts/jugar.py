"""Partida en consola entre dos personas.

    python3 jugar.py
    python3 jugar.py Ana Beto           (con nombres)
    python3 jugar.py Ana Beto 7         (y una semilla, para repetir la partida)

Al empezar pregunta si es una partida nueva o una que ya va en curso; en ese
caso se escribe el estado o se carga de un archivo (ver estado.py).

Cómo se escribe una jugada:

    3 azul 2             toma las azules de la fábrica 3 y las pone en la tira 2
    c naranjo 4          toma las naranjas del centro y las pone en la tira 4
    1 rosado 5 rosado    si la tira 5 se completa, deja una rosada en la ventana
    2 amarillo -         sin ninguna tira alcanzable: se rompen todas
    b                    opción B, el vidriero vuelve a la izquierda
    jugadas              lista todas las jugadas válidas
    salir                termina el programa

Los colores se pueden abreviar con dos letras: na, am, rs, az, tr.
"""

import sys
from pathlib import Path

# Añadir la raíz del proyecto (Azul-Vitral) al sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.azulvitral import CENTER, AzulVitralGame, InvalidMoveError, Move, Player
from src.estado import abrir_partida, leer_color
from src.impresion import print_final, print_game, turn_text
from src.partida import TwoPlayerMatch


def leer_jugada(texto: str) -> Move:
    """Convierte lo que escribe el jugador en una Move. Lanza ValueError si no se entiende."""
    partes = texto.split()
    if not partes:
        raise ValueError("Escribe una jugada.")
    if partes[0].lower() == "b" and len(partes) == 1:
        return Move.reset_glazier()
    if len(partes) not in (3, 4):
        raise ValueError("Una jugada es: origen color tira [pieza para la ventana].")
    origen, color, tira = partes[:3]
    if origen.lower() in ("c", "centro"):
        origen = CENTER
    elif origen.isdigit():
        origen = int(origen)
    else:
        raise ValueError(f"El origen {origen!r} no es una fábrica (1 a 5) ni el centro (c).")
    if tira == "-":
        tira = None
    elif tira.isdigit():
        tira = int(tira)
    else:
        raise ValueError(f"La tira {tira!r} tiene que ser un número de 1 a 8, o - si no hay ninguna.")
    ventana = leer_color(partes[3]) if len(partes) == 4 else None
    return Move.take(origen, leer_color(color), tira, ventana)


def describir(move: Move) -> str:
    if move.option == "B":
        return "b"
    origen = "c" if move.source == CENTER else str(move.source)
    tira = "-" if move.strip is None else str(move.strip)
    texto = f"{origen} {move.color.name.lower()} {tira}"
    if move.window_color is not None:
        texto += f" {move.window_color.name.lower()}"
    return texto


def pedir_por_consola(game: AzulVitralGame, player: Player) -> Move:
    while True:
        texto = input(f"\n{player.name}, tu jugada: ").strip()
        if texto.lower() == "salir":
            sys.exit(0)
        if texto.lower() == "jugadas":
            print("  " + "\n  ".join(describir(m) for m in game.legal_moves()))
            continue
        try:
            return leer_jugada(texto)
        except ValueError as err:
            print(f"  {err}")


def avisar(player: Player, move: Move, err: InvalidMoveError) -> None:
    print(f"  Esa jugada no vale: {err}")


def mostrar_turno(report, game: AzulVitralGame) -> None:
    print("\n-> " + turn_text(report))
    if not game.game_over:
        print()
        print_game(game)


def main():
    nombres = sys.argv[1:3] if len(sys.argv) >= 3 else ["Jugador 1", "Jugador 2"]
    semilla = int(sys.argv[3]) if len(sys.argv) >= 4 else None
    game = abrir_partida(nombres, seed=semilla)
    partida = TwoPlayerMatch(
        [p.name for p in game.players],
        inputs={p.name: pedir_por_consola for p in game.players},
        on_turn=mostrar_turno,
        on_invalid=avisar,
        game=game,
    )
    print_game(partida.game)
    partida.run()
    print()
    print_game(partida.game)
    print()
    print_final(partida.game)


if __name__ == "__main__":
    main()
