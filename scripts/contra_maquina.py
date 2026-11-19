"""Partida en consola contra la máquina.

    python3 contra_maquina.py
    python3 contra_maquina.py Ana        (con tu nombre)
    python3 contra_maquina.py Ana 7      (y una semilla, para repetir la partida)

La máquina no intenta ganar: en cada turno mira todas sus jugadas válidas y
elige una al azar, todas con la misma probabilidad.

Al empezar pregunta si es una partida nueva o una que ya va en curso. En una
partida en curso se escribe el estado (o se carga de un archivo) y después se
dice cuál de los dos jugadores eres tú; el otro lo juega la máquina.
Las jugadas se escriben igual que en jugar.py.
"""

import random
import sys
from pathlib import Path

# Añadir la raíz del proyecto (Azul-Vitral) al sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.azulvitral import AzulVitralGame, Move, Player
from src.estado import abrir_partida
from src.impresion import print_final, print_game
from scripts.jugar import avisar, describir, mostrar_turno, pedir_por_consola
from src.partida import TwoPlayerMatch

MAQUINA = "Máquina"


def jugador_al_azar(azar: random.Random):
    """Entrada de un jugador que elige al azar entre todas sus jugadas válidas."""
    def elegir(game: AzulVitralGame, player: Player) -> Move:
        jugadas = game.legal_moves()
        jugada = azar.choice(jugadas)
        print(f"\n{player.name} elige entre {len(jugadas)} jugadas válidas: {describir(jugada)}")
        return jugada
    return elegir


def elegir_humano(nombres) -> str:
    while True:
        texto = input(f"¿Cuál de los dos eres tú? ({nombres[0]} o {nombres[1]}): ").strip()
        if texto in nombres:
            return texto
        print("  Escribe uno de los dos nombres.")


def main():
    humano = sys.argv[1] if len(sys.argv) >= 2 else "Jugador"
    semilla = int(sys.argv[2]) if len(sys.argv) >= 3 else None
    if humano == MAQUINA:
        humano = "Jugador"

    game = abrir_partida([humano, MAQUINA], seed=semilla)
    nombres = [p.name for p in game.players]
    if humano not in nombres:
        humano = elegir_humano(nombres)
    maquina = next(n for n in nombres if n != humano)

    partida = TwoPlayerMatch(
        nombres,
        inputs={humano: pedir_por_consola, maquina: jugador_al_azar(random.Random(semilla))},
        on_turn=mostrar_turno,
        on_invalid=avisar,
        game=game,
    )
    print()
    print_game(partida.game)
    partida.run()
    print()
    print_game(partida.game)
    print()
    print_final(partida.game)


if __name__ == "__main__":
    main()
