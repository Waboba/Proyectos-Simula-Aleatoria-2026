"""Partida en consola contra la máquina (IA MaskablePPO o rival aleatorio).

Ejecutar desde la raíz del proyecto:
    python -m scripts.contra_maquina
    python -m scripts.contra_maquina Ana          (con tu nombre)
    python -m scripts.contra_maquina Ana 7        (con nombre y semilla)
    python -m scripts.contra_maquina Ana 7 --azar (forzar rival aleatorio)

Si existe models_checkpoint/ppo_azul_sintra_v1.zip, la máquina usa el modelo
entrenado con Aprendizaje por Refuerzo; en caso contrario, juega al azar.
"""

import random
import sys
from pathlib import Path

# Añadir la raíz del proyecto (Azul-Vitral) al sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.jugar import avisar, describir, mostrar_turno, pedir_por_consola
from src.azulvitral import AzulVitralGame, Move, Player
from src.estado import abrir_partida
from src.impresion import print_final, print_game
from src.partida import TwoPlayerMatch

MAQUINA = "Máquina"
MODEL_PATH = PROJECT_ROOT / "models_checkpoint" / "ppo_azul_sintra_v1.zip"


def jugador_al_azar(azar: random.Random):
    """Entrada de un jugador que elige al azar entre todas sus jugadas válidas."""

    def elegir(game: AzulVitralGame, player: Player) -> Move:
        jugadas = game.legal_moves()
        jugada = azar.choice(jugadas)
        print(
            f"\n{player.name} elige entre {len(jugadas)} jugadas válidas: "
            f"{describir(jugada)}"
        )
        return jugada

    return elegir


def jugador_ia(modelo):
    """Entrada de un jugador controlado por el modelo MaskablePPO entrenado."""
    from src.env_azul import AzulSintraEnv

    env_helper = AzulSintraEnv()

    def elegir(game: AzulVitralGame, player: Player) -> Move:
        env_helper.game = game
        player_idx = game.players.index(player)
        agent_name = f"player_{player_idx}"
        env_helper.agent_selection = agent_name

        obs_dict = env_helper.observe(agent_name)
        action, _ = modelo.predict(
            obs_dict["observation"],
            action_masks=obs_dict["action_mask"].astype(bool),
            deterministic=True,
        )
        jugada = env_helper.decode_action(int(action))
        jugadas = game.legal_moves()
        print(
            f"\n{player.name} (IA) elige entre {len(jugadas)} jugadas válidas: "
            f"{describir(jugada)}"
        )
        return jugada

    return elegir


def _crear_entrada_maquina(semilla, forzar_azar: bool = False):
    """Carga el modelo RL si está disponible; si no, usa el agente aleatorio."""
    if not forzar_azar and MODEL_PATH.exists():
        try:
            from sb3_contrib import MaskablePPO

            modelo = MaskablePPO.load(str(MODEL_PATH))
            print(f"Modelo RL cargado: {MODEL_PATH.name}")
            return jugador_ia(modelo)
        except ImportError:
            print("Aviso: sb3_contrib no está instalado; usando rival aleatorio.")

    if not forzar_azar and not MODEL_PATH.exists():
        print(f"Aviso: No se encontró {MODEL_PATH.name}; usando rival aleatorio.")
    return jugador_al_azar(random.Random(semilla))


def elegir_humano(nombres) -> str:
    while True:
        texto = input(
            f"¿Cuál de los dos eres tú? ({nombres[0]} o {nombres[1]}): "
        ).strip()
        if texto in nombres:
            return texto
        print("  Escribe uno de los dos nombres.")


def main():
    args = [a for a in sys.argv[1:] if a != "--azar"]
    forzar_azar = "--azar" in sys.argv[1:]

    humano = args[0] if len(args) >= 1 else "Jugador"
    semilla = int(args[1]) if len(args) >= 2 else None
    if humano == MAQUINA:
        humano = "Jugador"

    entrada_maquina = _crear_entrada_maquina(semilla, forzar_azar=forzar_azar)

    game = abrir_partida([humano, MAQUINA], seed=semilla)
    nombres = [p.name for p in game.players]
    if humano not in nombres:
        humano = elegir_humano(nombres)
    maquina = next(n for n in nombres if n != humano)

    partida = TwoPlayerMatch(
        nombres,
        inputs={humano: pedir_por_consola, maquina: entrada_maquina},
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
