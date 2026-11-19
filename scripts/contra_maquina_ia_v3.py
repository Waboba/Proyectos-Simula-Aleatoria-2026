"""Partida en consola contra la IA v3 (Behavioral Cloning + MaskablePPO).

Ejecutar desde la raíz del proyecto:
    python -m scripts.contra_maquina_ia_v3
    python -m scripts.contra_maquina_ia_v3 Ana              (con tu nombre)
    python -m scripts.contra_maquina_ia_v3 Ana 7            (con nombre y semilla)
    python -m scripts.contra_maquina_ia_v3 Ana 7 --v2       (enfrentar al modelo v2)
    python -m scripts.contra_maquina_ia_v3 Ana 7 --v1       (enfrentar al modelo v1)
    python -m scripts.contra_maquina_ia_v3 Ana 7 --heuristica (enfrentar a la heurística)
    python -m scripts.contra_maquina_ia_v3 Ana 7 --azar     (enfrentar al rival aleatorio)
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
from src.env_azul_v2 import AzulSintraEnvV2, decode_action, heuristic_move
from src.env_azul_v3 import AzulSintraEnvV3
from src.estado import abrir_partida
from src.impresion import print_final, print_game
from src.partida import TwoPlayerMatch

MAQUINA = "Máquina_v3"
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


def jugador_ia_v3(modelo):
    """Rival controlado por el modelo híbrido BC + MaskablePPO v3 (288 observaciones)."""
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
            f"\n{player.name} (IA v3) elige entre {len(jugadas)} jugadas válidas: "
            f"{describir(jugada)}"
        )
        return jugada

    return elegir


def jugador_ia_v2(modelo):
    """Rival controlado por el modelo MaskablePPO v2 (288 observaciones)."""
    helper = AzulSintraEnvV2()

    def elegir(game: AzulVitralGame, player: Player) -> Move:
        helper.game = game
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
            f"\n{player.name} (IA v2) elige entre {len(jugadas)} jugadas válidas: "
            f"{describir(jugada)}"
        )
        return jugada

    return elegir


def jugador_ia_v1(modelo):
    """Rival controlado por el modelo MaskablePPO v1 (253 observaciones)."""
    import src.env_azul as env_v1_mod

    helper = env_v1_mod.AzulSintraEnv()
    decode_v1 = getattr(
        helper,
        "decode_action",
        getattr(env_v1_mod, "decode_action", None),
    )

    def elegir(game: AzulVitralGame, player: Player) -> Move:
        helper.game = game
        player_idx = next(i for i, p in enumerate(game.players) if p is player)
        ag = f"player_{player_idx}"
        helper.agent_selection = ag

        obs_dict = helper.observe(ag)
        act, _ = modelo.predict(
            obs_dict["observation"],
            action_masks=obs_dict["action_mask"].astype(bool),
            deterministic=True,
        )
        jugada = decode_v1(int(act))
        jugadas = game.legal_moves()
        print(
            f"\n{player.name} (IA v1) elige entre {len(jugadas)} jugadas válidas: "
            f"{describir(jugada)}"
        )
        return jugada

    return elegir


def _configurar_rival(semilla, modo: str):
    """Carga el agente solicitado según los argumentos de consola."""
    from sb3_contrib import MaskablePPO

    if modo == "azar":
        print("Modo seleccionado: Rival Aleatorio")
        return jugador_al_azar(random.Random(semilla)), "Máquina_Azar"

    if modo == "heuristica":
        print("Modo seleccionado: Heurística Táctica")
        return jugador_heuristico(random.Random(semilla)), "Máquina_Heurística"

    if modo == "v1":
        if not MODEL_V1_PATH.exists():
            raise FileNotFoundError(f"No se encontró {MODEL_V1_PATH}")
        modelo = MaskablePPO.load(str(MODEL_V1_PATH), device="cpu")
        print(f"Modelo cargado: {MODEL_V1_PATH.name}")
        return jugador_ia_v1(modelo), "Máquina_v1"

    if modo == "v2":
        if not MODEL_V2_PATH.exists():
            raise FileNotFoundError(f"No se encontró {MODEL_V2_PATH}")
        modelo = MaskablePPO.load(str(MODEL_V2_PATH), device="cpu")
        print(f"Modelo cargado: {MODEL_V2_PATH.name}")
        return jugador_ia_v2(modelo), "Máquina_v2"

    if not MODEL_V3_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró {MODEL_V3_PATH}. Ejecuta primero scripts/train_rl_v3.py."
        )
    modelo = MaskablePPO.load(str(MODEL_V3_PATH), device="cpu")
    print(f"Modelo cargado: {MODEL_V3_PATH.name}")
    return jugador_ia_v3(modelo), MAQUINA


def elegir_humano(nombres) -> str:
    while True:
        texto = input(
            f"¿Cuál de los dos eres tú? ({nombres[0]} o {nombres[1]}): "
        ).strip()
        if texto in nombres:
            return texto
        print("  Escribe uno de los dos nombres.")


def main():
    flags = {"--azar", "--heuristica", "--v1", "--v2", "--v3"}
    args = [a for a in sys.argv[1:] if a not in flags]

    if "--azar" in sys.argv[1:]:
        modo = "azar"
    elif "--heuristica" in sys.argv[1:]:
        modo = "heuristica"
    elif "--v1" in sys.argv[1:]:
        modo = "v1"
    elif "--v2" in sys.argv[1:]:
        modo = "v2"
    else:
        modo = "v3"

    humano = args[0] if len(args) >= 1 else "Jugador"
    semilla = int(args[1]) if len(args) >= 2 else None

    entrada_maquina, nombre_maquina = _configurar_rival(semilla, modo)
    if humano == nombre_maquina:
        humano = "Jugador"

    game = abrir_partida([humano, nombre_maquina], seed=semilla)
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