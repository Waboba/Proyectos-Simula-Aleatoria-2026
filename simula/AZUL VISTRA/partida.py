"""Partida de 2 jugadores sobre el motor de azulvitral.py.

Cada jugador es una función que recibe la partida y su Player y devuelve
una Move. Así la misma clase sirve para consola, una interfaz gráfica o
una lista de jugadas fija.
"""

from typing import Callable, Dict, List, Optional, Tuple

from azulvitral import AzulVitralGame, InvalidMoveError, Move, Player, TurnReport

PlayerInput = Callable[[AzulVitralGame, Player], Move]


class TwoPlayerMatch:
    """Orquesta una partida completa entre dos jugadores."""

    def __init__(
        self,
        names: List[str],
        inputs: Optional[Dict[str, PlayerInput]] = None,
        boards: Tuple[str, str] = ("rojo", "amarillo"),
        first_player=None,
        seed: Optional[int] = None,
        on_turn: Optional[Callable[[TurnReport, AzulVitralGame], None]] = None,
        on_invalid: Optional[Callable[[Player, Move, InvalidMoveError], None]] = None,
    ):
        """inputs: nombre del jugador -> función que entrega su jugada.

        Se puede omitir si las jugadas se van a pasar a mano con play_turn().
        on_turn se llama después de cada turno válido. on_invalid se llama
        cuando una jugada no es válida; si no se entrega, el error se lanza.
        """
        self.game = AzulVitralGame(names, boards=boards, first_player=first_player, seed=seed)
        self.inputs = inputs or {}
        self.on_turn = on_turn
        self.on_invalid = on_invalid
        self.history: List[TurnReport] = []

    @property
    def current_player(self) -> Player:
        return self.game.get_current_player()

    @property
    def is_over(self) -> bool:
        return self.game.game_over

    def play_turn(self, move: Move) -> TurnReport:
        """Aplica la jugada del jugador de turno.

        Si no es válida lanza InvalidMoveError y el estado queda igual.
        """
        report = self.game.apply_move(move)
        self.history.append(report)
        if self.on_turn:
            self.on_turn(report, self.game)
        return report

    def ask_and_play(self) -> TurnReport:
        """Le pide la jugada a quien tiene el turno hasta que sea válida."""
        player = self.current_player
        if player.name not in self.inputs:
            raise ValueError(f"No hay función de entrada para {player.name}.")
        while True:
            move = self.inputs[player.name](self.game, player)
            try:
                return self.play_turn(move)
            except InvalidMoveError as err:
                if self.on_invalid is None:
                    raise
                self.on_invalid(player, move, err)

    def run(self) -> Optional[Player]:
        """Juega hasta el final. Devuelve el ganador, o None si hay empate técnico."""
        while not self.is_over:
            self.ask_and_play()
        return self.winner()

    def winner(self) -> Optional[Player]:
        return self.game.winner()
