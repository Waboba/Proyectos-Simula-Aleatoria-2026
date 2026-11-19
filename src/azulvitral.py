"""Motor de reglas de Azul: Vitrales de Sintra para 2 jugadores.

Sigue reglas_azul_sintra.md. Lo que el reglamento no cubría se preguntó
antes de programarlo; esos casos están marcados con "Acordado:".
"""

import json
import random
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


class Color(Enum):
    NARANJO = "Naranjo"
    AMARILLO = "Amarillo"
    ROSADO = "Rosado"
    AZUL = "Azul"
    TRANSPARENTE = "Transparente"


JOKER = "COMODIN"
CENTER = "centro"

# Casillas 1 a 10 del track de cristal roto.
BROKEN_GLASS_PENALTIES = [0, -1, -2, -4, -5, -8, -10, -12, -15, -18]
# Puntos impresos bajo las ventanas 1 a 8.
WINDOW_POINTS = [4, 3, 3, 2, 2, 1, 1, 2]
# Espacios con pieza alrededor de un ornamento -> puntos al final.
ORNAMENT_BONUS = {0: 0, 1: 0, 2: 3, 3: 6, 4: 10}

NUM_STRIPS = 8
STRIP_SIZE = 5
NUM_FACTORIES = 5
TILES_PER_FACTORY = 4
TILES_PER_COLOR = 20
NUM_ROUNDS = 6

STRIPS_FILE = Path(__file__).resolve().parent.parent / "datos" / "tiras_patron.json"

# Un espacio de una tira es un Color o JOKER.
Space = Union[Color, str]


class InvalidMoveError(Exception):
    """Jugada que no respeta las reglas. El mensaje explica el motivo."""


@dataclass
class PatternStrip:
    """Una tira de patrón de doble cara."""

    side_a: List[Space]
    side_b: List[Space]
    current_side: int = 0
    filled_spaces: List[Optional[Color]] = field(
        default_factory=lambda: [None] * STRIP_SIZE
    )
    is_removed: bool = False  # True tras completar la ventana 2 veces
    name: str = ""

    def pattern(self) -> List[Space]:
        """Los 5 espacios de la cara que está boca arriba."""
        return self.side_a if self.current_side == 0 else self.side_b

    def is_complete(self) -> bool:
        return all(t is not None for t in self.filled_spaces)

    def has_jokers(self, side: int) -> bool:
        return JOKER in (self.side_a if side == 0 else self.side_b)

    def free_slots_for(self, color: Color) -> List[int]:
        """Espacios libres donde puede ir una pieza de ese color.

        Acordado: primero van los espacios de su color y después los
        comodines, así el comodín queda libre mientras se pueda.
        """
        own, jokers = [], []
        for i, space in enumerate(self.pattern()):
            if self.filled_spaces[i] is not None:
                continue
            if space == color:
                own.append(i)
            elif space == JOKER:
                jokers.append(i)
        return own + jokers

    def empty_count(self) -> int:
        return self.filled_spaces.count(None)

    def tiles(self) -> List[Color]:
        return [t for t in self.filled_spaces if t is not None]

    def flip(self) -> None:
        self.current_side = 1 - self.current_side
        self.filled_spaces = [None] * STRIP_SIZE


@dataclass
class Factory:
    """Un expositor de fábrica."""

    id: int
    tiles: List[Color] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.tiles


@dataclass
class CenterPool:
    """El centro de la mesa."""

    tiles: List[Color] = field(default_factory=list)
    has_first_player_token: bool = False

    def is_empty(self) -> bool:
        """Sin cristales. La loseta inicial no es una pieza y no cuenta."""
        return not self.tiles


@dataclass
class Player:
    """Tablero, puntos, vidriero y track de cristal roto de un jugador."""

    name: str
    board: str
    strips: List[PatternStrip]
    # Una ventana por columna: [espacio superior, espacio inferior].
    windows: List[List[Optional[Color]]] = field(
        default_factory=lambda: [[None, None] for _ in range(NUM_STRIPS)]
    )
    glazier: int = 0  # columna 0 a 7
    score: int = 0
    broken_position: int = 1  # casilla 1 a 10
    has_first_player_token: bool = False

    def get_leftmost_active_strip_index(self) -> Optional[int]:
        """Índice de la tira más a la izquierda que sigue en el tablero."""
        for i, strip in enumerate(self.strips):
            if not strip.is_removed:
                return i
        return None

    def can_reset_glazier(self) -> bool:
        """Opción B: solo si hay una tira activa a la izquierda del vidriero.

        Acordado: el vidriero vuelve a la tira activa más a la izquierda. Si
        ya está ahí, o en una columna vacía a la izquierda de todas las tiras,
        la opción B no está disponible.
        """
        leftmost = self.get_leftmost_active_strip_index()
        return leftmost is not None and leftmost < self.glazier

    def reset_glazier(self) -> bool:
        if not self.can_reset_glazier():
            return False
        self.glazier = self.get_leftmost_active_strip_index()
        return True

    def reachable_strips(self) -> List[int]:
        """Tiras bajo el vidriero o a su derecha que siguen en el tablero."""
        return [
            i for i in range(self.glazier, NUM_STRIPS)
            if not self.strips[i].is_removed
        ]

    def broken_penalty(self) -> int:
        return BROKEN_GLASS_PENALTIES[self.broken_position - 1]

    def add_points(self, points: int) -> None:
        """Suma (o resta) puntos. El puntaje nunca baja de 0."""
        self.score = max(0, self.score + points)

    def leftover_tiles(self) -> int:
        return sum(len(s.tiles()) for s in self.strips if not s.is_removed)


@dataclass(frozen=True)
class Move:
    """Una jugada.

    Opción A: source es la fábrica (1 a 5) o CENTER, color el color que se
    toma y strip la columna de destino (1 a 8). Si la tira se completa,
    window_color indica qué pieza queda en la ventana. Si no queda ninguna
    tira bajo el vidriero ni a su derecha, strip va en None y todo se rompe.

    Opción B: solo option="B".
    """

    option: str
    source: Union[int, str, None] = None
    color: Optional[Color] = None
    strip: Optional[int] = None
    window_color: Optional[Color] = None

    @classmethod
    def take(cls, source, color, strip, window_color=None) -> "Move":
        return cls("A", source, color, strip, window_color)

    @classmethod
    def reset_glazier(cls) -> "Move":
        return cls("B")


@dataclass
class TurnReport:
    """Lo que pasó en un turno ya aplicado."""

    player: str
    move: Move
    taken: int = 0
    took_first_player_token: bool = False
    placed: int = 0
    broken: int = 0
    completed: bool = False
    color_bonus: int = 0
    window_points: int = 0
    strip_removed: bool = False
    overflows: int = 0
    round_ended: bool = False
    game_over: bool = False


@dataclass
class FinalScore:
    """Detalle de la puntuación final de un jugador."""

    leftover_tiles: int
    leftover_points: int
    broken_penalty: int
    ornament_points: List[int]
    score_before: int  # puntaje al terminar la ronda 6
    total: int


def _read_space(text: str) -> Space:
    return JOKER if text == JOKER else Color[text]


def load_strips(path=STRIPS_FILE) -> Dict[str, List[Tuple[List[Space], List[Space]]]]:
    """Lee tiras_patron.json: tablero -> lista de (anverso, reverso)."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    boards = {}
    for name, strips in data["tableros"].items():
        boards[name] = [
            ([_read_space(s) for s in t["anverso"]],
             [_read_space(s) for s in t["reverso"]])
            for t in strips
        ]
    return boards


def generate_default_strips_pool(board: str, path=STRIPS_FILE) -> List[PatternStrip]:
    """Las 8 tiras de un tablero, en el orden del archivo y por el anverso."""
    boards = load_strips(path)
    if board not in boards:
        raise ValueError(f"No existe el tablero {board!r}. Hay: {', '.join(boards)}")
    return [
        PatternStrip(side_a=a, side_b=b, name=f"{board}-{n}")
        for n, (a, b) in enumerate(boards[board], start=1)
    ]


class AzulVitralGame:
    """Estado de la partida y aplicación de las reglas."""

    def __init__(
        self,
        player_names: List[str],
        boards: Tuple[str, str] = ("rojo", "amarillo"),
        first_player: Union[int, str, None] = None,
        seed: Optional[int] = None,
        strips_file=STRIPS_FILE,
    ):
        """Prepara una partida de 2 jugadores.

        first_player puede ser 0, 1, el nombre del jugador o None (al azar).
        seed fija el azar para poder repetir una partida.
        """
        if len(player_names) != 2:
            raise ValueError("Este motor es solo para 2 jugadores.")
        if player_names[0] == player_names[1]:
            raise ValueError("Los jugadores necesitan nombres distintos.")
        if len(boards) != 2 or boards[0] == boards[1]:
            raise ValueError("Cada jugador necesita un tablero distinto.")

        self.rng = random.Random(seed)
        self.strips_file = strips_file
        self.players = [
            Player(name=name, board=board, strips=self._create_random_board(board))
            for name, board in zip(player_names, boards)
        ]
        self.factories = [Factory(id=i + 1) for i in range(NUM_FACTORIES)]
        self.center = CenterPool()
        self.bag: List[Color] = []
        self.tower: List[Color] = []
        self.round_marker: List[Optional[Color]] = []
        self.round = 0
        self.current = 0
        self.round_starter = 0
        self.game_over = False
        self.final_scores: Dict[str, FinalScore] = {}

        if first_player is None:
            first = self.rng.randrange(2)
        elif isinstance(first_player, str):
            if first_player not in player_names:
                raise ValueError(f"No hay ningún jugador llamado {first_player!r}.")
            first = player_names.index(first_player)
        elif first_player in (0, 1):
            first = first_player
        else:
            raise ValueError("first_player tiene que ser 0, 1, un nombre o None.")
        self.players[first].has_first_player_token = True
        self.round_starter = first

        # Marcador de ronda: un color de cada uno al azar en II a VI y una
        # pieza más, sacada de las 95 que quedan, en la posición I.
        pool = [c for c in Color for _ in range(TILES_PER_COLOR)]
        rounds_2_to_6 = list(Color)
        self.rng.shuffle(rounds_2_to_6)
        for c in rounds_2_to_6:
            pool.remove(c)
        first_round = pool.pop(self.rng.randrange(len(pool)))
        self.round_marker = [first_round] + rounds_2_to_6
        self.bag = pool

        self._start_new_round()

    # --- preparación y rondas ---

    def _create_random_board(self, board: str) -> List[PatternStrip]:
        """Baraja las 8 tiras y les da una cara al azar.

        Si la tira comodín queda con los comodines hacia arriba, se voltea.
        """
        strips = generate_default_strips_pool(board, self.strips_file)
        self.rng.shuffle(strips)
        for strip in strips:
            strip.current_side = self.rng.randrange(2)
            if strip.has_jokers(strip.current_side):
                strip.current_side = 1 - strip.current_side
        return strips

    def _draw_tiles_from_bag(self, count: int) -> List[Color]:
        """Saca piezas al azar. Si la bolsa se vacía, se rellena con la torre.

        Si ni así alcanza, devuelve las que haya.
        """
        drawn = []
        while len(drawn) < count:
            if not self.bag:
                if not self.tower:
                    break
                self.bag, self.tower = self.tower, []
            drawn.append(self.bag.pop(self.rng.randrange(len(self.bag))))
        return drawn

    def _start_new_round(self) -> None:
        """Llena las fábricas y deja la loseta inicial en el centro.

        Nunca quedan todas vacías: antes de rellenar, la pieza del marcador
        de la ronda que terminó ya volvió a la torre.
        """
        self.round += 1
        for factory in self.factories:
            factory.tiles = self._draw_tiles_from_bag(TILES_PER_FACTORY)

        # Empieza quien tomó la loseta inicial en la ronda anterior.
        holder = next(
            (i for i, p in enumerate(self.players) if p.has_first_player_token),
            None,
        )
        if holder is not None:
            self.round_starter = holder
        for p in self.players:
            p.has_first_player_token = False
        self.center.has_first_player_token = True
        self.current = self.round_starter

    def _end_round(self) -> None:
        self.tower.append(self.round_marker[self.round - 1])
        self.round_marker[self.round - 1] = None
        if self.round == NUM_ROUNDS:
            self.end_game_scoring()
        else:
            self._start_new_round()

    def is_round_over(self) -> bool:
        """True si no quedan piezas en las fábricas ni en el centro."""
        return self.center.is_empty() and all(f.is_empty() for f in self.factories)

    def current_round_color(self) -> Optional[Color]:
        if not 1 <= self.round <= NUM_ROUNDS:
            return None
        return self.round_marker[self.round - 1]

    def get_current_player(self) -> Player:
        return self.players[self.current]

    def _advance_turn(self) -> None:
        if self.is_round_over():
            self._end_round()
        else:
            self.current = 1 - self.current

    # --- validación ---

    def get_valid_placement_strips(self, player: Player, tiles: List[Color]) -> List[int]:
        """Tiras alcanzables (bajo el vidriero o a su derecha) donde cabe
        al menos una de las piezas."""
        if not tiles:
            return []
        return [
            i for i in player.reachable_strips()
            if player.strips[i].free_slots_for(tiles[0])
        ]

    def validate_move(self, move: Move) -> Optional[str]:
        """None si la jugada es válida; si no, el motivo."""
        try:
            self._check_move(move)
        except InvalidMoveError as err:
            return str(err)
        return None

    def _check_move(self, move: Move) -> None:
        if self.game_over:
            raise InvalidMoveError("La partida ya terminó.")
        if not isinstance(move, Move):
            raise InvalidMoveError("La jugada tiene que ser un objeto Move.")
        player = self.get_current_player()

        if move.option == "B":
            if any(x is not None for x in (move.source, move.color, move.strip, move.window_color)):
                raise InvalidMoveError("La opción B no lleva origen, color, tira ni pieza para la ventana.")
            if player.get_leftmost_active_strip_index() is None:
                raise InvalidMoveError("No quedan tiras en el tablero: no hay adónde mover el vidriero.")
            if not player.can_reset_glazier():
                raise InvalidMoveError(
                    "El vidriero ya está en la tira más a la izquierda; hay que elegir la opción A."
                )
            return

        if move.option != "A":
            raise InvalidMoveError(f"Opción desconocida: {move.option!r}. Tiene que ser 'A' o 'B'.")
        if move.color is None:
            raise InvalidMoveError("Falta indicar el color de las piezas que se toman.")
        if not isinstance(move.color, Color):
            raise InvalidMoveError(f"Color inválido: {move.color!r}. Usa Color.NARANJO, Color.AZUL, etc.")
        if move.window_color is not None and not isinstance(move.window_color, Color):
            raise InvalidMoveError(f"Pieza para la ventana inválida: {move.window_color!r}.")
        color = move.color

        if move.source == CENTER:
            if color not in self.center.tiles:
                raise InvalidMoveError(f"En el centro no hay piezas {color.value}.")
            count = self.center.tiles.count(color)
        elif isinstance(move.source, int) and 1 <= move.source <= NUM_FACTORIES:
            factory = self.factories[move.source - 1]
            if factory.is_empty():
                raise InvalidMoveError(f"La fábrica {move.source} está vacía.")
            if color not in factory.tiles:
                raise InvalidMoveError(f"La fábrica {move.source} no tiene piezas {color.value}.")
            count = factory.tiles.count(color)
        else:
            raise InvalidMoveError(
                f"Origen inválido: {move.source!r}. Tiene que ser una fábrica del 1 al 5 o el centro."
            )

        reachable = player.reachable_strips()
        if not reachable:
            # Acordado: sin tiras alcanzables igual se puede tomar; todo se rompe.
            if move.strip is not None or move.window_color is not None:
                raise InvalidMoveError(
                    "No queda ninguna tira bajo el vidriero ni a su derecha: la jugada va "
                    f"sin tira y las {count} piezas se rompen."
                )
            return

        if move.strip is None:
            raise InvalidMoveError("Falta elegir la tira donde van las piezas.")
        if not isinstance(move.strip, int) or not 1 <= move.strip <= NUM_STRIPS:
            raise InvalidMoveError(f"La tira {move.strip!r} no existe; las columnas van de 1 a 8.")
        index = move.strip - 1
        strip = player.strips[index]
        if strip.is_removed:
            raise InvalidMoveError(f"La tira de la columna {move.strip} ya se retiró del tablero.")
        if index < player.glazier:
            raise InvalidMoveError(
                f"La tira {move.strip} está a la izquierda del vidriero "
                f"(columna {player.glazier + 1}); solo se puede usar esa columna o una a su derecha."
            )

        free = strip.free_slots_for(color)
        if not free:
            options = self.get_valid_placement_strips(player, [color])
            if options:
                columns = ", ".join(str(i + 1) for i in options)
                raise InvalidMoveError(
                    f"En la tira {move.strip} no cabe ninguna pieza {color.value}, pero sí "
                    f"en la(s) tira(s) {columns}; hay que ponerlas en una de esas."
                )

        placed = min(count, len(free))
        completes = placed > 0 and placed == strip.empty_count()
        if completes:
            pieces = strip.tiles() + [color] * placed
            choices = ", ".join(sorted({p.value for p in pieces}))
            if move.window_color is None:
                raise InvalidMoveError(
                    f"La tira {move.strip} se completa con esta jugada: falta elegir qué "
                    f"pieza va a la ventana ({choices})."
                )
            if move.window_color not in pieces:
                raise InvalidMoveError(
                    f"No hay ninguna pieza {getattr(move.window_color, 'value', move.window_color)} "
                    f"en la tira completa; se puede elegir entre: {choices}."
                )
        elif move.window_color is not None:
            raise InvalidMoveError(
                f"La tira {move.strip} no se completa con esta jugada, así que no se elige pieza para la ventana."
            )

    def legal_moves(self) -> List[Move]:
        """Todas las jugadas validas del jugador de turno."""
        if self.game_over:
            return []
        candidates = [Move.reset_glazier()]
        sources = [(f.id, f.tiles) for f in self.factories] + [(CENTER, self.center.tiles)]
        for source, tiles in sources:
            for color in Color:
                if color not in tiles:
                    continue
                for strip in [None] + list(range(1, NUM_STRIPS + 1)):
                    for window in [None] + list(Color):
                        candidates.append(Move.take(source, color, strip, window))
        return [m for m in candidates if self.validate_move(m) is None]

    # --- aplicar jugadas ---

    def apply_move(self, move: Move) -> TurnReport:
        """Aplica la jugada del jugador de turno.

        Si no es válida lanza InvalidMoveError y no cambia nada.
        """
        self._check_move(move)
        player = self.get_current_player()
        report = TurnReport(player=player.name, move=move)
        round_before = self.round

        if move.option == "B":
            self._execute_reset_glazier_turn(player)
        else:
            if move.source == CENTER:
                report.took_first_player_token = self.center.has_first_player_token
                tiles, overflows = self._take_from_center(player, move.color)
                report.overflows += overflows
            else:
                tiles = self._take_from_factory(move.source - 1, move.color)
            report.taken = len(tiles)

            if move.strip is None:
                self.tower.extend(tiles)
                report.broken = len(tiles)
                report.overflows += self._add_broken_glass(player, len(tiles))
            else:
                self._place_tiles_on_strip(player, move.strip - 1, tiles, move.window_color, report)

        self._advance_turn()
        report.round_ended = self.round != round_before or self.game_over
        report.game_over = self.game_over
        return report

    def _take_from_factory(self, factory_index: int, color: Color) -> List[Color]:
        """Saca todas las piezas de un color de una fábrica; el resto va al centro."""
        factory = self.factories[factory_index]
        taken = [t for t in factory.tiles if t == color]
        self.center.tiles.extend(t for t in factory.tiles if t != color)
        factory.tiles = []
        return taken

    def _take_from_center(self, player: Player, color: Color) -> Tuple[List[Color], int]:
        """Saca todas las piezas de un color del centro.

        El primero en tomar del centro en la ronda se lleva la loseta inicial
        y avanza 1 en el track de cristal roto. Devuelve las piezas y cuántos
        desbordes del track hubo.
        """
        taken = [t for t in self.center.tiles if t == color]
        self.center.tiles = [t for t in self.center.tiles if t != color]
        overflows = 0
        if self.center.has_first_player_token:
            self.center.has_first_player_token = False
            player.has_first_player_token = True
            overflows = self._add_broken_glass(player, 1)
        return taken, overflows

    def _place_tiles_on_strip(
        self,
        player: Player,
        target_strip_index: int,
        tiles: List[Color],
        window_color: Optional[Color],
        report: TurnReport,
    ) -> None:
        """Pone las piezas en la tira; las que no caben se rompen."""
        strip = player.strips[target_strip_index]
        if target_strip_index > player.glazier:
            player.glazier = target_strip_index

        free = strip.free_slots_for(tiles[0]) if tiles else []
        placed = min(len(tiles), len(free))
        for slot in free[:placed]:
            strip.filled_spaces[slot] = tiles[0]

        broken = tiles[placed:]
        self.tower.extend(broken)
        report.placed = placed
        report.broken = len(broken)
        report.overflows += self._add_broken_glass(player, len(broken))

        if placed and strip.is_complete():
            self._process_completed_strip(player, target_strip_index, window_color, report)

    def _process_completed_strip(
        self,
        player: Player,
        strip_index: int,
        window_color: Color,
        report: TurnReport,
    ) -> None:
        """Paso 3: bonificación de color, pieza a la ventana y puntos de ventanas."""
        strip = player.strips[strip_index]
        round_color = self.current_round_color()
        bonus = sum(1 for t in strip.filled_spaces if t == round_color)

        rest = strip.tiles()
        rest.remove(window_color)
        self.tower.extend(rest)

        window = player.windows[strip_index]
        if window[0] is None:
            window[0] = window_color
            strip.flip()
        else:
            window[1] = window_color
            strip.filled_spaces = [None] * STRIP_SIZE
            strip.is_removed = True

        # La ventana recién completada más toda ventana a su derecha con
        # al menos una pieza, aunque ya se haya contado antes.
        points = WINDOW_POINTS[strip_index] + sum(
            WINDOW_POINTS[j]
            for j in range(strip_index + 1, NUM_STRIPS)
            if any(player.windows[j])
        )
        player.add_points(bonus + points)

        report.completed = True
        report.color_bonus = bonus
        report.window_points = points
        report.strip_removed = strip.is_removed

    def _execute_reset_glazier_turn(self, player: Player) -> bool:
        """Opción B: el vidriero vuelve a la tira activa más a la izquierda."""
        return player.reset_glazier()

    def _add_broken_glass(self, player: Player, count: int) -> int:
        """Avanza el track una casilla por pieza rota.

        Al llegar a la casilla 10 se pierden 18 puntos en ese mismo turno (sin
        bajar de 0) y el marcador vuelve a la casilla 1. Acordado: si quedaban
        piezas por contar, siguen avanzando desde ahí. Devuelve cuántas veces
        se llegó a la 10.
        """
        overflows = 0
        for _ in range(count):
            player.broken_position += 1
            if player.broken_position == len(BROKEN_GLASS_PENALTIES):
                player.add_points(BROKEN_GLASS_PENALTIES[-1])
                player.broken_position = 1
                overflows += 1
        return overflows

    # --- final ---

    def end_game_scoring(self) -> List[Tuple[Player, int]]:
        """Puntuación final tras la ronda 6, en el orden del reglamento.

        1. +1 por cada 3 piezas que quedaron en tiras incompletas.
        2. Se resta la casilla del track de cristal roto.
        3. Ornamentos: 4 espacios con pieza +10, 3 +6, 2 +3, 1 o 0 nada.

        El puntaje no baja de 0 en ningún paso.
        """
        if self.game_over:
            raise RuntimeError("La puntuación final ya se hizo.")
        for player in self.players:
            leftovers = player.leftover_tiles()
            leftover_points = leftovers // 3
            penalty = player.broken_penalty()
            ornaments = []
            for k in range(NUM_STRIPS // 2):
                spaces = player.windows[2 * k] + player.windows[2 * k + 1]
                ornaments.append(ORNAMENT_BONUS[sum(1 for s in spaces if s is not None)])
            before = player.score
            player.add_points(leftover_points)
            player.add_points(penalty)
            player.add_points(sum(ornaments))
            self.final_scores[player.name] = FinalScore(
                leftover_tiles=leftovers,
                leftover_points=leftover_points,
                broken_penalty=penalty,
                ornament_points=ornaments,
                score_before=before,
                total=player.score,
            )
        self.game_over = True
        return self.ranking()

    def ranking(self) -> List[Tuple[Player, int]]:
        """Jugadores de mejor a peor: más puntos y, si empatan, el que quedó en
        una casilla más baja del track de cristal roto."""
        ordered = sorted(self.players, key=lambda p: (-p.score, p.broken_position))
        return [(p, p.score) for p in ordered]

    def winner(self) -> Optional[Player]:
        """El ganador, o None si hay empate técnico. Solo con la partida terminada."""
        if not self.game_over:
            raise RuntimeError("La partida todavía no termina.")
        a, b = self.players
        if a.score != b.score:
            return a if a.score > b.score else b
        # Desempate: solo la casilla del track de cristal roto, sin contar
        # cuántas veces se llegó a la casilla 10.
        if a.broken_position != b.broken_position:
            return a if a.broken_position < b.broken_position else b
        return None
