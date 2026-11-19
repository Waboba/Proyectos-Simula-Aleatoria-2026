"""Funciones para mostrar la partida en consola.

Solo leen el estado, así que se pueden llamar en cualquier momento sin
cambiar nada. Cada print_x tiene su x_text que devuelve el texto, por si
otra interfaz lo quiere usar.
"""

from collections import Counter

from src.azulvitral import (
    CENTER,
    JOKER,
    NUM_ROUNDS,
    NUM_STRIPS,
    STRIP_SIZE,
    WINDOW_POINTS,
    AzulVitralGame,
    Color,
    Player,
    TurnReport,
)

SHORT = {
    Color.NARANJO: "NA",
    Color.AMARILLO: "AM",
    Color.ROSADO: "RS",
    Color.AZUL: "AZ",
    Color.TRANSPARENTE: "TR",
}
ADJECTIVE = {
    Color.NARANJO: ("naranja", "naranjas"),
    Color.AMARILLO: ("amarilla", "amarillas"),
    Color.ROSADO: ("rosada", "rosadas"),
    Color.AZUL: ("azul", "azules"),
    Color.TRANSPARENTE: ("transparente", "transparentes"),
}
ROMAN = ["I", "II", "III", "IV", "V", "VI"]
CELL = 6
LABEL = 14

LEGEND = (
    "[NA] pieza puesta   na espacio libre de ese color   ** comodín libre   "
    "-- tira retirada   V vidriero"
)


def pieces(n: int, color: Color) -> str:
    """'1 pieza naranja', '3 piezas azules'."""
    one, many = ADJECTIVE[color]
    return f"1 pieza {one}" if n == 1 else f"{n} piezas {many}"


def _tiles_text(tiles) -> str:
    if not tiles:
        return "vacía"
    count = Counter(tiles)
    parts = []
    for color in Color:
        if count[color]:
            one, many = ADJECTIVE[color]
            parts.append(f"{count[color]} {one if count[color] == 1 else many}")
    return ", ".join(parts)


def _row(label: str, cells) -> str:
    text = ("  " + label).ljust(LABEL) + "".join(str(c).center(CELL) for c in cells)
    return text.rstrip()


def _verb(n: int, one: str, many: str) -> str:
    return f"{one} {n}" if n == 1 else f"{many} {n}"


def header_text(game: AzulVitralGame) -> str:
    if game.game_over:
        status = "Partida terminada"
    else:
        status = (
            f"Ronda {game.round} de {NUM_ROUNDS}   color de la ronda: "
            f"{game.current_round_color().value}   turno de {game.get_current_player().name}"
        )
    marker = "  ".join(
        f"{ROMAN[i]}:{c.value if c else '-'}" for i, c in enumerate(game.round_marker)
    )
    return (
        f"{status}\n"
        f"Marcador de ronda: {marker}\n"
        f"Bolsa: {len(game.bag)} piezas   Torre de cristal: {len(game.tower)} piezas"
    )


def factories_text(game: AzulVitralGame) -> str:
    lines = ["Fábricas:"]
    for factory in game.factories:
        lines.append(f"  {factory.id}: {_tiles_text(factory.tiles)}")
    return "\n".join(lines)


def center_text(game: AzulVitralGame) -> str:
    if game.center.has_first_player_token:
        token = "con la loseta inicial"
    else:
        holder = next((p.name for p in game.players if p.has_first_player_token), None)
        token = f"la loseta inicial la tiene {holder}" if holder else "sin loseta inicial"
    tiles = _tiles_text(game.center.tiles) if game.center.tiles else "vacío"
    return f"Centro: {tiles} ({token})"


def _space_cell(player: Player, column: int, row: int) -> str:
    strip = player.strips[column]
    if strip.is_removed:
        return "--"
    tile = strip.filled_spaces[row]
    if tile is not None:
        return f"[{SHORT[tile]}]"
    space = strip.pattern()[row]
    return "**" if space == JOKER else SHORT[space].lower()


def player_text(game: AzulVitralGame, player: Player) -> str:
    playing = not game.game_over and game.get_current_player() is player
    token = "sí" if player.has_first_player_token else "no"
    broken = f"casilla {player.broken_position} ({player.broken_penalty()})"

    lines = [
        f"{player.name} (tablero {player.board})" + ("  <- le toca" if playing else ""),
        f"  puntos: {player.score}   cristal roto: {broken}   loseta inicial: {token}",
        _row("columna", range(1, NUM_STRIPS + 1)),
        _row("vidriero", ["V" if i == player.glazier else "" for i in range(NUM_STRIPS)]),
        _row("cara", [
            "--" if s.is_removed else ("anv" if s.current_side == 0 else "rev")
            for s in player.strips
        ]),
    ]
    for row in range(STRIP_SIZE):
        lines.append(_row(f"espacio {row + 1}", [
            _space_cell(player, i, row) for i in range(NUM_STRIPS)
        ]))
    for slot, label in ((0, "ventana sup"), (1, "ventana inf")):
        lines.append(_row(label, [
            SHORT[w[slot]] if w[slot] else "." for w in player.windows
        ]))
    lines.append(_row("valor", WINDOW_POINTS))

    ornaments = []
    for k in range(NUM_STRIPS // 2):
        filled = sum(1 for s in player.windows[2 * k] + player.windows[2 * k + 1] if s)
        ornaments.append(f"{k + 1}: {filled}/4")
    lines.append("  ornamentos: " + "   ".join(ornaments))
    return "\n".join(lines)


def game_text(game: AzulVitralGame) -> str:
    parts = [header_text(game), factories_text(game), center_text(game), LEGEND]
    parts += [player_text(game, p) for p in game.players]
    return "\n\n".join(parts)


def turn_text(report: TurnReport) -> str:
    move = report.move
    if move.option == "B":
        text = f"{report.player} devuelve el vidriero a la tira de más a la izquierda."
    else:
        origin = "del centro" if move.source == CENTER else f"de la fábrica {move.source}"
        text = f"{report.player} toma {pieces(report.taken, move.color)} {origin}"
        if report.took_first_player_token:
            text += " y se lleva la loseta inicial (1 cristal roto)"
        all_broken = "se rompe" if report.taken == 1 else "se rompen todas"
        if move.strip is None:
            text += f". No tiene ninguna tira alcanzable y {all_broken}."
        elif report.placed == report.taken:
            text += f". {'Va' if report.taken == 1 else 'Van'} a la tira {move.strip}."
        elif report.placed == 0:
            text += f". Elige la tira {move.strip}, pero no cabe y {all_broken}."
        else:
            text += (
                f". En la tira {move.strip} {_verb(report.placed, 'cabe', 'caben')} "
                f"y {_verb(report.broken, 'se rompe', 'se rompen')}."
            )
        if report.completed:
            fate = "se retira del tablero" if report.strip_removed else "se voltea"
            text += (
                f" Completa la tira: +{report.color_bonus} por el color de la ronda y "
                f"+{report.window_points} por ventanas. Deja una pieza "
                f"{ADJECTIVE[move.window_color][0]} en la ventana y la tira {fate}."
            )
    if report.overflows:
        times = "una vez" if report.overflows == 1 else f"{report.overflows} veces"
        text += (
            f" Llega a la casilla 10 del cristal roto ({times}): pierde 18 puntos "
            "ahora (sin bajar de 0) y el marcador vuelve a la casilla 1."
        )
    if report.game_over:
        text += " Fin de la partida."
    elif report.round_ended:
        text += " Fin de la ronda."
    return text


def result_text(game: AzulVitralGame) -> str:
    winner = game.winner()
    a, b = game.players
    score = f"{a.name}: {a.score} puntos, {b.name}: {b.score} puntos"
    if winner is None:
        return f"{score}. Empate técnico."
    if a.score == b.score:
        return (
            f"{score}. Empatan en puntos y gana {winner.name} por quedar en una "
            "casilla más baja del cristal roto."
        )
    return f"{score}. Gana {winner.name}."


def final_text(game: AzulVitralGame) -> str:
    lines = ["Puntuación final"]
    for player in game.players:
        detail = game.final_scores[player.name]
        orn = " ".join(f"+{p}" for p in detail.ornament_points)
        lines += [
            f"  {player.name}: {detail.score_before} puntos al terminar la ronda {NUM_ROUNDS}",
            f"    piezas sin usar en tiras: {detail.leftover_tiles} -> +{detail.leftover_points}",
            f"    cristal roto: casilla {player.broken_position} -> {detail.broken_penalty}",
            f"    ornamentos: {orn} = +{sum(detail.ornament_points)}",
            f"    total: {detail.total}",
        ]
    lines.append(result_text(game))
    return "\n".join(lines)


def print_header(game: AzulVitralGame) -> None:
    print(header_text(game))


def print_factories(game: AzulVitralGame) -> None:
    print(factories_text(game))


def print_center(game: AzulVitralGame) -> None:
    print(center_text(game))


def print_player(game: AzulVitralGame, player: Player) -> None:
    print(player_text(game, player))


def print_game(game: AzulVitralGame) -> None:
    print(game_text(game))


def print_turn(report: TurnReport) -> None:
    print(turn_text(report))


def print_final(game: AzulVitralGame) -> None:
    print(final_text(game))
