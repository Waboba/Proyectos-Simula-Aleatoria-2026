"""Cargar una partida que ya está en curso.

El estado se puede escribir respondiendo preguntas (preguntar_estado) o
leer de un archivo JSON (cargar_estado). partida_desde_estado lo revisa y
arma la partida; estado_de_partida hace lo contrario, para guardarla.

Cómo se escribe una tira (la cara que se ve, de arriba hacia abajo):

    tr tr AM AM am     minúscula: espacio libre de ese color
                       MAYÚSCULA: espacio con su pieza puesta
    az ** tr *na na    **: comodín libre    *na: comodín con una pieza naranja
    -                  la tira ya se retiró del tablero
"""

import json
import random
from collections import Counter

from src.azulvitral import (
    JOKER,
    NUM_FACTORIES,
    NUM_ROUNDS,
    NUM_STRIPS,
    STRIP_SIZE,
    STRIPS_FILE,
    TILES_PER_COLOR,
    TILES_PER_FACTORY,
    AzulVitralGame,
    CenterPool,
    Color,
    Factory,
    PatternStrip,
    Player,
    load_strips,
)

CORTOS = {
    "na": Color.NARANJO,
    "am": Color.AMARILLO,
    "rs": Color.ROSADO,
    "az": Color.AZUL,
    "tr": Color.TRANSPARENTE,
}
BREVE = {c: k for k, c in CORTOS.items()}


def leer_color(texto: str) -> Color:
    """Un color por su nombre o sus dos primeras letras (az, azul, AZUL...)."""
    texto = texto.lower()
    if texto in CORTOS:
        return CORTOS[texto]
    for color in Color:
        if len(texto) >= 2 and color.name.lower().startswith(texto):
            return color
    raise ValueError(f"No conozco el color {texto!r}.")


def leer_colores(texto: str):
    """Lista de colores separados por espacio. '-' o vacío es ninguno."""
    texto = texto.strip()
    if texto in ("", "-"):
        return []
    return [leer_color(t) for t in texto.split()]


def leer_tira(texto: str):
    """Devuelve (patrón, piezas) de una tira escrita como arriba, o None si se retiró."""
    partes = texto.split()
    if partes == ["-"]:
        return None
    if len(partes) != STRIP_SIZE:
        raise ValueError(f"Una tira tiene {STRIP_SIZE} espacios y escribiste {len(partes)}.")
    patron, piezas = [], []
    for parte in partes:
        if parte.startswith("*"):
            patron.append(JOKER)
            piezas.append(None if parte == "**" else leer_color(parte[1:]))
        else:
            color = leer_color(parte)
            patron.append(color)
            piezas.append(color if parte.isupper() else None)
    return patron, piezas


def escribir_tira(strip: PatternStrip) -> str:
    if strip.is_removed:
        return "-"
    partes = []
    for space, pieza in zip(strip.pattern(), strip.filled_spaces):
        if space == JOKER:
            partes.append("**" if pieza is None else "*" + BREVE[pieza])
        else:
            partes.append(BREVE[space].upper() if pieza else BREVE[space])
    return " ".join(partes)


def leer_ventanas(texto: str):
    """'1:am 3:az/rs' -> ventana 1 con amarilla arriba, ventana 3 con azul arriba y rosada abajo."""
    ventanas = [[None, None] for _ in range(NUM_STRIPS)]
    for parte in texto.split():
        if ":" not in parte:
            raise ValueError(f"{parte!r} no tiene la forma columna:color, por ejemplo 3:az o 3:az/rs.")
        columna, piezas = parte.split(":", 1)
        if not columna.isdigit() or not 1 <= int(columna) <= NUM_STRIPS:
            raise ValueError(f"La columna {columna!r} tiene que ir de 1 a 8.")
        colores = [leer_color(c) for c in piezas.split("/")]
        if len(colores) > 2:
            raise ValueError(f"La ventana {columna} tiene solo dos espacios.")
        ventanas[int(columna) - 1] = colores + [None] * (2 - len(colores))
    return ventanas


def escribir_ventanas(windows) -> str:
    partes = []
    for i, (arriba, abajo) in enumerate(windows):
        if arriba:
            partes.append(f"{i + 1}:{BREVE[arriba]}" + (f"/{BREVE[abajo]}" if abajo else ""))
    return " ".join(partes)


def _armar_tiras(tablero: str, textos, nombre: str, strips_file=STRIPS_FILE):
    """Reconoce cada tira por la cara que se ve y le pone sus piezas."""
    if len(textos) != NUM_STRIPS:
        raise ValueError(f"{nombre}: tienen que ser {NUM_STRIPS} tiras y hay {len(textos)}.")
    pares = load_strips(strips_file)[tablero]
    usadas = set()
    tiras = [None] * NUM_STRIPS
    for col, texto in enumerate(textos, start=1):
        leida = leer_tira(texto)
        if leida is None:
            continue
        patron, piezas = leida
        encontrada = None
        for n, (anverso, reverso) in enumerate(pares):
            for lado, cara in ((0, anverso), (1, reverso)):
                if cara == patron:
                    encontrada = (n, lado)
        if encontrada is None:
            raise ValueError(f"{nombre}, columna {col}: esa cara no es de ninguna tira del tablero {tablero}.")
        n, lado = encontrada
        if n in usadas:
            raise ValueError(f"{nombre}, columna {col}: esa tira ya está en otra columna.")
        usadas.add(n)
        if all(p is not None for p in piezas):
            raise ValueError(
                f"{nombre}, columna {col}: la tira está completa; al completarse se vacía, "
                "así que no puede quedar así."
            )
        anverso, reverso = pares[n]
        tiras[col - 1] = PatternStrip(side_a=anverso, side_b=reverso, current_side=lado,
                                      filled_spaces=list(piezas), name=f"{tablero}-{n + 1}")
    # Las retiradas se completan con las tiras que sobraron; da lo mismo cuál.
    sobrantes = [n for n in range(NUM_STRIPS) if n not in usadas]
    for col in range(NUM_STRIPS):
        if tiras[col] is None:
            n = sobrantes.pop(0)
            anverso, reverso = pares[n]
            tiras[col] = PatternStrip(side_a=anverso, side_b=reverso, is_removed=True,
                                      name=f"{tablero}-{n + 1}")
    return tiras


def partida_desde_estado(estado: dict, seed=None, strips_file=STRIPS_FILE) -> AzulVitralGame:
    """Arma una partida a partir del estado. Lanza ValueError con el motivo si algo no cuadra.

    Las piezas que no están a la vista (ni en la mesa, ni en los tableros, ni
    en la torre, ni en el marcador de ronda) se dan por estar en la bolsa.
    """
    ronda = estado["ronda"]
    if not isinstance(ronda, int) or not 1 <= ronda <= NUM_ROUNDS:
        raise ValueError("La ronda tiene que ir de 1 a 6.")
    colores_ronda = [leer_color(c) for c in estado["colores_ronda"]]
    if len(colores_ronda) != NUM_ROUNDS - ronda + 1:
        raise ValueError(
            f"En la ronda {ronda} quedan {NUM_ROUNDS - ronda + 1} colores en el marcador "
            f"(de la ronda {ronda} a la 6) y hay {len(colores_ronda)}."
        )

    datos = estado["jugadores"]
    if len(datos) != 2:
        raise ValueError("Tienen que ser 2 jugadores.")
    nombres = [d["nombre"] for d in datos]
    if nombres[0] == nombres[1]:
        raise ValueError("Los jugadores necesitan nombres distintos.")
    tableros = [d["tablero"] for d in datos]
    if tableros[0] == tableros[1]:
        raise ValueError("Cada jugador necesita un tablero distinto.")
    conocidos = load_strips(strips_file)

    players = []
    for d in datos:
        if d["tablero"] not in conocidos:
            raise ValueError(f"No existe el tablero {d['tablero']!r}. Hay: {', '.join(conocidos)}.")
        if not isinstance(d["puntos"], int) or d["puntos"] < 0:
            raise ValueError(f"{d['nombre']}: los puntos tienen que ser un entero de 0 para arriba.")
        if not isinstance(d["cristal_roto"], int) or not 1 <= d["cristal_roto"] <= 9:
            raise ValueError(f"{d['nombre']}: la casilla del cristal roto va de 1 a 9 (en la 10 vuelve a la 1).")
        if not isinstance(d["vidriero"], int) or not 1 <= d["vidriero"] <= NUM_STRIPS:
            raise ValueError(f"{d['nombre']}: la columna del vidriero va de 1 a 8.")
        tiras = _armar_tiras(d["tablero"], d["tiras"], d["nombre"], strips_file)
        ventanas = [[leer_color(c) if c else None for c in v] for v in d["ventanas"]]
        for col, (arriba, abajo) in enumerate(ventanas, start=1):
            if abajo and not arriba:
                raise ValueError(f"{d['nombre']}, ventana {col}: no puede haber pieza abajo sin una arriba.")
            if bool(abajo) != tiras[col - 1].is_removed:
                raise ValueError(
                    f"{d['nombre']}, columna {col}: la tira se retira justo cuando su ventana "
                    "recibe la segunda pieza; las dos cosas tienen que ir juntas."
                )
        players.append(Player(name=d["nombre"], board=d["tablero"], strips=tiras,
                              windows=ventanas, glazier=d["vidriero"] - 1,
                              score=d["puntos"], broken_position=d["cristal_roto"]))

    fabricas = [[leer_color(c) for c in f] for f in estado["fabricas"]]
    if len(fabricas) != NUM_FACTORIES:
        raise ValueError(f"Tienen que ser {NUM_FACTORIES} fábricas.")
    for i, f in enumerate(fabricas, start=1):
        if len(f) > TILES_PER_FACTORY:
            raise ValueError(f"La fábrica {i} tiene {len(f)} piezas; caben {TILES_PER_FACTORY}.")
    centro = [leer_color(c) for c in estado["centro"]]
    torre = [leer_color(c) for c in estado.get("torre", [])]
    if not centro and not any(fabricas):
        raise ValueError("No hay piezas en las fábricas ni en el centro: esa ronda ya terminó.")

    loseta, turno = estado["loseta"], estado["turno"]
    if loseta != "centro" and loseta not in nombres:
        raise ValueError("La loseta inicial tiene que estar en el centro o la tiene uno de los jugadores.")
    if turno not in nombres:
        raise ValueError(f"El turno tiene que ser de {nombres[0]} o de {nombres[1]}.")

    # Lo que se ve no puede pasar de 20 piezas por color; el resto está en la bolsa.
    vistas = Counter(colores_ronda) + Counter(centro) + Counter(torre)
    for f in fabricas:
        vistas += Counter(f)
    for p in players:
        vistas += Counter(t for s in p.strips for t in s.tiles())
        vistas += Counter(c for w in p.windows for c in w if c)
    bolsa = []
    for color in Color:
        faltan = TILES_PER_COLOR - vistas[color]
        if faltan < 0:
            raise ValueError(f"Hay {vistas[color]} piezas {color.value} a la vista y en el juego hay solo {TILES_PER_COLOR}.")
        bolsa += [color] * faltan

    game = AzulVitralGame.__new__(AzulVitralGame)
    game.rng = random.Random(seed)
    game.strips_file = strips_file
    game.players = players
    game.factories = [Factory(id=i + 1, tiles=f) for i, f in enumerate(fabricas)]
    game.center = CenterPool(tiles=centro, has_first_player_token=(loseta == "centro"))
    game.bag = bolsa
    game.tower = torre
    game.round_marker = [None] * (ronda - 1) + colores_ronda
    game.round = ronda
    game.current = nombres.index(turno)
    game.round_starter = game.current
    game.game_over = False
    game.final_scores = {}
    for p in players:
        p.has_first_player_token = (p.name == loseta)
    return game


def estado_de_partida(game: AzulVitralGame) -> dict:
    """El estado de una partida en el mismo formato que lee partida_desde_estado."""
    def nombre(c):
        return c.name

    holder = next((p.name for p in game.players if p.has_first_player_token), "centro")
    return {
        "ronda": game.round,
        "colores_ronda": [nombre(c) for c in game.round_marker[game.round - 1:]],
        "turno": game.get_current_player().name,
        "loseta": "centro" if game.center.has_first_player_token else holder,
        "fabricas": [[nombre(c) for c in f.tiles] for f in game.factories],
        "centro": [nombre(c) for c in game.center.tiles],
        "torre": [nombre(c) for c in game.tower],
        "jugadores": [
            {
                "nombre": p.name,
                "tablero": p.board,
                "puntos": p.score,
                "cristal_roto": p.broken_position,
                "vidriero": p.glazier + 1,
                "tiras": [escribir_tira(s) for s in p.strips],
                "ventanas": [[nombre(c) if c else None for c in w] for w in p.windows],
            }
            for p in game.players
        ],
    }


def guardar_estado(estado: dict, ruta: str) -> None:
    """Guarda el estado con cada lista en una línea, para que se pueda corregir a mano."""
    def linea(valor):
        return json.dumps(valor, ensure_ascii=False)

    filas = ["{"]
    for clave, valor in estado.items():
        if clave != "jugadores":
            filas.append(f"  {linea(clave)}: {linea(valor)},")
    filas.append('  "jugadores": [')
    for i, jugador in enumerate(estado["jugadores"]):
        filas.append("    {")
        claves = list(jugador)
        for j, clave in enumerate(claves):
            coma = "," if j < len(claves) - 1 else ""
            if clave == "tiras":
                filas.append('      "tiras": [')
                filas += [f"        {linea(t)}," for t in jugador["tiras"]]
                filas[-1] = filas[-1].rstrip(",")
                filas.append(f"      ]{coma}")
            else:
                filas.append(f"      {linea(clave)}: {linea(jugador[clave])}{coma}")
        filas.append("    }" + ("," if i < len(estado["jugadores"]) - 1 else ""))
    filas += ["  ]", "}"]
    with open(ruta, "w", encoding="utf-8") as fh:
        fh.write("\n".join(filas) + "\n")


def cargar_estado(ruta: str) -> dict:
    with open(ruta, encoding="utf-8") as fh:
        return json.load(fh)


# --- preguntas por consola ---

def _preguntar(pregunta: str, convertir, entrada=input, salida=print):
    """Repite la pregunta hasta que la respuesta se pueda convertir."""
    while True:
        texto = entrada(pregunta).strip()
        try:
            return convertir(texto)
        except ValueError as err:
            salida(f"  {err}")


def _una_de(opciones):
    def convertir(texto):
        if texto not in opciones:
            raise ValueError(f"Escribe una de estas opciones: {', '.join(opciones)}.")
        return texto
    return convertir


def _entero(desde: int, hasta: int):
    def convertir(texto):
        if not texto.lstrip("-").isdigit() or not desde <= int(texto) <= hasta:
            raise ValueError(f"Tiene que ser un número de {desde} a {hasta}.")
        return int(texto)
    return convertir


def preguntar_estado(entrada=input, salida=print) -> dict:
    """Pregunta el estado de la partida en curso, dato por dato."""
    p = lambda pregunta, convertir: _preguntar(pregunta, convertir, entrada, salida)
    salida("Estado de la partida en curso. Los colores se escriben na, am, rs, az, tr.\n")

    ronda = p("Ronda actual (1 a 6): ", _entero(1, NUM_ROUNDS))
    cuantos = NUM_ROUNDS - ronda + 1

    def colores_ronda(texto):
        colores = leer_colores(texto)
        if len(colores) != cuantos:
            raise ValueError(f"Tienen que ser {cuantos} colores, de la ronda {ronda} a la 6.")
        return [c.name for c in colores]

    marcador = p(f"Colores del marcador de ronda, de la ronda {ronda} a la 6: ", colores_ronda)

    jugadores = []
    tableros = list(load_strips())
    for i in (1, 2):
        salida(f"\nJugador {i}")
        usados = [j["nombre"] for j in jugadores]

        def nombre_nuevo(texto):
            if not texto or texto in usados:
                raise ValueError("Escribe un nombre, distinto al del otro jugador.")
            return texto

        nombre = p("  Nombre: ", nombre_nuevo)
        libres = [t for t in tableros if t not in [j["tablero"] for j in jugadores]]
        tab = p(f"  Tablero ({', '.join(libres)}): ", _una_de(libres))
        puntos = p("  Puntos: ", _entero(0, 999))
        casilla = p("  Casilla del cristal roto (1 a 9): ", _entero(1, 9))
        vidriero = p("  Columna del vidriero (1 a 8): ", _entero(1, NUM_STRIPS))
        salida("  Tiras, la cara que se ve de arriba hacia abajo. Minúscula: espacio libre;")
        salida("  MAYÚSCULA: con pieza; **: comodín libre; *na: comodín con pieza; -: retirada.")
        caras = load_strips()[tab]
        textos = []
        for col in range(1, NUM_STRIPS + 1):
            def tira(texto, col=col):
                leida = leer_tira(texto)
                if leida is not None and not any(leida[0] in (a, r) for a, r in caras):
                    raise ValueError(f"Esa cara no es de ninguna tira del tablero {tab}.")
                return texto
            textos.append(p(f"    columna {col}: ", tira))
        ventanas = p("  Ventanas con pieza (ej. 1:am 3:az/rs; Enter si ninguna): ", leer_ventanas)
        jugadores.append({
            "nombre": nombre, "tablero": tab, "puntos": puntos, "cristal_roto": casilla,
            "vidriero": vidriero, "tiras": textos,
            "ventanas": [[c.name if c else None for c in v] for v in ventanas],
        })

    salida("\nMesa (escribe - si está vacía)")
    fabricas = []
    for i in range(1, NUM_FACTORIES + 1):
        def fabrica(texto):
            colores = leer_colores(texto)
            if len(colores) > TILES_PER_FACTORY:
                raise ValueError(f"En una fábrica caben {TILES_PER_FACTORY} piezas.")
            return [c.name for c in colores]
        fabricas.append(p(f"  Fábrica {i}: ", fabrica))
    centro = p("  Centro: ", lambda t: [c.name for c in leer_colores(t)])
    nombres = [j["nombre"] for j in jugadores]

    loseta = p(f"  ¿Dónde está la loseta inicial? (centro, {nombres[0]} o {nombres[1]}): ",
               _una_de(["centro"] + nombres))
    turno = p(f"  ¿A quién le toca? ({nombres[0]} o {nombres[1]}): ", _una_de(nombres))
    torre = p("  Piezas en la torre de cristal (Enter si no se saben; se dan por estar en la bolsa): ",
              lambda t: [c.name for c in leer_colores(t)])

    return {
        "ronda": ronda, "colores_ronda": marcador, "turno": turno, "loseta": loseta,
        "fabricas": fabricas, "centro": centro, "torre": torre, "jugadores": jugadores,
    }


def abrir_partida(nombres_nuevos, entrada=input, salida=print, seed=None) -> AzulVitralGame:
    """Pregunta si la partida es nueva, se escribe el estado o se carga de un archivo."""
    salida("1) Partida nueva")
    salida("2) Partida en curso: escribir el estado")
    salida("3) Partida en curso: cargar un archivo .json")
    opcion = _preguntar("Elige 1, 2 o 3: ", _una_de(["1", "2", "3"]), entrada, salida)
    if opcion == "1":
        return AzulVitralGame(nombres_nuevos, seed=seed)
    while True:
        if opcion == "2":
            estado = preguntar_estado(entrada, salida)
        else:
            ruta = entrada("Ruta del archivo: ").strip()
            try:
                estado = cargar_estado(ruta)
            except (OSError, json.JSONDecodeError) as err:
                salida(f"  No se pudo leer el archivo: {err}")
                continue
        try:
            game = partida_desde_estado(estado, seed=seed)
        except (ValueError, KeyError, TypeError) as err:
            salida(f"\nEl estado no cuadra: {err}")
            if opcion == "2":
                ruta = entrada("Ruta para guardar lo que escribiste y corregirlo (Enter para no guardar): ").strip()
                if ruta:
                    guardar_estado(estado, ruta)
                    salida(f"Guardado en {ruta}. Corrígelo y elige la opción 3.")
            opcion = _preguntar("¿Probar de nuevo? 2) escribir  3) archivo: ",
                                _una_de(["2", "3"]), entrada, salida)
            continue
        if opcion == "2":
            ruta = entrada("Ruta para guardar este estado (Enter para no guardar): ").strip()
            if ruta:
                guardar_estado(estado, ruta)
        return game
