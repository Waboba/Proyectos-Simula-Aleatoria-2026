# Documentación

Qué hace cada clase y cada método, y cómo se armó la partida de ejemplo.

Convención: en las jugadas (`Move`) y en lo que se imprime, las fábricas van
de 1 a 5 y las tiras de 1 a 8, como en la mesa. Adentro del motor, las listas
empiezan en 0 (la tira 1 es `strips[0]`).

## azulvitral.py: el motor

### Constantes

| Nombre | Qué es |
|---|---|
| `Color` | Los 5 colores de las piezas: `NARANJO`, `AMARILLO`, `ROSADO`, `AZUL`, `TRANSPARENTE` |
| `JOKER` | Marca de un espacio comodín en una tira (`"COMODIN"`) |
| `CENTER` | Origen "centro de la mesa" en una jugada (`"centro"`) |
| `BROKEN_GLASS_PENALTIES` | Valor de las casillas 1 a 10 del cristal roto: 0, -1, -2, -4, -5, -8, -10, -12, -15, -18 |
| `WINDOW_POINTS` | Puntos bajo las ventanas 1 a 8: 4, 3, 3, 2, 2, 1, 1, 2 |
| `ORNAMENT_BONUS` | Espacios con pieza alrededor de un ornamento → puntos: 0 y 1 → 0, 2 → 3, 3 → 6, 4 → 10 |
| `NUM_STRIPS`, `STRIP_SIZE`, `NUM_FACTORIES`, `TILES_PER_FACTORY`, `TILES_PER_COLOR`, `NUM_ROUNDS` | 8 tiras, 5 espacios, 5 fábricas, 4 piezas por fábrica, 20 piezas por color, 6 rondas |
| `STRIPS_FILE` | Ruta a `datos/tiras_patron.json` |

### `InvalidMoveError`

Excepción que lanza el motor cuando una jugada no respeta las reglas. El
mensaje dice el motivo, por ejemplo: "La tira 2 está a la izquierda del
vidriero (columna 4); solo se puede usar esa columna o una a su derecha."

### `PatternStrip`: una tira de patrón

Campos: `side_a` y `side_b` (los 5 espacios de cada cara: un `Color` o
`JOKER`), `current_side` (0 o 1, la cara que está arriba), `filled_spaces` (la
pieza puesta en cada espacio, o `None`), `is_removed` (la tira ya se retiró) y
`name` (por ejemplo `rojo-3`, la tira 3 del tablero rojo en el JSON).

| Método | Qué hace |
|---|---|
| `pattern()` | Los 5 espacios de la cara de arriba |
| `is_complete()` | Si los 5 espacios tienen pieza |
| `has_jokers(side)` | Si esa cara tiene comodines |
| `free_slots_for(color)` | Espacios libres donde puede ir una pieza de ese color: primero los de su color y al final los comodines |
| `empty_count()` | Cuántos espacios quedan libres |
| `tiles()` | Las piezas puestas |
| `flip()` | Voltea la tira y la deja vacía |

### `Factory`: una fábrica

Campos: `id` (1 a 5) y `tiles` (sus piezas). Método `is_empty()`.

### `CenterPool`: el centro de la mesa

Campos: `tiles` y `has_first_player_token` (si la loseta inicial está ahí).
`is_empty()` mira solo las piezas: la loseta no cuenta.

### `Player`: un jugador

Campos: `name`, `board` (rojo, amarillo, gris o azul), `strips` (sus 8 tiras,
una por columna; una tira retirada sigue en la lista con `is_removed`),
`windows` (8 ventanas, cada una `[superior, inferior]`), `glazier` (columna
del vidriero, 0 a 7), `score`, `broken_position` (casilla del cristal roto, 1
a 10) y `has_first_player_token`.

| Método | Qué hace |
|---|---|
| `get_leftmost_active_strip_index()` | La tira más a la izquierda que sigue en el tablero |
| `can_reset_glazier()` | Si la opción B está disponible: tiene que haber una tira activa a la izquierda del vidriero |
| `reset_glazier()` | Mueve el vidriero a esa tira |
| `reachable_strips()` | Tiras activas bajo el vidriero o a su derecha |
| `broken_penalty()` | Valor de la casilla actual del cristal roto |
| `add_points(n)` | Suma o resta puntos; el puntaje nunca baja de 0 |
| `leftover_tiles()` | Piezas que quedan en las tiras (para la puntuación final) |

### `Move`: una jugada

Campos: `option` (`"A"` o `"B"`), `source` (fábrica 1 a 5 o `CENTER`),
`color`, `strip` (tira 1 a 8, o `None` si no queda ninguna alcanzable) y
`window_color` (la pieza que queda en la ventana si la tira se completa).
Se arma con `Move.take(source, color, strip, window_color)` o con
`Move.reset_glazier()`.

### `TurnReport`: lo que pasó en un turno

Lo devuelve `apply_move()`: quién jugó, la jugada, cuántas piezas tomó, si se
llevó la loseta inicial, cuántas puso y cuántas se rompieron, si completó la
tira (con la bonificación de color y los puntos de ventanas), si la tira se
retiró, cuántas veces llegó a la casilla 10, y si terminó la ronda o la
partida.

### `FinalScore`: detalle de la puntuación final

`leftover_tiles` y `leftover_points` (piezas sobrantes y sus puntos),
`broken_penalty` (valor de la casilla del cristal roto), `ornament_points` (lo
que dio cada uno de los 4 ornamentos), `score_before` (puntaje al terminar la
ronda 6) y `total`.

### Funciones sueltas

| Función | Qué hace |
|---|---|
| `load_strips(path)` | Lee el JSON: tablero → lista de (anverso, reverso) |
| `generate_default_strips_pool(board)` | Las 8 tiras de un tablero como `PatternStrip` |

### `AzulVitralGame`: la partida

`AzulVitralGame(player_names, boards=("rojo", "amarillo"), first_player=None, seed=None)`
prepara todo según el reglamento:
- Baraja las tiras de cada jugador y les da una cara al azar; si la cara de
  comodines queda arriba, la voltea.
- Arma el marcador de ronda: un color de cada uno en II a VI, y en I una pieza
  sacada de las 95 que quedan.
- Deja 94 piezas en la bolsa, llena las 5 fábricas y pone la loseta inicial en
  el centro.

`first_player` puede ser 0, 1, un nombre o `None` (al azar). `seed` fija el
azar para repetir una partida.

Campos principales: `players`, `factories`, `center`, `bag`, `tower` (la torre
de cristal), `round_marker` (los 6 colores de ronda; `None` los que ya salieron),
`round`, `current` (índice del jugador de turno), `game_over` y `final_scores`.

Métodos públicos:

| Método | Qué hace |
|---|---|
| `apply_move(move)` | Valida la jugada completa y, si vale, la aplica. Si no vale, lanza `InvalidMoveError` y no cambia nada. Devuelve un `TurnReport` |
| `validate_move(move)` | `None` si la jugada vale; si no, el motivo (sin lanzar nada) |
| `legal_moves()` | Todas las jugadas válidas del jugador de turno |
| `get_current_player()` | El jugador al que le toca |
| `get_valid_placement_strips(player, tiles)` | Tiras alcanzables donde cabe al menos una de esas piezas |
| `is_round_over()` | Si ya no quedan piezas en las fábricas ni en el centro |
| `current_round_color()` | El color de la ronda actual |
| `end_game_scoring()` | Puntuación final. La llama sola el motor al terminar la ronda 6 |
| `ranking()` | Jugadores de mejor a peor |
| `winner()` | El ganador, o `None` si hay empate técnico |

Métodos internos (los usa `apply_move`, no validan nada por su cuenta):

| Método | Qué hace |
|---|---|
| `_check_move(move)` | Revisa la jugada contra las reglas y lanza `InvalidMoveError` con el motivo |
| `_create_random_board(board)` | Baraja las tiras de un tablero |
| `_draw_tiles_from_bag(n)` | Saca piezas al azar; si la bolsa se vacía, pasa la torre a la bolsa; si tampoco alcanza, devuelve las que haya |
| `_start_new_round()` | Llena las fábricas (de a 4, en orden, hasta donde alcance), pone la loseta en el centro y le da el turno a quien la tomó la ronda anterior |
| `_end_round()` | La pieza del marcador va a la torre; después de la ronda 6 llama a la puntuación final |
| `_advance_turn()` | Pasa el turno o cierra la ronda |
| `_take_from_factory(i, color)` | Saca las piezas de ese color de una fábrica; el resto va al centro |
| `_take_from_center(player, color)` | Saca las piezas de ese color del centro; el primero de la ronda se lleva la loseta y 1 cristal roto |
| `_place_tiles_on_strip(...)` | Mueve el vidriero si hace falta, pone las piezas y rompe las que no caben |
| `_process_completed_strip(...)` | Bonificación de color, pieza a la ventana, volteo o retiro de la tira, puntos de ventanas en cascada |
| `_execute_reset_glazier_turn(player)` | Opción B |
| `_add_broken_glass(player, n)` | Avanza el track n casillas; en la 10 resta 18 en ese turno y vuelve a la 1 |

El orden dentro de un turno sigue el reglamento:
1. La loseta inicial, si se toma del centro.
2. Las piezas rotas.
3. Los puntos por completar la tira.

Por eso, si un jugador desborda y completa una tira en el mismo turno, primero
queda en 0 (si no le alcanzaba para los 18) y después suma lo de la tira.

## partida.py: `TwoPlayerMatch`

`TwoPlayerMatch(names, inputs=None, boards=..., first_player=None, seed=None, on_turn=None, on_invalid=None, game=None)`
crea la partida (`self.game`) y la lleva turno a turno.

- `game`: una partida ya armada (por ejemplo una en curso cargada con
  `estado.py`). Si se entrega, se sigue desde ahí y no se usan `boards`,
  `first_player` ni `seed`.

- `inputs`: un diccionario nombre → función. Cada función recibe `(game,
  player)` y devuelve la `Move` de ese jugador. Así la entrada puede venir de
  la consola, de una interfaz gráfica o de una lista fija.
- `on_turn(report, game)`: se llama después de cada jugada válida.
- `on_invalid(player, move, error)`: se llama cuando una jugada no vale, y se
  vuelve a pedir. Si no se entrega, el error se lanza.

| Método | Qué hace |
|---|---|
| `play_turn(move)` | Aplica la jugada del jugador de turno (para pasar jugadas a mano) |
| `ask_and_play()` | Le pide la jugada a quien le toca hasta que entregue una válida |
| `run()` | Juega hasta el final y devuelve el ganador (`None` si hay empate técnico) |
| `winner()` | El ganador |
| `current_player`, `is_over` | Jugador de turno y si la partida terminó |
| `history` | Lista de `TurnReport` de toda la partida |

## impresion.py: mostrar el estado

Solo leen el estado; se pueden llamar en cualquier momento. Cada `print_x`
tiene su `x_text` que devuelve el texto en vez de imprimirlo.

| Función | Muestra |
|---|---|
| `print_header` | Ronda, color de la ronda, de quién es el turno, marcador de ronda, bolsa y torre |
| `print_factories` | Contenido de cada fábrica |
| `print_center` | Contenido del centro y quién tiene la loseta inicial |
| `print_player` | Puntos, casilla del cristal roto, loseta, posición del vidriero, las 8 tiras con sus piezas, las ventanas y cuántos espacios hay ocupados alrededor de cada ornamento |
| `print_game` | Todo lo anterior junto |
| `print_turn` | Una línea con lo que pasó en un turno |
| `print_final` | El detalle de la puntuación final y el resultado |

En las tiras, `[NA]` es una pieza puesta, `na` un espacio libre de ese color,
`**` un comodín libre y `--` una tira retirada.

## jugar.py: partida en consola

| Función | Qué hace |
|---|---|
| `leer_jugada(texto)` | Convierte lo que escribe el jugador (`3 azul 2`, `c na 4 na`, `b`) en una `Move` |
| `describir(move)` | Lo contrario: escribe una `Move` como texto |
| `pedir_por_consola(game, player)` | La función de entrada que usa `TwoPlayerMatch` |
| `avisar`, `mostrar_turno` | Lo que se imprime cuando una jugada no vale y después de cada turno |

Al empezar llama a `abrir_partida` (de `estado.py`), así que se puede partir de
cero o seguir una partida en curso.

## contra_maquina.py: partida contra la máquina

| Función | Qué hace |
|---|---|
| `jugador_al_azar(azar)` | Devuelve una función de entrada para `TwoPlayerMatch` que pide `game.legal_moves()` y elige una con `azar.choice`, o sea, todas las jugadas válidas con la misma probabilidad. Imprime cuántas había y cuál eligió |
| `elegir_humano(nombres)` | En una partida cargada, pregunta cuál de los dos jugadores eres tú |
| `main()` | Abre la partida (nueva o en curso) y la juega: tú por consola, la máquina al azar |

`legal_moves()` cuenta cada variante como una jugada distinta: tomar azul de la
fábrica 2 y ponerlo en la columna 3 es una jugada, en la columna 5 es otra, y
cada color posible para la ventana también. Opción B (mover el vidriero) es una
jugada más.

## estado.py: cargar una partida en curso

El estado es un diccionario con estas llaves: `ronda`, `colores_ronda`, `turno`,
`loseta` (`"centro"` o el nombre de quien la tiene), `fabricas` (5 listas de
colores), `centro`, `torre` y `jugadores`. Cada jugador trae `nombre`,
`tablero`, `puntos`, `cristal_roto`, `vidriero` y sus `tiras` y `ventanas` como
texto.

| Función | Qué hace |
|---|---|
| `leer_color(texto)` | Lee un color por su nombre o sus dos primeras letras |
| `leer_colores(texto)` | Una lista de colores separados por espacios (`-` es ninguno) |
| `leer_tira(texto)` / `escribir_tira(strip)` | Una tira como se ve: minúscula libre, MAYÚSCULA puesta, `**` comodín libre, `*na` comodín con pieza, `-` retirada |
| `leer_ventanas(texto)` / `escribir_ventanas(windows)` | Las ventanas con pieza: `1:am 3:az/rs` (arriba/abajo) |
| `partida_desde_estado(estado, seed=None)` | Arma un `AzulVitralGame` desde el estado. Reconoce cada tira por su cara visible, pone en la bolsa las piezas que no están a la vista, y rechaza con `ValueError` (y el motivo) un estado que no puede darse: más de 20 piezas de un color, una tira repetida, una ventana a medias, una fábrica con más de 4, etc. |
| `estado_de_partida(game)` | Lo contrario: saca el estado de una partida |
| `guardar_estado(estado, ruta)` / `cargar_estado(ruta)` | Escribe y lee el estado como JSON |
| `preguntar_estado(entrada, salida)` | Pregunta el estado dato por dato y vuelve a preguntar lo que no se entiende |
| `abrir_partida(nombres_nuevos, ...)` | El menú del principio: partida nueva, escribir el estado o cargar un archivo. Si el estado escrito no cuadra, ofrece guardarlo para corregirlo |

Lo que no se puede saber mirando la mesa se resuelve así: las piezas que no
están a la vista se dan por estar en la bolsa, y la torre queda vacía si no se
dice qué tiene.

## Cómo se hizo la partida de ejemplo

`ejemplo_partida.py` reproduce una partida completa entre Ana (tablero rojo) y
Beto (tablero amarillo), con la semilla 2026 y Ana como jugadora inicial. La
semilla fija el reparto: mismas tiras, mismo marcador de ronda y mismas piezas
en las fábricas cada vez.

Las jugadas no se escribieron a mano: las armó `generar_ejemplo.py`. Ese programa juega los
dos lados con una estrategia simple:
1. Para cada jugada válida, la prueba en una copia de la partida.
2. Le pone una nota:
   - los puntos que gana;
   - menos 0,6 por pieza rota;
   - más 0,35 por pieza puesta;
   - menos 0,25 por cada columna que mueve el vidriero a la derecha;
   - más 0,15 por cada pieza que deja en una tira sin completar.
   - La opción B vale `0,25 × columna del vidriero − 1,2`: conviene más
     mientras más a la derecha esté.
3. Elige la de mejor nota; si empatan, la de la tira más a la izquierda.

No busca jugar bien, solo que la partida pase por todas las reglas. Se probaron
las semillas 2020 a 2039, y se eligió 2026 porque su partida tiene de todo:
66 turnos (32 de Ana, 34 de Beto), 9 veces la opción B, 18 tiras completas
(varias con puntos en cascada), 5 tiras retiradas, 15 piezas rotas y un
desborde del cristal roto. En ese desborde, Beto tenía 16 puntos y quedó en 0
antes de sumar la tira que completó.

`generar_ejemplo.py` imprime las jugadas en el formato que usa
`ejemplo_partida.py`, que es el que quedó copiado en el archivo. El resultado
final es Ana 39, Beto 50. Hay una prueba (`test_el_ejemplo_llega_al_final`) que
juega el ejemplo entero y revisa ese resultado, así que si un cambio en el
motor vuelve inválida alguna jugada, la prueba lo avisa.

## env_azul.py y env_azul_v2.py: entornos de Aprendizaje por Refuerzo

Exponen `AzulVitralGame` como un entorno multi-agente por turnos (`PettingZoo AECEnv`) y un entorno de un solo jugador (`Gymnasium`) compatible con `sb3-contrib`.

### Codificación de acciones (`1.621` índices discretos)

| Función | Qué hace |
|---|---|
| `encode_move(move)` | Convierte un `Move` en un entero de `0` a `1620`: `0` para Opción B (`Move.reset_glazier()`), y `1..1620` para Opción A según `6 orígenes × 5 colores × 9 destinos de tira × 6 opciones de ventana` |
| `decode_action(action_idx)` | Operación inversa exacta de `encode_move`: reconstruye el objeto `Move` a partir de su índice entero |

### Observación y política heurística (`env_azul_v2.py`)

| Función | Qué hace |
|---|---|
| `build_observation_v2(game, player_idx)` | Genera un vector `float32` normalizado de **288 dimensiones** desde la perspectiva del jugador `player_idx`: fábricas (`25`), centro y loseta inicial (`6`), ronda actual y marcador completo de 6 rondas (`31`), suministro en bolsa y torre (`7`), estado egocéntrico de tiras, ventanas, vidriero, cristal roto y puntaje de ambos jugadores (`218`), más indicador de asiento (`1`) |
| `heuristic_move(game, rng)` | Evalúa todas las jugadas de `game.legal_moves()` sin copiar el estado (priorizando colocación sin romper piezas, bonificación de color de ronda, puntuación en cascada y pares de ornamentos). Alcanza ~63 puntos promedio contra el agente aleatorio |

### Clases de entorno

| Clase | Qué hace |
|---|---|
| `AzulSintraEnvV2` | Subclase de `pettingzoo.AECEnv` para `player_0` y `player_1`. Entrega `{"observation": ..., "action_mask": ...}` en `observe()`, valida jugadas legales en `action_mask()` y asigna recompensa inmediata ($\Delta \text{score}$ + *shaping* por cascadas y ornamentos) más $\pm 60$ al finalizar la ronda 6 |
| `AzulGymWrapperV2` | Subclase de `gymnasium.Env` que fija a la IA en `player_0` o `player_1` al azar en cada episodio y ejecuta automáticamente los turnos del oponente (incluyendo dobles turnos al cambiar de ronda) dentro de `step()` |

## Scripts de entrenamiento, evaluación y pruebas RL

| Archivo | Qué hace |
|---|---|
| `scripts/train_rl.py` / `scripts/train_rl_v2.py` | Vectorizan 4 entornos paralelos con `DummyVecEnv` + `ActionMasker` y entrenan `MaskablePPO` (`200.000` pasos en `v1`; `500.000` pasos con currículo Self-Play + Heurística en `v2`) |
| `scripts/evaluar_ia.py` / `scripts/evaluar_ia_v2.py` | Evalúan los modelos cargados con `deterministic=True` en 100 partidas alternando el jugador inicial |
| `scripts/contra_maquina_v2.py` | Permite jugar por consola contra `ppo_azul_sintra_v2.zip` (o contra `--v1`, `--heuristica` y `--azar`) |
| `tests/test_env.py` | Verifica la biyección de las 1.621 acciones, el rango de observaciones, `pettingzoo.test.api_test`, la igualdad entre `action_mask` y `legal_moves()`, y la integración con `ActionMasker` |
