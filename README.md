# Azul: Vitrales de Sintra (2 jugadores)

Simulación del juego en Python 3, sin dependencias externas.

## Archivos

- `azulvitral.py`: motor de reglas (tiras, fábricas, centro, jugadores y la partida).
- `partida.py`: `TwoPlayerMatch`, que lleva los turnos y le pide la jugada a cada jugador.
- `impresion.py`: funciones para mostrar el estado en consola. Solo leen.
- `jugar.py`: partida en consola entre dos personas.
- `contra_maquina.py`: partida en consola contra la máquina, que elige al azar.
- `estado.py`: cargar una partida que ya va en curso (respondiendo preguntas o desde un archivo).
- `ejemplo_partida.py`: una partida completa con jugadas fijas.
- `generar_ejemplo.py`: el programa que armó las jugadas del ejemplo.
- `datos/tiras_patron.json`: las 8 tiras de cada uno de los 4 tableros, con sus dos caras,
  transcritas de las capturas de Tabletop Simulator.
- `datos/estado_ejemplo.json`: una partida en curso de ejemplo, para cargarla o copiar el formato.
- `tests/test_motor.py`: pruebas del motor.
- `DOCUMENTACION.md`: cada clase con sus métodos, y cómo se hizo la partida de ejemplo.

## Cómo jugar

Desde la carpeta del proyecto:

    python3 jugar.py                  (los jugadores se llaman Jugador 1 y Jugador 2)
    python3 jugar.py Ana Beto         (con nombres)
    python3 jugar.py Ana Beto 7       (con una semilla, para repetir el mismo reparto)

Para jugar contra la máquina:

    python3 contra_maquina.py             (tú te llamas Jugador)
    python3 contra_maquina.py Ana         (con tu nombre)
    python3 contra_maquina.py Ana 7       (con una semilla)

La máquina no intenta ganar: en cada turno mira todas sus jugadas válidas y
elige una al azar, todas con la misma probabilidad, y muestra cuál eligió.

Los dos programas preguntan primero:

    1) Partida nueva
    2) Partida en curso: escribir el estado
    3) Partida en curso: cargar un archivo .json

Con una partida en curso, en `contra_maquina.py` además se dice cuál de los dos
jugadores eres tú. Cómo se entrega el estado está más abajo, en "Cargar una
partida en curso".

El programa muestra la mesa y le pide la jugada a quien le toca. Se escribe así:

| Escribes | Qué hace |
|---|---|
| `3 azul 2` | Toma todas las azules de la fábrica 3 y las pone en la tira 2 |
| `c naranjo 4` | Toma todas las naranjas del centro y las pone en la tira 4 |
| `1 rosado 5 rosado` | Si la tira 5 se completa, la cuarta palabra es la pieza que queda en la ventana |
| `2 amarillo -` | Si no queda ninguna tira bajo el vidriero ni a su derecha: se rompen todas |
| `b` | Opción B: el vidriero vuelve a la tira de más a la izquierda |
| `jugadas` | Lista todas las jugadas válidas en ese momento |
| `salir` | Termina el programa |

Los colores se pueden abreviar con dos letras: `na`, `am`, `rs`, `az`, `tr`.
Las fábricas van de 1 a 5 y las tiras (columnas) de 1 a 8, como en la mesa.
Si la jugada no vale, el programa dice por qué y la vuelve a pedir.

### En cada turno

- **Opción A.** Tomar todas las piezas de un color de una fábrica (el resto
  pasa al centro) o del centro. El primero que toma del centro en la ronda se
  lleva la loseta inicial y avanza 1 en el cristal roto. Todas van a una sola
  tira, bajo el vidriero o a su derecha; si es a la derecha, el vidriero se
  mueve. Las que no caben se rompen y avanzan el track de cristal roto. Si
  ninguna tira alcanzable tiene espacio para ese color, se elige igual una tira
  y se rompen todas; si alguna tiene espacio, hay que usarla.
- **Opción B.** El vidriero vuelve a la tira de más a la izquierda.

Al completar una tira: +1 por cada pieza del color de la ronda, una pieza va a
la ventana (la primera vez arriba y la tira se voltea; la segunda abajo y la
tira se retira) y se suman los puntos de esa ventana más los de toda ventana a
su derecha que ya tenga piezas.

Al llegar a la casilla 10 del cristal roto se pierden 18 puntos en ese mismo
turno y el marcador vuelve a la casilla 1. El puntaje nunca baja de 0.

Al final de la ronda 6: +1 por cada 3 piezas que quedaron en las tiras, se
resta la casilla del cristal roto y se suman los ornamentos (por cada uno, sus
4 espacios: 0 o 1 con pieza +0, 2 +3, 3 +6, 4 +10). Gana el que tiene más
puntos; si empatan, el que quedó en una casilla más baja del cristal roto, y si
siguen empatados es empate técnico.

## Cargar una partida en curso

Con la opción 2 el programa pregunta dato por dato y, si algo no sirve, dice por
qué y lo vuelve a preguntar:

1. La ronda actual y los colores que quedan en el marcador de ronda, desde la
   ronda actual hasta la 6 (por ejemplo `az rs am tr` en la ronda 3).
2. Por cada jugador: nombre, tablero, puntos, casilla del cristal roto, columna
   del vidriero, sus 8 tiras y sus ventanas.
3. Las 5 fábricas, el centro, dónde está la loseta inicial, a quién le toca y
   las piezas de la torre de cristal.

Cada tira se escribe como se ve, de arriba hacia abajo:

| Escribes | Significa |
|---|---|
| `tr` | espacio transparente libre |
| `TR` | espacio transparente con su pieza puesta |
| `**` | comodín libre |
| `*na` | comodín con una pieza naranja |
| `-` | la tira ya se retiró del tablero |

Por ejemplo `tr tr am am am` es una tira con dos espacios transparentes y tres
amarillos, todos libres, y `tr tr AM AM AM` es la misma tira con los tres
amarillos ya puestos. Con la cara que se ve, el programa reconoce qué tira es
y qué tiene al reverso.

Las ventanas se escriben juntas, solo las que tienen pieza: `1:tr 3:az/rs` es
una transparente arriba en la ventana 1, y en la 3 una azul arriba y una
rosada abajo. Las fábricas y el centro se escriben con sus colores (`az az na
rs`) o `-` si están vacíos. Si no se sabe qué hay en la torre se deja en blanco:
las piezas que no están a la vista se dan por estar en la bolsa.

Al terminar, el programa revisa que todo cuadre (que no haya más de 20 piezas
de un color, que una tira no esté en dos columnas, que una tira retirada tenga
su ventana llena, etc.). Si algo no cuadra dice qué, y ofrece guardar lo
escrito en un archivo para corregirlo y cargarlo con la opción 3. Si todo está
bien, ofrece guardarlo igual, para no tener que escribirlo de nuevo.

El archivo es un JSON con los mismos datos; `datos/estado_ejemplo.json` sirve
de modelo.

## Otros comandos

    python3 ejemplo_partida.py             partida de ejemplo, la mesa al empezar cada ronda
    python3 ejemplo_partida.py --detalle   la mesa después de cada turno
    python3 generar_ejemplo.py             vuelve a armar las jugadas del ejemplo
    python3 -m unittest -v                 corre las pruebas

## Usarlo desde código

    from azulvitral import CENTER, Color, Move
    from impresion import print_game, print_turn
    from partida import TwoPlayerMatch

    partida = TwoPlayerMatch(["Ana", "Beto"], seed=1, first_player="Ana")
    print_game(partida.game)
    reporte = partida.play_turn(Move.take(3, Color.AZUL, 2))  # fábrica 3, tira 2
    print_turn(reporte)

Si la jugada completa la tira, se agrega la pieza que queda en la ventana:
`Move.take(CENTER, Color.NARANJO, 4, Color.NARANJO)`. La opción B es
`Move.reset_glazier()`. Una jugada inválida lanza `InvalidMoveError` con el
motivo y no cambia nada; `game.validate_move(move)` devuelve el motivo sin
lanzar y `game.legal_moves()` lista las jugadas válidas.

Para otra interfaz, a `TwoPlayerMatch` se le pasa en `inputs` una función por
jugador que recibe `(game, player)` y devuelve una `Move`; `run()` juega hasta el
final. `jugar.py` es un ejemplo de eso.

### Archivos de Aprendizaje por Refuerzo (RL)

- `src/env_azul.py`: entorno PettingZoo/Gymnasium v1 (`253` características, `1.621` acciones con máscara).
- `src/env_azul_v2.py`: entorno v2 (`288` características con marcador completo de 6 rondas, *reward shaping* de cascadas/ornamentos y política heurística táctica).
- `scripts/train_rl.py` / `scripts/train_rl_v2.py`: entrenamiento de `MaskablePPO` (`v1` contra azar; `v2` con currículo Self-Play + Heurística).
- `scripts/evaluar_ia.py` / `scripts/evaluar_ia_v2.py`: evaluación cuantitativa en 100 partidas alternando jugador inicial.
- `scripts/contra_maquina.py` / `scripts/contra_maquina_v2.py`: partida en consola contra la IA entrenada (`--v1`, `--heuristica`, `--azar`).
- `models_checkpoint/`: pesos entrenados (`ppo_azul_sintra_v1.zip` y `ppo_azul_sintra_v2.zip`).
- `tests/test_env.py`: pruebas unitarias del entorno RL, biyección de acciones y API de PettingZoo/Gymnasium.

## Inteligencia Artificial (Aprendizaje por Refuerzo)

El proyecto entrena un agente mediante **Proximal Policy Optimization con enmascaramiento de acciones (`MaskablePPO`)** sobre `PettingZoo` (ciclo por turnos de 2 jugadores) y `Gymnasium`:

- **Espacio de acciones biyectivo (`1.621` acciones):** Codifica de forma reversible la Opción B (`0`) y todas las combinaciones de Opción A (`1..1620`: origen, color, tira de destino y color de ventana). En cada turno, `ActionMasker` filtra las jugadas ilegales antes de muestrear la política.
- **Evolución de modelos:**
  - **`v1` (`200.000` pasos):** Entrenada frente a un oponente aleatorio (`253` características).
  - **`v2` (`500.000` pasos):** Entrenada con vector enriquecido (`288` características) y currículo mixto (**45% Heurística táctica**, **35% Self-Play contra `v1`**, **20% Aleatorio**).

### Comandos de IA

```bash
# Jugar en consola contra la IA v2 (o usar --v1 / --heuristica / --azar)
python -m scripts.contra_maquina_v2

# Evaluar rendimiento en 100 partidas contra cada rival
python -m scripts.evaluar_ia_v2

# Reentrenar modelo v2 y visualizar curvas en TensorBoard
python -m scripts.train_rl_v2
tensorboard --logdir=./tensorboard_logs/

# Ejecutar toda la batería de pruebas (motor + entorno RL)
python -m unittest -v
