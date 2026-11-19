# Azul: Vitrales de Sintra (2 jugadores)

Simulación del juego en Python 3, sin dependencias externas.

## Archivos

- `azulvitral.py`: motor de reglas (tiras, fábricas, centro, jugadores y la partida).
- `partida.py`: `TwoPlayerMatch`, que lleva los turnos y le pide la jugada a cada jugador.
- `impresion.py`: funciones para mostrar el estado en consola. Solo leen.
- `datos/tiras_patron.json`: las 8 tiras de cada uno de los 4 tableros, con sus dos caras,
  transcritas de las capturas de Tabletop Simulator.
- `ejemplo_partida.py`: una partida completa con jugadas fijas.
- `tests/test_motor.py`: pruebas del motor.

## Uso

    python3 ejemplo_partida.py
    python3 ejemplo_partida.py --detalle
    python3 -m unittest -v

## Jugar desde código

    from azulvitral import CENTER, Color, Move
    from impresion import print_game, print_turn
    from partida import TwoPlayerMatch

    partida = TwoPlayerMatch(["Ana", "Beto"], seed=1, first_player="Ana")
    print_game(partida.game)
    reporte = partida.play_turn(Move.take(3, Color.AZUL, 2))  # fábrica 3, tira 2
    print_turn(reporte)

Las fábricas van de 1 a 5 y las columnas de 1 a 8, igual que en la mesa. Si la
jugada completa la tira, se agrega la pieza que queda en la ventana:
`Move.take(CENTER, Color.ROJO, 4, Color.ROJO)`. La opción B es
`Move.reset_glazier()`.

Una jugada inválida lanza `InvalidMoveError` con el motivo y no cambia nada.
`game.validate_move(move)` devuelve el motivo sin lanzar y `game.legal_moves()`
lista todas las jugadas válidas.

Para conectar una interfaz, a `TwoPlayerMatch` se le pasa en `inputs` una
función por jugador que recibe `(game, player)` y devuelve una `Move`, y
`run()` juega hasta el final. Con `on_invalid` se le avisa al jugador que su
jugada no vale y se le vuelve a pedir.

## Casos que el reglamento no cubría

Se preguntaron antes de programarlos. En el código están marcados con "Acordado:".

- Opción B: el vidriero vuelve a la tira activa más a la izquierda. Si ya está
  ahí, o en una columna vacía a la izquierda de todas las tiras, la opción B no
  está disponible.
- Comodines: primero se llenan los espacios del color y después los comodines.
- Desborde del track: al llegar a la casilla 10 se pierden 18 y el marcador
  vuelve a la 1; las piezas rotas que faltaban por contar siguen avanzando.
- Desempate: gana quien perdió menos en total por cristal roto (los -18 de cada
  desborde más el valor de la casilla final).
- Sin ninguna tira bajo el vidriero ni a su derecha: se puede tomar igual y se
  rompe todo.
- Si nadie toma del centro en una ronda, la siguiente la empieza quien empezó esa.

Otros detalles: el reglamento no pone mínimo al puntaje, así que puede quedar
negativo. Cada jugador usa uno de los cuatro tableros (por defecto rojo y amarillo).

## Cambios respecto al esqueleto

- `Color.VERDE` pasó a `Color.ROSADO`: en las piezas no hay verde.
- `BROKEN_GLASS_PENALTIES` tiene los 10 valores del reglamento.
- Solo 2 jugadores y solo la bonificación de ornamentos (se sacó `PalaceSide`).
- Las jugadas entran como un objeto `Move` por `apply_move()`, que valida todo
  antes de tocar el estado. Los métodos que mueven piezas quedaron internos
  (con `_` adelante).
