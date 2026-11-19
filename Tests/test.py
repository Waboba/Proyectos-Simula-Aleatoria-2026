import numpy as np

from Clases.carta import nombre_de
from Clases.jugador import Jugador
from Clases.mesa import Mesa

rng = np.random.default_rng(0)

# 1. Preparación
jugadores = [Jugador("Robbie"), Jugador("Benja")]
mesa = Mesa(rng)

def nombres(cartas):
    return [nombre_de(c) for c in cartas]

# 2. Pre-Flop (Repartir 2 cartas a cada jugador)
for jugador in jugadores:
    jugador.recibir_carta(mesa.mazo.entregar_carta())
    jugador.recibir_carta(mesa.mazo.entregar_carta())

# 3. Flop
mesa.repartir_flop()
print(f"Flop: {nombres(mesa.cartas_comunitarias)}")

# 4. Turn
mesa.repartir_turn()
print(f"Turn: {nombres(mesa.cartas_comunitarias)}")

# 5. River
mesa.repartir_river()
print(f"River: {nombres(mesa.cartas_comunitarias)}")

for jugador in jugadores:
    print(f"{jugador}")