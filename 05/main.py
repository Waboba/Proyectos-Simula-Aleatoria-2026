"""
Poker Texas Hold'em - 1 vs 1 (heads-up), por consola.

Toda la lógica del juego vive en motor.py (MotorPoker), que es puro y
no imprime ni pide input. Este archivo es solo la capa de consola:
muestra el estado y traduce lo que escribe la persona a llamadas de
motor.aplicar_accion(). Así el mismo motor sirve para jugar por
consola, para guardar/cargar partidas, o para que una IA juegue sin
pasar por aquí.
"""

from Clases.jugador import Jugador
from Clases.motor import MotorPoker


# ----------------------------------------------------------------------
# Utilidades de presentación
# ----------------------------------------------------------------------

def mostrar_estado(motor: MotorPoker) -> None:
    jugador_a, jugador_b = motor.jugadores
    print(f"\n--- {motor.fase} ---")
    print(f"Pozo: {motor.mesa.pozo}")
    if motor.mesa.cartas_comunitarias:
        print(f"Mesa: {motor.mesa.cartas_comunitarias}")
    print(f"{jugador_a.nombre}: {jugador_a.fichas} fichas | apuesta actual: {jugador_a.apuesta_actual}")
    print(f"{jugador_b.nombre}: {jugador_b.fichas} fichas | apuesta actual: {jugador_b.apuesta_actual}")


def pedir_accion_cli(info: dict) -> str:
    print(f"\nTurno de {info['jugador']}")

    mapa = {"pasar": "p", "igualar": "i", "subir": "s", "retirarse": "r"}
    etiquetas = {
        "pasar": "[p]asar",
        "igualar": f"[i]gualar ({info['a_igualar']} fichas)",
        "subir": "[s]ubir",
        "retirarse": "[r]etirarse",
    }
    opciones_texto = ", ".join(etiquetas[a] for a in info["acciones"])
    letras_validas = {mapa[a]: a for a in info["acciones"]}

    while True:
        letra = input(f"{opciones_texto} > ").strip().lower()
        if letra in letras_validas:
            return letras_validas[letra]
        print("Opción inválida, intenta de nuevo.")


def pedir_monto_subida(fichas_disponibles: int, minimo: int) -> int:
    while True:
        try:
            monto = int(input(f"¿Cuánto quieres subir (mínimo {minimo}, tienes {fichas_disponibles})? > "))
        except ValueError:
            print("Ingresa un número válido.")
            continue
        if monto < minimo and monto < fichas_disponibles:
            print(f"La subida mínima es {minimo}.")
            continue
        return monto


# ----------------------------------------------------------------------
# Una mano completa jugada por consola, usando el motor
# ----------------------------------------------------------------------

def jugar_mano_por_consola(motor: MotorPoker) -> None:
    motor.iniciar_mano()
    print(f"\n{motor.jugadores[motor.dealer_idx].nombre} reparte (ciega chica {motor.ciega_pequena}), "
          f"{motor.jugadores[1 - motor.dealer_idx].nombre} pone ciega grande ({motor.ciega_grande}).")

    fase_mostrada = None
    while not motor.esta_terminada():
        if motor.fase != fase_mostrada:
            mostrar_estado(motor)
            fase_mostrada = motor.fase

        info = motor.acciones_legales()
        jugador = motor.jugador_en_turno()

        if jugador is None:
            break

        print(f"(mano de {jugador.nombre}: {jugador.mano})")

        accion = pedir_accion_cli(info)
        monto = 0
        if accion == "subir":
            monto = pedir_monto_subida(info["fichas_disponibles"], motor.ciega_grande)

        motor.aplicar_accion(accion, monto)

    mostrar_estado(motor)

    if motor.ganadores is None:
        print("\nError: La mano terminó pero no se registraron ganadores.")
        return

    if len(motor.ganadores) == 1:
        print(f"\n¡Gana {motor.ganadores[0]}!")
    else:
        print(f"\n¡Empate entre {' y '.join(motor.ganadores)}! Se dividió el pozo.")


# ----------------------------------------------------------------------
# Programa principal
# ----------------------------------------------------------------------

def main() -> None:
    print("=== Poker 1v1 ===")
    nombre_a = input("Nombre del jugador 1: ").strip() or "Jugador 1"
    nombre_b = input("Nombre del jugador 2: ").strip() or "Jugador 2"

    jugador_a = Jugador(nombre_a)
    jugador_b = Jugador(nombre_b)

    dealer_idx = 0
    while jugador_a.fichas > 0 and jugador_b.fichas > 0:
        print("\n" + "=" * 40)
        motor = MotorPoker(jugador_a, jugador_b, dealer_idx=dealer_idx)
        jugar_mano_por_consola(motor)

        print(f"\nFichas -> {jugador_a.nombre}: {jugador_a.fichas} | {jugador_b.nombre}: {jugador_b.fichas}")

        if jugador_a.fichas == 0 or jugador_b.fichas == 0:
            break

        continuar = input("\n¿Jugar otra mano? [s/n] > ").strip().lower()
        if continuar != "s":
            break

        dealer_idx = 1 - dealer_idx  # el botón rota cada mano

    if jugador_a.fichas == 0:
        print(f"\n{jugador_b.nombre} gana la partida!")
    elif jugador_b.fichas == 0:
        print(f"\n{jugador_a.nombre} gana la partida!")
    else:
        print("\nPartida terminada.")


if __name__ == "__main__":
    main()