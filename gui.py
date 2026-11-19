"""
Vista gráfica (tkinter) para Poker Texas Hold'em 1 vs 1.

Maneja la interfaz de usuario sobre el MotorPoker.
Ejecutar con: python gui.py
"""
import numpy as np
import tkinter as tk
from tkinter import messagebox

from Clases.jugador import Jugador
from Clases.motor import MotorPoker

# Paleta de colores estilo Casino
FELT = "#0f5228"
FELT_DARK = "#09361a"
FELT_BORDER = "#1b7a3e"
GOLD = "#ffd700"
GOLD_DARK = "#c5a000"
CARD_BG = "#ffffff"
CARD_BACK = "#1e3c72"
PALOS_ROJOS = {"corazon", "corazón", "diamante", "♥", "♦"}
SIMBOLOS_PALOS = {
    "pica": "♠", "trebol": "♣", "trébol": "♣", "corazon": "♥", "corazón": "♥", "diamante": "♦",
    "♠": "♠", "♣": "♣", "♥": "♥", "♦": "♦"
}

FONT_MAIN = "Helvetica"


def es_palo_rojo(pinta: str) -> bool:
    return str(pinta).lower() in PALOS_ROJOS


def obtener_simbolo(pinta: str) -> str:
    pinta_str = str(pinta).lower()
    return SIMBOLOS_PALOS.get(pinta_str, pinta_str)


def crear_carta_widget(parent: tk.Widget, carta=None, oculta: bool = False) -> tk.Frame:
    """Crea una carta visualmente estilizada.
    Soporta tanto enteros (0-51) de NumPy/RL como objetos Carta.
    """
    if oculta or carta is None:
        frame = tk.Frame(parent, bg=CARD_BACK, bd=2, relief="raised", width=64, height=90)
        frame.pack_propagate(False)
        tk.Label(frame, text="♠♣\n♥♦", font=(FONT_MAIN, 12, "bold"), bg=CARD_BACK, fg="#3b67b0").pack(expand=True)
        return frame

    # 1. Si la carta es un entero (0-51) proveniente del nuevo Mazo/Mesa numérico
    if isinstance(carta, (int, np.integer)):
        card_id = int(carta)
        numero_val = (card_id // 4) + 2
        pintas_lista = ["pica", "trebol", "diamante", "corazon"]
        pinta_str = pintas_lista[card_id % 4]
    else:
        # 2. Si la carta es una instancia de la clase Carta
        pinta_str = getattr(carta, "pinta", "")
        numero_val = getattr(carta, "figura", None) or getattr(carta, "numero", "")

    # Mapeo de valores a texto visual (A, K, Q, J)
    valores_map = {14: "A", 13: "K", 12: "Q", 11: "J", 10: "10"}
    if isinstance(numero_val, int) and numero_val in valores_map:
        valor_texto = valores_map[numero_val]
    else:
        valor_texto = str(numero_val) if numero_val is not None else ""

    simbolo = obtener_simbolo(pinta_str)
    color = "#d63031" if es_palo_rojo(pinta_str) else "#2d3436"

    frame = tk.Frame(parent, bg=CARD_BG, bd=2, relief="raised", width=64, height=90)
    frame.pack_propagate(False)

    # Esquina superior izquierda
    tk.Label(frame, text=f"{valor_texto}{simbolo}", font=(FONT_MAIN, 9, "bold"), bg=CARD_BG, fg=color, anchor="w").pack(fill="x", padx=3, pady=(2, 0))
    # Símbolo central
    tk.Label(frame, text=simbolo, font=(FONT_MAIN, 22), bg=CARD_BG, fg=color).pack(expand=True)

    return frame


class PokerGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Texas Hold'em 1v1 - Engine GUI")
        root.configure(bg=FELT_DARK)
        root.geometry("960x760")
        root.minsize(900, 700)

        self.motor: MotorPoker | None = None
        self.jugadores: tuple[Jugador, Jugador] | None = None
        self.dealer_idx = 0
        self.mano_visible = False

        self.frame = tk.Frame(root, bg=FELT_DARK)
        self.frame.pack(fill="both", expand=True)
        self.mostrar_inicio()

    def limpiar(self):
        for widget in self.frame.winfo_children():
            widget.destroy()

    # ------------------------------------------------------------------
    # Pantalla de Inicio (Reparada)
    # ------------------------------------------------------------------
    def mostrar_inicio(self):
        self.limpiar()

        card_box = tk.Frame(self.frame, bg=FELT, bd=4, relief="ridge", padx=40, pady=30)
        card_box.pack(expand=True)

        tk.Label(
            card_box, text="♠ ♥ TEXAS HOLD'EM 1v1 ♦ ♣",
            font=(FONT_MAIN, 26, "bold"), bg=FELT, fg=GOLD
        ).pack(pady=(0, 20))

        # Recuperar nombres previos si ya se jugó una partida
        nombre1 = self.jugadores[0].nombre if self.jugadores else ""
        nombre2 = self.jugadores[1].nombre if self.jugadores else ""
        nombres = [nombre1, nombre2]

        self.entradas = []
        for i in range(2):
            lbl_frame = tk.Frame(card_box, bg=FELT)
            lbl_frame.pack(fill="x", pady=6)

            tk.Label(
                lbl_frame, text=f"Jugador {i+1}:", bg=FELT, fg="white",
                font=(FONT_MAIN, 12, "bold"), width=10, anchor="w"
            ).pack(side="left")

            # Colores explícitos para evitar fallos de renderizado al recrear el widget
            entry = tk.Entry(
                lbl_frame, font=(FONT_MAIN, 12), justify="center", bd=2,
                bg="white", fg="#2d3436", insertbackground="black", relief="sunken"
            )
            entry.insert(0, nombres[i])
            entry.pack(side="left", fill="x", expand=True, padx=5)
            self.entradas.append(entry)

        # Botón Iniciar Partida estilizado
        btn_iniciar = self.crear_boton_accion(
            card_box, "Iniciar Partida", self.empezar,
            color_base=GOLD, color_hover=GOLD_DARK, width=18
        )
        btn_iniciar.config(fg="black")
        btn_iniciar.pack(pady=(25, 0))

        # Dar foco automático a la primera casilla y refrescar interfaz
        if self.entradas:
            self.entradas[0].focus_set()
        self.root.update_idletasks()

        
    def empezar(self):
        n1 = self.entradas[0].get().strip() or ""
        n2 = self.entradas[1].get().strip() or ""
        self.jugadores = (Jugador(n1), Jugador(n2))
        self.dealer_idx = 0
        self.nueva_mano()

    def nueva_mano(self):
        if not self.jugadores:
            return
        self.motor = MotorPoker(*self.jugadores, dealer_idx=self.dealer_idx)
        self.motor.iniciar_mano()
        self.mano_visible = False
        self.refrescar()

    # ------------------------------------------------------------------
    # Renderizado Principal de la Mesa
    # ------------------------------------------------------------------
    def refrescar(self):
        self.limpiar()
        m = self.motor
        if not m:
            return

        terminada = m.esta_terminada()
        en_turno = None if terminada else m.jugador_en_turno()

        # 1. Contenedor del Tapete de la Mesa (Borde exterior)
        mesa_frame = tk.Frame(self.frame, bg=FELT, bd=6, relief="ridge")
        mesa_frame.pack(fill="both", expand=True, padx=30, pady=15)

        # 2. Frame contenedor INTERNO (se expande para centrar todo verticalmente)
        mesa_inner = tk.Frame(mesa_frame, bg=FELT)
        mesa_inner.pack(expand=True, fill="both", padx=20, pady=15)

        # Asiento Oponente (Jugador 2 - arriba)
        self.dibujar_asiento(mesa_inner, 1, terminada, en_turno)

        # Zona Centro (Comunitarias + Pozo)
        centro = tk.Frame(mesa_inner, bg=FELT_DARK, bd=3, relief="solid")
        centro.pack(pady=15, padx=80, fill="x")

        tk.Label(
            centro, text=f"• {m.fase} •", font=(FONT_MAIN, 11, "bold"),
            bg=FELT_DARK, fg="#aaa"
        ).pack(pady=(4, 0))

        tk.Label(
            centro, text=f"POZO: ${m.mesa.pozo}", font=(FONT_MAIN, 22, "bold"),
            bg=FELT_DARK, fg=GOLD
        ).pack()

        cards_row = tk.Frame(centro, bg=FELT_DARK)
        cards_row.pack(pady=10)

        comunes = list(m.mesa.cartas_comunitarias or [])
        for i in range(5):
            if i < len(comunes):
                crear_carta_widget(cards_row, comunes[i]).pack(side="left", padx=5)
            else:
                # Slot vacío
                slot = tk.Frame(cards_row, bg=FELT, bd=1, relief="sunken", width=64, height=90)
                slot.pack_propagate(False)
                slot.pack(side="left", padx=5)

        # Asiento Usuario (Jugador 1 - abajo)
        self.dibujar_asiento(mesa_inner, 0, terminada, en_turno)

        # 3. Panel Inferior de Controles
        panel_acciones = tk.Frame(self.frame, bg=FELT_DARK)
        panel_acciones.pack(fill="x", pady=(0, 15), padx=30)

        if terminada:
            self.dibujar_final(panel_acciones)
        else:
            self.dibujar_acciones(panel_acciones, en_turno)

    def dibujar_asiento(self, parent, idx, terminada, en_turno):
        m = self.motor
        if not m:
            return

        j = m.jugadores[idx]
        es_turno = en_turno is j

        # Borde de destaque si es su turno
        bg_card = "#165b2f" if es_turno else FELT
        bd_color = GOLD if es_turno else FELT_BORDER

        caja = tk.Frame(parent, bg=bg_card, bd=2, relief="solid" if es_turno else "flat", padx=15, pady=6)
        caja.pack(pady=5, fill="x", padx=60)

        top_info = tk.Frame(caja, bg=bg_card)
        top_info.pack(fill="x")

        dealer_badge = " [D]" if idx == m.dealer_idx else ""
        name_color = GOLD if es_turno else "white"

        tk.Label(top_info, text=f"{j.nombre}{dealer_badge}", font=(FONT_MAIN, 14, "bold"), bg=bg_card, fg=name_color).pack(side="left")
        tk.Label(top_info, text=f"Fichas: ${j.fichas}  |  Apuesta: ${j.apuesta_actual}", font=(FONT_MAIN, 12), bg=bg_card, fg="#e0e0e0").pack(side="right")

        cards_row = tk.Frame(caja, bg=bg_card)
        cards_row.pack(pady=4)

        mostrar = terminada or (es_turno and self.mano_visible)
        cartas = list(j.mano or [])
        if not cartas:
            cartas = [None, None]

        for c in cartas:
            crear_carta_widget(cards_row, c, oculta=not mostrar).pack(side="left", padx=4)

        if es_turno and not terminada:
            btn_txt = "🙈 Ocultar Cartas" if self.mano_visible else "👁️ Ver Cartas"
            tk.Button(cards_row, text=btn_txt, font=(FONT_MAIN, 9, "bold"), bg="#34495e", fg="white", activebackground="#2c3e50", relief="flat", cursor="hand2", command=self.alternar_mano).pack(side="left", padx=15)

    def alternar_mano(self):
        self.mano_visible = not self.mano_visible
        self.refrescar()

    # ------------------------------------------------------------------
    # Panel de Controles de Apuesta
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Helper para Botones Estilizados (Compatibles con macOS y Hover)
    # ------------------------------------------------------------------
    def crear_boton_accion(self, parent, texto, comando, color_base, color_hover, estado="normal", width=14):
        """Crea un botón con soporte real de color de fondo y efecto Hover para macOS/Windows/Linux."""
        if estado == "disabled":
            lbl = tk.Label(
                parent, text=texto, font=(FONT_MAIN, 11, "bold"),
                bg="#2c3e50", fg="#7f8c8d", width=width, pady=8,
                relief="flat", bd=0
            )
            return lbl

        lbl = tk.Label(
            parent, text=texto, font=(FONT_MAIN, 11, "bold"),
            bg=color_base, fg="white", width=width, pady=8,
            relief="flat", cursor="hand2", bd=0
        )

        # Efectos interactivos al pasar el cursor
        lbl.bind("<Enter>", lambda e: lbl.config(bg=color_hover))
        lbl.bind("<Leave>", lambda e: lbl.config(bg=color_base))
        lbl.bind("<Button-1>", lambda e: comando())
        return lbl

    # ------------------------------------------------------------------
    # Panel de Controles de Apuesta (Espaciado Ajustado)
    # ------------------------------------------------------------------
    def dibujar_acciones(self, panel, jugador):
        m = self.motor
        if not m or not jugador:
            return

        info = m.acciones_legales()
        acciones = info["acciones"]
        a_igualar = info["a_igualar"]
        disponibles = info["fichas_disponibles"]
        puedo_subir = "subir" in acciones

        # Encabezado con turno activo
        header = tk.Frame(panel, bg=FELT_DARK)
        header.pack(fill="x", pady=(0, 10))
        tk.Label(
            header, text=f"TURNO DE: {info['jugador'].upper()}",
            font=(FONT_MAIN, 13, "bold"), bg=FELT_DARK, fg=GOLD
        ).pack(side="left")

        # Fila principal de botones
        btn_box = tk.Frame(panel, bg=FELT_DARK)
        btn_box.pack(fill="x")

        # 1. Botón Retirarse (Fold)
        st_fold = "normal" if "retirarse" in acciones else "disabled"
        btn_fold = self.crear_boton_accion(
            btn_box, "Retirarse (Fold)", lambda: self.actuar("retirarse"),
            color_base="#d63031", color_hover="#ff7675", estado=st_fold, width=15
        )
        btn_fold.pack(side="left", padx=(0, 6))

        # 2. Botón Pasar / Igualar
        if a_igualar > 0:
            st_call = "normal" if "igualar" in acciones else "disabled"
            btn_call = self.crear_boton_accion(
                btn_box, f"Igualar (${a_igualar})", lambda: self.actuar("igualar"),
                color_base="#0984e3", color_hover="#74b9ff", estado=st_call, width=16
            )
            btn_call.pack(side="left", padx=(0, 6))
        else:
            st_pass = "normal" if "pasar" in acciones else "disabled"
            btn_pass = self.crear_boton_accion(
                btn_box, "Pasar (Check)", lambda: self.actuar("pasar"),
                color_base="#00b894", color_hover="#55efc4", estado=st_pass, width=15
            )
            btn_pass.pack(side="left", padx=(0, 6))

        # 3. Módulo de Subida / Apuesta (Ajustes de espacio interno)
        subir_frame = tk.Frame(btn_box, bg="#111d15", bd=1, relief="solid", padx=12, pady=6)
        subir_frame.pack(side="left", padx=(6, 0), fill="x", expand=True)

        if puedo_subir:
            minimo = min(m.ciega_grande, max(disponibles - a_igualar, 1))
            maximo = max(disponibles - a_igualar, minimo)
            st_subir = "normal"
        else:
            minimo = 0
            maximo = 0
            st_subir = "disabled"

        self.val_subida = tk.IntVar(value=minimo)

        # A) Slider (Deslizador)
        slider = tk.Scale(
            subir_frame, from_=minimo, to=maximo, orient="horizontal",
            variable=self.val_subida, bg="#111d15",
            fg="white" if puedo_subir else "#7f8c8d",
            highlightthickness=0,
            troughcolor="#1e3c28" if puedo_subir else "#18261c",
            activebackground=GOLD, length=250,
            state=st_subir
        )
        slider.pack(side="left", padx=(0, 8))

        # B) Casilla Numérica (Spinbox)
        spin = tk.Spinbox(
            subir_frame, from_=minimo, to=maximo, textvariable=self.val_subida,
            width=6, font=(FONT_MAIN, 11), bd=1, justify="center",
            state=st_subir,
            bg="white", fg="#2d3436",
            disabledbackground="#2c3e50", disabledforeground="#7f8c8d"
        )
        spin.pack(side="left", padx=(0, 14))

        # C) Botón All-In
        btn_max = self.crear_boton_accion(
            subir_frame, "All-In", lambda: self.val_subida.set(maximo) if puedo_subir else None,
            color_base="#6c5ce7", color_hover="#a29bfe", estado=st_subir, width=7
        )
        btn_max.pack(side="left", padx=(0, 8))  # 8px de separación con el botón Subir

        # D) Botón Confirmar Subida
        btn_subir = self.crear_boton_accion(
            subir_frame, "Subir", lambda: self.actuar("subir") if puedo_subir else None,
            color_base="#f1c40f", color_hover="#f39c12", estado=st_subir, width=9
        )
        if puedo_subir:
            btn_subir.config(fg="black")
        btn_subir.pack(side="left", padx=(0, 0))


    def actuar(self, accion: str):
        m = self.motor
        if not m:
            return

        monto = 0
        if accion == "subir":
            info = m.acciones_legales()
            try:
                monto = self.val_subida.get()
            except AttributeError:
                monto = 0

            if monto <= 0:
                messagebox.showwarning("Subir", "Ingresa un monto válido.")
                return

        try:
            m.aplicar_accion(accion, monto)
        except Exception as e:
            messagebox.showerror("Acción Rechazada", str(e))
            return

        self.mano_visible = False
        self.refrescar()

    # ------------------------------------------------------------------
    # Fin de la Mano / Partida (Botones Estilizados)
    # ------------------------------------------------------------------
    def dibujar_final(self, panel):
        m = self.motor
        if not m:
            return

        if not m.ganadores:
            texto = "La mano terminó sin ganadores."
        elif len(m.ganadores) == 1:
            texto = f"🎉 ¡{m.ganadores[0]} gana el pozo!"
        else:
            texto = f"🤝 ¡Empate entre {' y '.join(m.ganadores)}!"

        tk.Label(
            panel, text=texto, font=(FONT_MAIN, 16, "bold"),
            bg=FELT_DARK, fg=GOLD
        ).pack(pady=(4, 8))

        # Caso: Un jugador se quedó sin fichas (Fin de la partida)
        if self.jugadores:
            a, b = self.jugadores
            if a.fichas == 0 or b.fichas == 0:
                campeon = b if a.fichas == 0 else a
                tk.Label(
                    panel, text=f"🏆 {campeon.nombre.upper()} ES EL GANADOR DE LA PARTIDA 🏆",
                    font=(FONT_MAIN, 14, "bold"), bg=FELT_DARK, fg="white"
                ).pack(pady=(0, 10))

                btn_inicio = self.crear_boton_accion(
                    panel, "Nueva Partida", self.mostrar_inicio,
                    color_base="#2ed573", color_hover="#26af5f", width=16
                )
                btn_inicio.pack(pady=4)
                return

        # Caso: Controles para siguiente mano o salir
        btn_box = tk.Frame(panel, bg=FELT_DARK)
        btn_box.pack(pady=6)

        # 1. Botón Siguiente Mano - Dorado llamativo
        btn_sig = self.crear_boton_accion(
            btn_box, "Siguiente Mano ➔", self.siguiente_mano,
            color_base="#f1c40f", color_hover="#f39c12", width=18
        )
        btn_sig.config(fg="black")
        btn_sig.pack(side="left", padx=8)

        # 2. Botón Salir al Menú - Gris pizarra elegante
        btn_menu = self.crear_boton_accion(
            btn_box, "Salir al Menú", self.mostrar_inicio,
            color_base="#4a5568", color_hover="#718096", width=15
        )
        btn_menu.pack(side="left", padx=8)


    def siguiente_mano(self):
        self.dealer_idx = 1 - self.dealer_idx
        self.nueva_mano()


def main():
    root = tk.Tk()
    PokerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()