"""
Juega Texas Hold'em heads-up contra uno de tus agentes, en el navegador.

    pip install streamlit
    streamlit run app_web.py

Busca agentes en  resultados/**/agentes/agente_*.npz  (o sube un .npz desde la barra lateral).
Toda la lógica del juego vive en Clases/partida_web.py; este archivo solo dibuja.
"""
import glob
import os
import tempfile

import numpy as np
import streamlit as st

from Clases.abstraccion import N_ESTADOS
from Clases.agente_q import AgenteQ
from partida_web import PartidaContraBot
from Clases.simulacion import PoliticaAleatoria

st.set_page_config(page_title="Poker vs Agente", page_icon="♠️", layout="centered")

OPCION_UNIFORME = "Uniforme (acciones al azar)"
OPCION_SUBIR = "Subir un archivo .npz…"
SIMBOLOS = ("♠", "♣", "♦", "♥")           # Pica, Trébol, Diamante, Corazón (orden de carta.py)


# ----------------------------------------------------------------------------
# dibujo
# ----------------------------------------------------------------------------
def carta_html(c: int | None) -> str:
    base = ("display:inline-flex;flex-direction:column;align-items:center;justify-content:center;"
            "width:58px;height:84px;border-radius:8px;margin:0 4px;font-weight:700;"
            "box-shadow:0 2px 6px rgba(0,0,0,.45);font-family:Helvetica,Arial,sans-serif;")
    if c is None:
        return f'<span style="{base}background:#1e3c72;color:#3b67b0;border:2px solid #3b67b0;font-size:18px">♠♣<br>♥♦</span>'
    num, pinta = (c >> 2) + 2, c & 3
    etiqueta = {11: "J", 12: "Q", 13: "K", 14: "A"}.get(num, str(num))
    color = "#c0392b" if pinta in (2, 3) else "#111111"
    return (f'<span style="{base}background:#ffffff;color:{color};border:1px solid #cccccc;font-size:22px">'
            f'{etiqueta}<span style="font-size:26px;line-height:1">{SIMBOLOS[pinta]}</span></span>')


def fila_cartas(cartas: list[int], total: int = 0, ocultas: bool = False) -> str:
    if ocultas:
        return "".join(carta_html(None) for _ in range(total))
    html = "".join(carta_html(c) for c in cartas)
    html += "".join('<span style="display:inline-block;width:58px;height:84px;margin:0 4px;border-radius:8px;'
                    'border:2px dashed rgba(255,255,255,.25)"></span>' for _ in range(max(total - len(cartas), 0)))
    return html


def asiento_html(nombre: str, fichas: int, apuesta: int, es_dealer: bool, cartas_html: str, jugada: str | None) -> str:
    dealer = ' <span style="background:#ffd700;color:#000;border-radius:10px;padding:1px 8px;font-size:12px">D</span>' if es_dealer else ""
    ap = f' · apuesta <b>{apuesta}</b>' if apuesta else ""
    jug = f'<div style="color:#ffd700;font-size:14px;margin-top:4px">{jugada}</div>' if jugada else ""
    return (f'<div style="text-align:center;margin:6px 0">'
            f'<div style="font-size:15px;margin-bottom:6px"><b>{nombre}</b>{dealer} · fichas <b>{fichas}</b>{ap}</div>'
            f'{cartas_html}{jug}</div>')


def mesa_html(v: dict) -> str:
    nombre_h, nombre_b = v["nombres"]
    dealer = v["dealer_idx"]
    bot = asiento_html(nombre_b, v["fichas"][1], v["apuestas"][1], dealer == 1,
                       fila_cartas(v["mano_bot"] or [], 2, ocultas=v["mano_bot"] is None), v["jugada_bot"])
    yo = asiento_html(nombre_h, v["fichas"][0], v["apuestas"][0], dealer == 0,
                      fila_cartas(v["mano_humano"], 2), v["jugada_humano"])
    centro = (f'<div style="text-align:center;margin:14px 0"><div style="font-size:14px;opacity:.85;margin-bottom:6px">'
              f'{v["fase"]} · pozo <b>{v["pozo"]}</b></div>{fila_cartas(v["mesa"], 5)}</div>')
    return (f'<div style="background:radial-gradient(ellipse at center,#14702f 0%,#0b4a20 100%);color:#fff;'
            f'border:4px solid #5b3a1a;border-radius:24px;padding:14px 10px">{bot}{centro}{yo}</div>')


# ----------------------------------------------------------------------------
# estado de la sesión
# ----------------------------------------------------------------------------
def agentes_disponibles() -> dict[str, str]:
    rutas = sorted(glob.glob(os.path.join("resultados", "mejores_agentes", "agente_*.npz"), recursive=True))
    return {os.path.splitext(os.path.basename(r))[0].replace("_", " ").capitalize(): r for r in rutas}


def crear_partida(fuente: str, ruta: str | None, subido, tau: float, fichas: int, reiniciar: bool) -> None:
    st.session_state.pop("error", None)
    semilla = int(np.random.SeedSequence().entropy % (2**31))
    if fuente == OPCION_UNIFORME:
        politica, nombre = PoliticaAleatoria.uniforme(np.random.default_rng(semilla)), "Uniforme"
    else:
        if fuente == OPCION_SUBIR:
            if subido is None:
                st.session_state.error = "Sube un archivo .npz primero."
                return
            with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
                tmp.write(subido.getvalue())
                ruta = tmp.name
            nombre = os.path.splitext(subido.name)[0]
        else:
            nombre = os.path.splitext(os.path.basename(ruta))[0]
        try:
            politica = AgenteQ.cargar(ruta, np.random.default_rng(semilla + 1))
        except Exception as e:
            st.session_state.error = f"No se pudo cargar el agente: {e}"
            return
        if politica.Q.shape[0] != N_ESTADOS:
            st.session_state.error = (f"Este agente tiene {politica.Q.shape[0]} estados pero el abstraccion.py actual "
                                      f"define {N_ESTADOS}: fue entrenado con otra versión de la abstracción.")
            return
        politica.epsilon = 0.0
        politica.tau_eval = float(tau)
    st.session_state.partida = PartidaContraBot(politica, nombre_bot=nombre, fichas_iniciales=int(fichas),
                                                reiniciar_fichas=reiniciar, semilla=semilla)
    st.session_state.rival = nombre


def _ejecutar(fn, *args) -> None:
    try:
        fn(*args)
        st.session_state.pop("error", None)
    except ValueError as e:
        st.session_state.error = str(e)


# ----------------------------------------------------------------------------
# barra lateral
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Rival")
    disponibles = agentes_disponibles()
    opciones = list(disponibles) + [OPCION_UNIFORME, OPCION_SUBIR]
    fuente_sel = st.selectbox("Agente", opciones, index=len(disponibles) - 1 if disponibles else 0)
    subido = st.file_uploader("Archivo del agente (.npz)", type="npz") if fuente_sel == OPCION_SUBIR else None
    es_agente = fuente_sel not in (OPCION_UNIFORME,)
    tau = st.slider("Variabilidad del agente", 0.0, 1.0, 0.0, 0.05, disabled=not es_agente,
                    help="0 = siempre la mejor acción según su tabla Q (determinista, más explotable). "
                         "Más alto = elige acciones menos buenas con más frecuencia.")
    with st.expander("Reglas"):
        fichas_ini = st.number_input("Fichas iniciales", 200, 100_000, 1000, step=100,
                                     help="Los agentes se entrenaron con 1000 fichas y ciegas 10/20.")
        reiniciar = st.checkbox("Reiniciar fichas cada mano", value=False,
                                help="Apagado: partida completa hasta que alguien quiebre. "
                                     "Encendido: cada mano empieza con las fichas iniciales (como en el entrenamiento).")
        st.caption("Ciegas 10 / 20.")
    if st.button("Nueva partida", type="primary", use_container_width=True):
        ruta = disponibles.get(fuente_sel)
        crear_partida(fuente_sel, ruta, subido, tau, fichas_ini, reiniciar)

    p_side = st.session_state.get("partida")
    if p_side is not None:
        st.divider()
        st.subheader("Marcador")
        vs = p_side.vista()
        signo = "+" if vs["neto_total"] >= 0 else ""
        st.metric("Tu balance (fichas)", f'{signo}{vs["neto_total"]}')
        c1, c2 = st.columns(2)
        c1.metric("Manos", vs["manos"])
        c2.metric("bb/mano", f'{vs["bb_por_mano"]:+.2f}')

# Primera carga: arranca con el último agente disponible (o el uniforme)
if "partida" not in st.session_state and "error" not in st.session_state:
    ruta0 = disponibles.get(fuente_sel)
    crear_partida(fuente_sel, ruta0, subido, tau, fichas_ini, reiniciar)

# ----------------------------------------------------------------------------
# pantalla principal
# ----------------------------------------------------------------------------
st.title("Poker 1 vs 1")
if "error" in st.session_state:
    st.error(st.session_state.error)

partida: PartidaContraBot | None = st.session_state.get("partida")
if partida is None:
    st.info("Elige un rival en la barra lateral y pulsa **Nueva partida**.")
    st.stop()

v = partida.vista()
st.caption(f"Rival: **{st.session_state.get('rival', '')}** · ciegas 10/20")
st.markdown(mesa_html(v), unsafe_allow_html=True)

if v["terminada"]:
    r = v["resultado"]
    if r["empate"]:
        st.info("🤝 Empate: el pozo se reparte.")
    elif r["gana_humano"]:
        st.success(f"🎉 Ganas la mano: **+{r['neto_humano']}** fichas" + (" (el rival se retiró)" if r["por_retiro"] else ""))
    else:
        st.error(f"Pierdes la mano: **{r['neto_humano']}** fichas" + (" (te retiraste)" if r["por_retiro"] else ""))
    if v["partida_terminada"]:
        if v["campeon"] == v["nombres"][0]:
            st.balloons()
            st.success("🏆 ¡Ganaste la partida!")
        else:
            st.error(f"🏆 {v['campeon']} gana la partida. Pulsa **Nueva partida** para la revancha.")
    else:
        st.button("Siguiente mano ➔", type="primary", on_click=_ejecutar, args=(partida.nueva_mano,))
elif v["turno_humano"]:
    acciones = partida.acciones_humano()
    cols = st.columns(len(acciones))
    for col, a in zip(cols, acciones):
        col.button(a["etiqueta"], key=f"acc_{a['id']}", use_container_width=True,
                   type="primary" if a["id"] == 1 else "secondary",
                   on_click=_ejecutar, args=(partida.actuar, a["id"]))
    rango = partida.rango_subida()
    if rango:
        with st.expander("Subida personalizada"):
            lo, hi = rango
            monto = st.number_input("Fichas a subir por encima de igualar", min_value=lo, max_value=hi,
                                    value=lo, step=10, key=f"monto_{v['manos']}_{v['fase']}_{v['pozo']}")
            st.button("Subir", on_click=_ejecutar, args=(partida.subir, int(monto)))

with st.expander("Historial de la mano"):
    for linea in v["historial"] or ["(sin acciones todavía)"]:
        st.text(linea)
