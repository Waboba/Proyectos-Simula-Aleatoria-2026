# Azul: Vitrales de Sintra — Aprendizaje por Refuerzo Híbrido

Agente de Inteligencia Artificial para el juego de mesa **Azul: Vitrales de Sintra**, desarrollado mediante **Aprendizaje por Refuerzo Profundo con Enmascaramiento de Acciones (`MaskablePPO`)**, **recompensa telescópica exacta** y **búsqueda *lookahead* acotada a 1 paso ($K=6$)**.

> **Proyecto de Inteligencia Artificial · Noviembre de 2026**  
> **Autoras:** Josefina Gaete & Fran Loyola

---

## Demo Visual Interactivo (D3.js)

https://github.com/josefinagaetep/Proyectos-Simula-Aleatoria-2026/raw/main/iav4%20vs%20heuristica.mp4

El proyecto incluye un visualizador web en **D3.js** (`visualizador_azul.html`) que reconstruye partidas completas a partir de los historiales `.json` generados por los scripts, permitiendo inspeccionar jugada a jugada las fábricas, los vitrales de 8 columnas, la posición del vidriero, el track de cristal roto y la evolución del marcador.

---

## Sobre el Juego: *Azul: Vitrales de Sintra*

**Azul: Vitrales de Sintra** es un juego de estrategia secuencial por turnos para 2 jugadores, de **suma cero** y con información estocástica (robo aleatorio desde una bolsa de 100 fichas en 5 colores: *Azul, Amarillo, Rosado, Transparente y Naranjo*):

- **Objetivo y Duración:** La partida dura **6 rondas**. Cada jugador gestiona un tablero individual compuesto por **8 tiras verticales de vitral de doble cara** (columnas `C1` a `C8`, de 5 espacios cada una).
- **Turno de Juego (Opción A / Opción B):**
  - **Opción A (Tomar y colocar):** El jugador elige un color de una de las **5 fábricas** (que inician cada ronda con 4 fichas al azar) o del **centro de la mesa**, toma *todas* las fichas de ese color (moviendo las sobrantes de la fábrica al centro) y las coloca en una tira situada **debajo o a la derecha de su peón vidriero**.
  - **Opción B (Reiniciar vidriero):** Si no conviene o no es posible avanzar hacia la derecha, el jugador puede gastar su turno para devolver el vidriero al extremo izquierdo del tablero.
- **Puntuación en Cascada, Color de Moda y Ornamentos:** Al llenar los 5 espacios de una tira, esta se puntúa sumando su valor base (`[4, 3, 3, 2, 2, 1, 1, 2]`), el puntaje de todas las tiras ya completadas a su derecha (**cascada**), $+1\text{ pt}$ por cada ficha del **color favorito de la ronda** y bonos de hasta $+10\text{ pts}$ por **ornamentos adyacentes**. La primera vez que se completa una tira se voltea a su cara B; la segunda vez se retira del tablero.
- **Track de Cristal Roto:** Las fichas tomadas que no quepan en la tira elegida (más la ficha de primer jugador del centro) avanzan en una escala creciente de penalización (`0, -1, -2, -4, -5, -8, -10, -12, -15, -18`). Si se desborda la casilla `#10`, se descuentan **18 puntos inmediatos** y el marcador reinicia en `#1`.
- **Complejidad Combinatoria:** El juego posee $\approx 4{,}79 \times 10^{16}$ configuraciones iniciales posibles, $16.108.764$ ($\approx 1{,}61 \times 10^7$) ofertas distintas en fábricas por ronda y un espacio de hasta **1.621 acciones discretas** por turno.

---

## Evolución de Versiones (`IA v1` $\rightarrow$ `IA v4`)

El desarrollo siguió una **cadena evolutiva curricular** en la que cada versión debía superar el **$>60\%$ de victorias** frente a su predecesora antes de avanzar de etapa:

| Versión | Checkpoint | Arquitectura y Entrenamiento | Hito Principal | Victorias vs. Heurística | Rating Elo |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **Aleatorio** | — | Selección uniforme sobre máscara legal | Línea base mínima | — | `1000` |
| **`IA v1`** | `ppo_azul_sintra_v1.zip` | Estado `253D` · `MaskablePPO` (`200k` pasos) | Política base; vence **98,0 %** al azar | `2,0 %` | `1623` |
| **`IA v2`** | `ppo_azul_sintra_v2.zip` | Estado `288D` · `MaskablePPO` (`500k` pasos) + currículo mixto | Incorpora tablero rival completo y conteo de bolsa/descarte; vence **68,0 %** a `IA v1` | `8,0 %` | `1813` |
| **`IA v3`** | `ppo_azul_sintra_v3.zip` | *Behavioral Cloning* (`30k`) + `MaskablePPO` (`400k` pasos) | Preentrenamiento supervisado sobre jugadas expertas + RL; vence **68,0 %** a `IA v2` | `14,0 %` | `1998` |
| **Heurística** | *Reglas tácticas* | Simulación 1 paso orientada a ganancia inmediata y bajo roto | Rival de referencia experto | `50,0 %` | `2284` |
| **`IA v4` ($K=1$)** | `ppo_azul_sintra_v4.zip` | *Transfer Learning* desde `v3` (`lr=8e-5`, `400k` pasos) + **Recompensa Telescópica** | Política neuronal pura entrenada con currículo (`50% Heur. / 30% v3 / 15% v2 / 5% Azar`) | `23,0 %` | — |
| **`IA v4` ($K=6$)** | `ppo_azul_sintra_v4.zip` | **`IA v4` + Búsqueda Híbrida *Lookahead* Top-$K$** | Supera a la Heurística (**56,0 %**), vence **82,0 %** a `IA v3` y logra **76,5 %** global | **`56,0 %`** | **`2326*`** |

*\*Elo efectivo estimado mediante el modelo Bradley-Terry a partir de torneos balanceados de $N = 100$ partidas por rival.*

---

## Cómo se Utilizan las Versiones (Resumen Unificado)

**Las versiones `v2`, `v3` y `v4` comparten exactamente la misma interfaz de entrada/salida:** todas emplean el vector de observación de **288 dimensiones (`288D`)**, el espacio de **1.621 acciones discretas** con enmascaramiento legal y la clase cargadora `AgenteHibridoV4` (`src/env_azul_v4.py`).

- **`IA v4` (por defecto):** carga `models_checkpoint/ppo_azul_sintra_v4.zip` con `--k 6` (activa el *lookahead* híbrido a 1 paso, $\approx 4\text{ ms}$ por jugada).
- **`IA v2` e `IA v3`:** pueden ejecutarse con los mismos scripts pasando su `.zip` en `--modelo` y fijando `--k 1` (o mediante los flags `--v3` y `--v2`).
- **`IA v1`:** emplea la representación inicial de `253D` (previa a la ampliación a `288D`); los scripts `contra_maquina_ia_v4.py` (`--v1`) y `evaluar_ia_v4.py` detectan y adaptan su dimensión automáticamente.

---

## Guía de Uso

Todos los comandos deben ejecutarse desde la **raíz del proyecto** (`Proyectos-Simula-Aleatoria-2026/`).

### 1. Jugar contra `IA v4` en Consola

Por defecto, `contra_maquina_ia_v4.py` carga `models_checkpoint/ppo_azul_sintra_v4.zip` con búsqueda híbrida **$K = 6$**:

```bash
# Partida estándar contra IA v4 (K=6) indicando tu nombre
python -m scripts.contra_maquina_ia_v4 Ana

# Partida con nombre y semilla reproducible
python -m scripts.contra_maquina_ia_v4 Ana 7

# Elegir el nivel de búsqueda Lookahead (--k 1 = modo reactivo puro, --k 12 = búsqueda amplia)
python -m scripts.contra_maquina_ia_v4 Ana --k 1
python -m scripts.contra_maquina_ia_v4 Ana --k 12

# Hacer que IA v4 juegue el primer turno
python -m scripts.contra_maquina_ia_v4 Ana --ia-empieza
```

También puedes enfrentar desde el mismo script a las versiones anteriores y líneas base:

```bash
python -m scripts.contra_maquina_ia_v4 Ana --v3
python -m scripts.contra_maquina_ia_v4 Ana --v2
python -m scripts.contra_maquina_ia_v4 Ana --v1
python -m scripts.contra_maquina_ia_v4 Ana --heuristica
python -m scripts.contra_maquina_ia_v4 Ana --azar
```

Si deseas especificar manualmente la ruta del checkpoint:

```bash
python -m scripts.contra_maquina_ia_v4 Ana --modelo models_checkpoint/ppo_azul_sintra_v4.zip --k 6
```

---

### 2. Exportar Partidas para el Visualizador Web (`D3.js`)

Simula una partida completa entre `IA v4` ($K=6$) y la Heurística Táctica guardando el historial turno a turno en JSON:

```bash
python -m scripts.contra_maquina_ia_v4 --demo-heuristica --json partida_historial.json
```

Luego abre **`visualizador_azul.html`** en tu navegador y carga el archivo **`partida_historial.json`** generado.

---

### 3. Evaluación Experimental (Benchmark Balanceado)

El script `evaluar_ia_v4.py` ejecuta series de $N$ partidas deterministas por rival alternando estrictamente el asiento inicial (`50%` jugador 0 / `50%` jugador 1) para eliminar la ventaja de salida:

```bash
python -m scripts.evaluar_ia_v4 --partidas 100 --k 6
```

---

### 4. Reentrenamiento de `IA v4` (*Transfer Learning*)

Para reentrenar `IA v4` desde `models_checkpoint/ppo_azul_sintra_v3.zip` sobre el entorno telescópico `AzulSintraEnvV4`:

```bash
python -m scripts.train_rl_v4 --timesteps 400000 --n-envs 4 --lr 8e-5
```

---

## Requisitos y Estructura del Repositorio

### Instalación de Dependencias

Se recomienda Python **3.11 – 3.13** (por ejemplo, en un entorno Conda):

```bash
pip install numpy torch gymnasium pettingzoo stable-baselines3 sb3-contrib
```

> **Nota en Windows / Conda:** Si el comando `python` apunta a una instalación global distinta, activa tu entorno (`conda activate grafica`) o ejecuta usando la ruta directa al intérprete del entorno:
> ```powershell
> & "C:\Users\<USUARIO>\anaconda3\envs\grafica\python.exe" -m scripts.contra_maquina_ia_v4 Ana
> ```


disclaimer: los códigos fueron generados con IA, por optimización de tiempo y uso de librerías convenientes
