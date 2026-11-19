#Para que funcione en Wine
import os
os.environ["KMP_AFFINITY"] = "disabled"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#Librerías
import sys
import csv
import shutil
import json  #Lee y escribe el historial de visitas.

import numpy as np
from collections import deque
import ctypes
import time

import torch
import torch.nn as nn
import torch.optim as optim
import random

from capturar_juego import GeometryDashCapture
from gd_lector_estado import GDStateReader



#----------------------------------------------------------#
#                     Apretar botones                      #
#----------------------------------------------------------#

#Código para apretar botones (hacer que salte)
#Utiliza la barra espaciadora para saltar o mantener el vuelo

PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort), ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong), ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

#La unión conserva MouseInput para respetar el tamaño de INPUT de Windows
VK_SPACE = 0x20
KEYEVENTF_KEYUP = 0x0002

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long), ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong), ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong), ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput), ("mi", MouseInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("ii", Input_I)]

def press_key():
    #Simula mantener presionada la barra espaciadora
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(VK_SPACE, 0, 0, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def release_key():
    #Simula soltar la barra espaciadora
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(VK_SPACE, 0, KEYEVENTF_KEYUP, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def detener_con_f8():  #Consulta F8 aunque la ventana activa sea el juego.
    estado = ctypes.windll.user32.GetAsyncKeyState(0x77)  #0x77 es el código de F8.
    return bool(estado & 0x8001)  #Detecta si está presionada o hubo una pulsación reciente.

#Configuración de los checkpoints; usar práctica con autocheckpoints desactivados.
MIN_VISITAS_CHECKPOINT = 100  #Llegadas necesarias para autorizar un porcentaje.
VISITAS_FILE = "visitas_totales.json"  #Historial asociado a los modelos guardados.

def colocar_checkpoint():  #Pulsa y suelta Z sin modificar la barra espaciadora.
    extra = ctypes.c_ulong(0)  #Dato auxiliar requerido por la estructura de Windows.
    eventos = (Input * 2)()  #Prepara los dos eventos de teclado.
    for i, flags in enumerate((0, KEYEVENTF_KEYUP)):  #Primero presiona y después suelta.
        eventos[i].type = 1  #Indica que el evento corresponde al teclado.
        eventos[i].ii.ki = KeyBdInput(0x5A, 0, flags, 0, ctypes.pointer(extra))  #Z.
    enviados = ctypes.windll.user32.SendInput(2, eventos, ctypes.sizeof(Input))  #Envía ambos eventos.
    if enviados != 2:  #No registra un checkpoint si Windows rechaza el envío.
        raise RuntimeError("No se pudo enviar la pulsación completa de Z.")  #Detiene para revisar.



#----------------------------------------------------------#
#                   Función recompensa                     #
#----------------------------------------------------------#

# Evita premiar un salto artificial del porcentaje durante una reaparición tardía.
# Con 60 FPS, avanzar más de 2 puntos porcentuales en una lectura es inusual.
MAX_AVANCE_POR_PASO = 2.0


def calcular_reward(estado_actual, estado_anterior):
    # La muerte se detecta por el aumento del contador de intentos.
    muerte = estado_actual["attempts"] > estado_anterior["attempts"]
    if muerte:
        return -2.0, True  # Nunca se suma progreso en el mismo paso de la muerte.

    # Recompensa continua: +1 por cada 3 puntos porcentuales REALMENTE avanzados.
    progreso = estado_actual["percent"] - estado_anterior["percent"]
    if progreso <= 0 or progreso > MAX_AVANCE_POR_PASO:
        return 0.0, False  # Retroceso o salto espurio asociado a la reaparición.

    return progreso / 3.0, False



#----------------------------------------------------------#
#             Obtener capturas y tomar acción              #
#----------------------------------------------------------#

class GeometryDashEnv:
    def __init__(self):

        self.visitas_totales = [0] * 101  #Un contador para cada porcentaje entero.
        self.visitas_epoca = [False] * 101  #Evita contar dos veces en un mismo intento.
        self.ultimo_checkpoint = 0  #La partida actual comienza sin checkpoints del código.
        if os.path.exists(VISITAS_FILE):  #Recupera las visitas si hay un guardado anterior.
            with open(VISITAS_FILE, "r", encoding="utf-8") as f:  #Abre el historial.
                visitas = json.load(f)  #Convierte el JSON en una lista de Python.
            if not isinstance(visitas, list) or len(visitas) != 101:  #Comprueba su tamaño.
                raise ValueError("El historial debe tener 101 contadores de visitas.")  #Evita cargar datos inválidos.
            if any(type(v) is not int or v < 0 for v in visitas):  #Comprueba los contadores.
                raise ValueError("Las visitas deben ser enteros no negativos.")  #Evita cargar datos inválidos.
            self.visitas_totales = visitas  #Continúa desde el historial guardado.

        self.max_percent = 0.0
        self.primer_paso_tras_reset = False  # Evita recompensar una reaparición no sincronizada.

        #Inicializa recolección de datos con los códigos del Fraik
        self.screen_capture = GeometryDashCapture(image_size=(84, 84))
        self.state_reader = GDStateReader()

        self.frame_stack = deque(maxlen=4)
        self.intentos_anteriores = 0
        self.porcentaje_anterior = 0.0

    def step(self, accion):
        #Hacer accion
        if accion == 1:
            press_key()
        else:
            release_key()

        #Delay para que el juego procese el frame (sync con fps del juego)
        time.sleep(1/60)

        #Captura screenshot y lo transforma a 84x84 en escala de grises
        current_frame = self.screen_capture.capture()
        self.frame_stack.append(current_frame)

        #Lee el estado: percent, attempts, vehicle, and is_upside_down
        estado = self.state_reader.read()

        #Diccionario con el estado anterior para calcular el reward
        prev_estado = {
            "attempts": self.intentos_anteriores,
            "percent": self.porcentaje_anterior
        }

        #Actualiza el porcentaje máximo alcanzado
        self.max_percent = max(self.max_percent, estado["percent"])

        # Calcular recompensa usando SOLO el avance desde la observación anterior.
        reward, done = calcular_reward(estado, prev_estado)

        # La primera lectura tras reset puede capturar la reaparición, no un avance real.
        # Si hubo muerte, se conserva siempre la penalización de -2.
        if self.primer_paso_tras_reset and not done:
            reward = 0.0
        self.primer_paso_tras_reset = False

        if not done and reward > 0:
            # Cuenta solo porcentajes efectivamente cruzados, no los anteriores al checkpoint.
            self.actualizar_checkpoints(prev_estado["percent"], estado["percent"])
        # Si no hubo avance, hubo muerte o apareció un salto espurio, no se registran visitas.

        #Actualizar estados
        self.intentos_anteriores = estado["attempts"]
        self.porcentaje_anterior = estado["percent"]

        #Formatear estado como numpy array (4, 84, 84)
        state = np.array(self.frame_stack)
        info = estado

        return state, reward, done, False, info

    def actualizar_checkpoints(self, porcentaje_anterior, porcentaje):
        # Registra solo umbrales enteros cruzados EN ESTE INTENTO (no los previos al spawn).
        anterior = max(0, min(100, int(porcentaje_anterior)))
        actual = max(0, min(100, int(porcentaje)))
        for p in range(anterior + 1, actual + 1):
            if not self.visitas_epoca[p]:  # Cada porcentaje se cuenta una vez por intento.
                self.visitas_epoca[p] = True
                self.visitas_totales[p] += 1
        autorizado = max(  #Busca el porcentaje más avanzado que cumple el umbral.
            (p for p in range(1, 100) if self.visitas_totales[p] >= MIN_VISITAS_CHECKPOINT),  #Excluye el final del nivel.
            default=0,  #Cero significa que todavía no hay autorización.
        )
        if self.ultimo_checkpoint < autorizado and self.porcentaje_anterior < autorizado <= porcentaje < 100:  #Exige avance, cruce actual y nivel sin terminar.
            colocar_checkpoint()  #Solicita al juego guardar el checkpoint con Z.
            self.ultimo_checkpoint = autorizado  #Evita repetirlo o retroceder en esta partida.
            print(f"Checkpoint solicitado al cruzar {autorizado}% ({porcentaje:.2f}% observado).")  #Informa la posición observada.

    def guardar_visitas(self):  #Guarda el historial junto con las redes.
        with open(VISITAS_FILE + ".tmp", "w", encoding="utf-8") as f:  #Escribe primero un archivo temporal.
            json.dump(self.visitas_totales, f)  #Serializa los 101 contadores.
        os.replace(VISITAS_FILE + ".tmp", VISITAS_FILE)  #Reemplaza el historial cuando terminó de escribir.

    def reset(self):

        release_key()

        #Reset estado
        estado = self.state_reader.read()
        self.intentos_anteriores = estado["attempts"]
        self.porcentaje_anterior = estado["percent"]  # Baseline real del reset, incluso con checkpoint.
        self.primer_paso_tras_reset = True  # Protege la primera lectura si aún está reapareciendo.

        self.visitas_epoca = [False] * 101  # Solo cuentan los nuevos umbrales cruzados.

        #Llenar frame inicial
        initial_frame = self.screen_capture.capture()
        for _ in range(4):
            self.frame_stack.append(initial_frame)

        return np.array(self.frame_stack), estado

    def close(self):
        release_key()
        #Liberar memoria
        self.screen_capture.close()
        self.state_reader.close()



#----------------------------------------------------------#
#                        Agentes                           #
#----------------------------------------------------------#

class DQN(nn.Module):
    def __init__(self, input_shape=(4, 84, 84), num_actions=2):
        super(DQN, self).__init__()

        #Capas para procesar los frames
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        #Forma del input
        conv_out_size = self._get_conv_out(input_shape)

        #Conecta capas
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        #Escala valor de pixeles de 0-255 a 0.0-1.0
        x = x.float() / 255.0
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)

#-------------------------------------------------------------------#

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Cubo
cubo_net = DQN().to(device)
cubo_target_net = DQN().to(device)
cubo_target_net.load_state_dict(cubo_net.state_dict())
cubo_optimizer = optim.Adam(cubo_net.parameters(), lr=1e-4)
cubo_replay_buffer = deque(maxlen=50000)

#Nave
nave_net = DQN().to(device)
nave_target_net = DQN().to(device)
nave_target_net.load_state_dict(nave_net.state_dict())
nave_optimizer = optim.Adam(nave_net.parameters(), lr=1e-4)
nave_replay_buffer = deque(maxlen=50000)

#ID del vehiculo obtenido por el mod
VEHICLE_CUBO = 0
VEHICLE_NAVE = 1


#-------------------------------------------------------------------#

#Importar entrenamiento previo

print("\nBuscando entrenamiento previo...\n")
if os.path.exists("cubo_model.pth"):
    cubo_net.load_state_dict(torch.load("cubo_model.pth"))
    cubo_target_net.load_state_dict(cubo_net.state_dict())
    print("Modelo del cubo cargado exitosamente.")

if os.path.exists("nave_model.pth"):
    nave_net.load_state_dict(torch.load("nave_model.pth"))
    nave_target_net.load_state_dict(nave_net.state_dict())
    print("Modelo de la nave cargado exitosamente.")

#-------------------------------------------------------------------#

#Elección epsilon-greedy basado en tipo de vehículo
def elegir_accion(estado, vehicle_id, epsilon):

    if random.random() < epsilon:
        return random.randint(0, 1) #Acción random

    estado_tensor = torch.FloatTensor(estado).unsqueeze(0).to(device)

    with torch.no_grad():
        if vehicle_id == VEHICLE_CUBO:
            q_values = cubo_net(estado_tensor)
        elif vehicle_id == VEHICLE_NAVE:
            q_values = nave_net(estado_tensor)
        else:
            #Usar cubo si no se sabe cuál es
            q_values = cubo_net(estado_tensor)

    return q_values.max(1)[1].item() #Retorna acción con mayor Q-value



#----------------------------------------------------------#
#                           DDQL                           #
#----------------------------------------------------------#

BATCH_SIZE = 32
GAMMA = 0.99  # Learning rate, mismo que el paper

def opti_model(primary_net, target_net, optimizer, replay_buffer):

    #Verifica que se tienen suficientes samples
    if len(replay_buffer) < BATCH_SIZE:
        return

    #Muestrea los frames de manera uniforme
    batch = random.sample(replay_buffer, BATCH_SIZE)

    #Batch con formato PyTorch
    states = torch.FloatTensor(np.array([t[0] for t in batch])).to(device)
    actions = torch.LongTensor([t[1] for t in batch]).view(-1, 1).to(device)
    rewards = torch.FloatTensor([t[2] for t in batch]).to(device)
    next_states = torch.FloatTensor(np.array([t[3] for t in batch])).to(device)
    dones = torch.FloatTensor([t[4] for t in batch]).to(device)

    #Calcula Q(s, a, \theta)
    current_q_values = primary_net(states).gather(1, actions).squeeze(1)

    #DDQL para Y
    #La red primaria calcula \max Q^*(s', a') eligiendo "la mejor acción"
    next_actions = primary_net(next_states).max(1)[1].view(-1, 1)

    #Red objetivo evalúa la acción elegida
    with torch.no_grad():
        next_q_values = target_net(next_states).gather(1, next_actions).squeeze(1)

    # Y = r(a,s) + \eta * Q_target(s', a')
    #si murió (done == 1), la esperanza de la recompenza futura es 0
    expected_q_values = rewards + (GAMMA * next_q_values * (1 - dones))

    #Calcula pérdida J(\theta)
    loss = nn.MSELoss()(current_q_values, expected_q_values)

    #Optimizar
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#----------------------------------------------------------#
#                      Training Loop                       #
#----------------------------------------------------------#

TARGET_UPDATE = 500 #Actualizar red objetivo y guardar cada 500 pasos totales
BACKUP_DIR = "backups" #Copias históricas: paso_500, paso_1000, ...
NUM_EPISODES = 50000

#Cargar pasos si existen
global_steps_nave = 0
global_steps_cubo = 0
global_max_percent = 0
global_epocas = 0

if os.path.exists("steps_nave.txt"):
    with open("steps_nave.txt", "r") as f:
        global_steps_nave = int(f.read())
    print(f"\nRetomando entrenamiento desde el paso {global_steps_nave}")
if os.path.exists("steps_cubo.txt"):
    with open("steps_cubo.txt", "r") as f:
        global_steps_cubo = int(f.read())
    print(f"\nRetomando entrenamiento desde el paso {global_steps_cubo}")
if os.path.exists("epocas.txt"):
    with open("epocas.txt", "r") as f:
        global_epocas = int(f.read())
    print(f"Retomando entrenamiento desde la época {global_epocas}")
if os.path.exists("max_perc.txt"):
    with open("max_perc.txt", "r") as f:
        global_max_percent = float(f.read())
    print(f"Porcentaje máximo alcanzado {global_max_percent}\n")



def train_agents(env):
    state, info = env.reset()
    vehicle = info["vehicle"]

    #Usar los pasos cargados (=0 si no hay)
    steps_done_nave = global_steps_nave
    steps_done_cubo = global_steps_cubo
    saved_epocas = global_epocas
    this_percent = 0

    #Cargar el max_percent guardado en el env
    env.max_percent = global_max_percent

    with open('datos_entrenamiento.csv', mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        csv_buffer = []

        for epoca in range(NUM_EPISODES):
            while True:

                #Calcula epsilon (exploración-explotación tradeoff) para cada vehículo
                if vehicle == VEHICLE_NAVE:
                    epsilon = 0
                else:
                    epsilon = 0


                #Elige acción basado en el vehículo activo
                accion = elegir_accion(state, vehicle, epsilon)
                
                #Hace un paso
                next_state, reward, done, truncated, next_info = env.step(accion)
                next_vehicle = next_info["vehicle"]

                detener = detener_con_f8()  #Revisa la solicitud de salida después del paso.
                if detener:  #Si detectó F8, comienza la detención.
                    release_key()  #Suelta espacio antes de entrenar y guardar.
                
                if done == False:
                    this_percent = next_info["percent"]


                #Guarda la transicion en el buffer correcto
                transicion = (state, accion, reward, next_state, done)
                if vehicle == VEHICLE_CUBO:
                    cubo_replay_buffer.append(transicion)
                    opti_model(cubo_net, cubo_target_net, cubo_optimizer, cubo_replay_buffer)
                elif vehicle == VEHICLE_NAVE:
                    nave_replay_buffer.append(transicion)
                    opti_model(nave_net, nave_target_net, nave_optimizer, nave_replay_buffer)


                #Dio un paso
                if vehicle == VEHICLE_NAVE:
                    steps_done_nave += 1
                else:
                    steps_done_cubo += 1

                steps_done = steps_done_cubo + steps_done_nave


                #Siguiente estado
                state = next_state
                vehicle = next_vehicle

                epoc = epoca+saved_epocas

                #Actualización de la red objetivo cada C=TARGET_UPDATE pasos
                if steps_done % TARGET_UPDATE == 0 or detener:  #Guarda cada 500 pasos o al detectar F8.
                    cubo_target_net.load_state_dict(cubo_net.state_dict())
                    nave_target_net.load_state_dict(nave_net.state_dict())

                    #Guarda redes al disco
                    torch.save(cubo_net.state_dict(), "cubo_model.pth")
                    torch.save(nave_net.state_dict(), "nave_model.pth")

                    # Guardar los contadores de pasos
                    with open("steps_cubo.txt", "w") as f:
                        f.write(str(steps_done_cubo))
                    with open("steps_nave.txt", "w") as f:
                        f.write(str(steps_done_nave))
                    # Guardar % máximo alcnazado
                    with open("max_perc.txt", "w") as f:
                        f.write(str(env.max_percent))
                    # Guardar las épocas realizadas
                    with open("epocas.txt", "w") as f:
                        f.write(str(epoc))
                    # Guardar datos de entrenamiento
                    for row in csv_buffer:
                        writer.writerow(row)
                    csv_buffer.clear()
                    file.flush()

                    env.guardar_visitas()  #Guarda las visitas correspondientes a estos pesos y contadores.

                    #Conservar una copia de cada guardado, sin reemplazar los anteriores.
                    backup_path = os.path.join(BACKUP_DIR, f"paso_{steps_done}")
                    os.makedirs(backup_path, exist_ok=True)
                    for filename in (
                        "cubo_model.pth", "nave_model.pth",
                        "steps_cubo.txt", "steps_nave.txt",
                        "max_perc.txt", "epocas.txt", "datos_entrenamiento.csv",
                        VISITAS_FILE,  #Incluye el historial en cada copia de 500 pasos.
                    ):
                        shutil.copy2(filename, os.path.join(backup_path, filename))

                    print(f"\n---Red actualizada en paso {steps_done} | Copia: {backup_path}---")

                if detener:  #Sale después de completar el guardado.
                    print("Entrenamiento detenido con F8; estado guardado.")  #Confirma la salida.
                    return  #Termina train_agents y ejecuta el finally que cierra el entorno.
                if done:
                    print(f"\nÉpoca {epoc} | Pasos: {steps_done} | Epsilon: {epsilon:.3f} | Porcentaje alcanzado: {this_percent:.3f} % |  Porcentaje máximo   : {env.max_percent:.3f} %")
                    csv_buffer.append([epoc, steps_done, epsilon, this_percent, env.max_percent])

                    state, info = env.reset()
                    vehicle = info["vehicle"]
                    break



#----------------------------------------------------------#
#                        Ejecutar                          #
#----------------------------------------------------------#


if __name__ == "__main__":

    print("\nIniciando entorno...\n")
    env = GeometryDashEnv()

    print("Usa modo práctica, desactiva los autocheckpoints y comienza sin checkpoints previos.")  #Preparación necesaria para que el código controle Z.
    print("¡Cambia a la ventana de Geometry Dash ahora!")
    for i in range(5, 0, -1):
        print(f"Comienza en {i}...")
        time.sleep(1)

    try:
        print("\nIniciando entrenamiento...")
        train_agents(env)
    except KeyboardInterrupt:
        print("\nEntrenamiento interrumpido por el usuario.")
    finally:
        env.close()
        print("Entorno cerrado y memoria liberada.")
