"""
Agente de Q-learning tabular mejorado.

    Q[estado, accion]  con  estado = abstraccion.estado_de_obs(obs)
                            accion in 0..4 (ver motor.NOMBRES_ACCION)

Mejoras respecto a la versión base:
  - Tasa de aprendizaje adaptativa por par (estado, acción) según el número de visitas,
    acotada inferiormente por `alpha_min` (evita olvidar jugadas raras por ruido).
  - Soporte para desempate/evaluación estocástica mediante temperatura Softmax (`tau_eval`),
    lo que produce políticas mixtas más difíciles de explotar en póker heads-up.
"""
import hashlib
import json

import numpy as np

from .abstraccion import N_ESTADOS, estado_de_obs
from .motor import N_ACCIONES

#Definimos del agente Q
class AgenteQ:
    usa_obs = True
#Se definen los parametros usuales del Q-learning, ver glosario para la definición.
    def __init__(
        self,
        nombre: str = "AgenteQ",
        alpha: float = 0.25,
        gamma: float = 1.0,
        epsilon: float = 1.0,
        rng: np.random.Generator | None = None,
        alpha_min: float = 0.015,
        decaimiento_alpha: float = 0.02,
        tau_eval: float = 0.0,
    ) -> None:
        self.nombre = nombre
        self.alpha = alpha
        self.alpha_min = alpha_min
        self.decaimiento_alpha = decaimiento_alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.tau_eval = tau_eval
        self.rng = rng if rng is not None else np.random.default_rng()
        self.Q = np.zeros((N_ESTADOS, N_ACCIONES), dtype=np.float32)
        self.visitas = np.zeros((N_ESTADOS, N_ACCIONES), dtype=np.int64)
        self.episodios = 0

    #Definimos una función que calcule la mejor acción
    def _mejor_accion(self, estado: int, mascara: np.ndarray, tau: float = 0.0) -> int:
        #Extraemos el estado en que estamos
        q = self.Q[estado] 
        #Verificamos si estamos con temperatura de entropía, en caso que si:
        if tau > 1e-6:
            #Extraemos los indices de las acciones legales
            legales = np.flatnonzero(mascara)
            #Si hay exactamente una acción legal
            if len(legales) == 1:
                #Entonces devolvemos el número asociado a la acción
                return int(legales[0])
            #Si hay más de una, filtramos mediante indexación por lista y creamos un vector para construir la fórmula de probabilidad softmax
            #Particularmente, se utiliza la diferencia para no explotar memoria y no afectan a la implementanción
            logits = (q[legales] - q[legales].max()) / tau
            #Calculamos las exponenciales y las clipeamos para no tener outlayers demasiado grandes
            probs = np.exp(np.clip(logits, -30.0, 0.0))
            #Normalizamos
            probs /= probs.sum()
            #Luego, en base a esas probabilidades escogemes una acción legal aleatoria
            return int(legales[int(self.rng.choice(len(legales), p=probs))])
        #Tomamos todas las acciones ilegales en q y les ponemos -infinito
        q_enmascarado = np.where(mascara, q, -np.inf)
        #Buscamos el valor q más alto, si hay solo una ganadora devolvemos esa y si hay empate se elige uniforme.
        mejores = np.flatnonzero(q_enmascarado == q_enmascarado.max())
        return int(mejores[0]) if len(mejores) == 1 else int(mejores[int(self.rng.integers(len(mejores)))])

    #Encargado de escoger a partir de la observación que hacer    
    def elegir(self, obs: np.ndarray, mascara: np.ndarray, explorar: bool = False) -> tuple[int, int]:
        """(estado, accion). Con explorar=True usa epsilon-greedy."""
        #A partir de la observación extraemos el indice de la Q-table
        estado = estado_de_obs(obs)
        #Si el agente puede explorar y el número aleatorio es menor a epsilon entonces elegimos al azar entre las legales
        if explorar and self.rng.random() < self.epsilon:
            legales = np.flatnonzero(mascara)
            return estado, int(legales[int(self.rng.integers(len(legales)))])
        #Si no salió el dado de exploración entonces explotamos lo que sabemos (tau=0). Si estuvieramos evaluando usamos nuestro nivel deseado de temperatura.
        tau = 0.0 if explorar else self.tau_eval
        #Finalmente, retornamos el estado y su respectiva mejor acción
        return estado, self._mejor_accion(estado, mascara, tau=tau)

    def __call__(self, obs: np.ndarray, mascara: np.ndarray) -> int:
        return self.elegir(obs, mascara, explorar=False)[1]

    #Ahora veremos el aprendizaje del agente Q
    #Definimos el alpha efectivo como un alpha que decae a medida que más visitamos un estado
    def _alpha_efectivo(self, estado: int, accion: int) -> float:
        n = self.visitas[estado, accion]
        lr = self.alpha / (1.0 + self.decaimiento_alpha * n)
        return max(self.alpha_min, lr)
    #Método que permite al agente
    def actualizar(
        self,
        estado: int,
        accion: int,
        recompensa: float,
        estado_sig: int | None,
        mascara_sig: np.ndarray | None,
        terminal: bool,
    ) -> None:
        #Si ya no hay mano que jugar obtenemos la recompensa
        if terminal:
            objetivo = recompensa
        #Si hay mano que jugar entonces a la recompensa le sumamos la mejor siguiente jugada posible ponderada por la visión a futuro
        else:
            objetivo = recompensa + self.gamma * float(self.Q[estado_sig][mascara_sig].max())
        #Obtenemos nuestro lr variable
        lr = self._alpha_efectivo(estado, accion)
        #Y completamos la fórmula del Q -learning
        self.Q[estado, accion] += lr * (objetivo - self.Q[estado, accion])
        #Y sumamos una visita a ese indice
        self.visitas[estado, accion] += 1
    
    #Método que refresca las visitas para mantener un porcentaje de lo aprendido pero lograr que el agente aprenda denuevo
    def preparar_warm_start(self, factor_visitas: float = 0.25) -> None:
        """Reduce el conteo de visitas heredado para permitir adaptación rápida frente a nuevos rivales."""
        self.visitas = (self.visitas * factor_visitas).astype(np.int64)
    #Método que permite cuantos estados de la Q-table se han visitado
    def estados_visitados(self) -> int:
        return int((self.visitas.sum(axis=1) > 0).sum())
    #Genera un código de barras asociado a la tabla Q que me permite detectar rápidamente cambios
    def hash_q(self) -> str:
        """SHA-256 de la tabla Q (float32 contiguo)."""
        return hashlib.sha256(np.ascontiguousarray(self.Q, dtype=np.float32).tobytes()).hexdigest()

    #Genera una vista de lectura el agente para estudiarlo
    def vista_greedy(self, rng: np.random.Generator) -> "AgenteQ":
        """Vista de solo lectura para evaluar sin alterar el rng del entrenamiento."""
        v = AgenteQ.__new__(AgenteQ)
        v.nombre, v.alpha, v.gamma = self.nombre, self.alpha, self.gamma
        v.alpha_min, v.decaimiento_alpha = self.alpha_min, self.decaimiento_alpha
        v.epsilon = 0.0
        v.tau_eval = self.tau_eval
        v.rng = rng
        v.Q, v.visitas, v.episodios = self.Q, self.visitas, self.episodios
        return v

    #Copia de un agente independiente
    def copiar(self, nombre: str | None = None) -> "AgenteQ":
        nuevo = AgenteQ(
            nombre or self.nombre,
            self.alpha,
            self.gamma,
            0.0,
            np.random.default_rng(int(self.rng.integers(2**31))),
            alpha_min=self.alpha_min,
            decaimiento_alpha=self.decaimiento_alpha,
            tau_eval=self.tau_eval,
        )
        nuevo.Q = self.Q.copy()
        nuevo.visitas = self.visitas.copy()
        nuevo.episodios = self.episodios
        return nuevo
    #Congela al agente 
    def guardar(self, ruta: str, extra: dict | None = None) -> None:
        meta = {
            "nombre": self.nombre,
            "alpha": self.alpha,
            "alpha_min": self.alpha_min,
            "decaimiento_alpha": self.decaimiento_alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "tau_eval": self.tau_eval,
            "episodios": self.episodios,
            "extra": extra or {},
        }
        np.savez_compressed(ruta, Q=self.Q, visitas=self.visitas, meta=np.array(json.dumps(meta)))
    #Carga un agente congelado
    @classmethod
    def cargar(cls, ruta: str, rng: np.random.Generator | None = None) -> "AgenteQ":
        with np.load(ruta, allow_pickle=False) as datos:
            meta = json.loads(str(datos["meta"]))
            agente = cls(
                meta["nombre"],
                meta["alpha"],
                meta["gamma"],
                meta["epsilon"],
                rng,
                alpha_min=meta.get("alpha_min", 0.015),
                decaimiento_alpha=meta.get("decaimiento_alpha", 0.02),
                tau_eval=meta.get("tau_eval", 0.0),
            )
            agente.Q = datos["Q"].astype(np.float32)
            agente.visitas = datos["visitas"].astype(np.int64)
            agente.episodios = int(meta["episodios"])
        return agente