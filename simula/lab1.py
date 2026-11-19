import numpy as np

N = 100000
np.random.seed(42)

# 1. Monte Carlo Estándar
U = np.random.uniform(0, 1, N)
X = np.exp(U)
mu_mc = np.mean(X)
var_mc = np.var(X, ddof=1) / N

# 2. Variables Antitéticas
U_half = np.random.uniform(0, 1, N // 2)
X_ant = 0.5 * (np.exp(U_half) + np.exp(1 - U_half))
mu_ant = np.mean(X_ant)
var_ant = np.var(X_ant, ddof=1) / (N // 2)

# 3. Variable de Control (Usando Y = U como control, E[Y] = 0.5)
Y = U
ey = 0.5
cov_xy = np.cov(X, Y)[0, 1]
var_y = np.var(Y, ddof=1)
c_opt = -cov_xy / var_y

X_ctrl = X + c_opt * (Y - ey)
mu_ctrl = np.mean(X_ctrl)
var_ctrl = np.var(X_ctrl, ddof=1) / N

print(f"Valor Real: {np.exp(1) - 1:.6f}")
print(f"MC Estándar:  {mu_mc:.6f} | Varianza Estimador: {var_mc:.2e}")
print(f"Antitética:   {mu_ant:.6f} | Varianza Estimador: {var_ant:.2e}")
print(f"Var Control:  {mu_ctrl:.6f} | Varianza Estimador: {var_ctrl:.2e}")


N_total = 10000
k = 4
N_i = N_total // k  # Asignación uniforme por estrato
w_i = 1 / k         # Peso de cada estrato

mu_estrat = 0
var_estrat_sum = 0

for i in range(k):
    # Generar muestras en el estrato [i/k, (i+1)/k]
    a, b = i / k, (i + 1) / k
    U_estrato = np.random.uniform(a, b, N_i)
    X_estrato = U_estrato**2  # Función a evaluar

    mu_i = np.mean(X_estrato)
    var_i = np.var(X_estrato, ddof=1)

    mu_estrat += w_i * mu_i
    var_estrat_sum += (w_i**2) * (var_i / N_i)

print(f"Estimador Estratificado: {mu_estrat:.6f}")
print(f"Varianza Estimador:      {var_estrat_sum:.2e}")


# Matriz de transición P (filas suman 1)
P = np.array([
    [0.7, 0.2, 0.1],
    [0.3, 0.4, 0.3],
    [0.2, 0.3, 0.5]
])

# 1. Simulación de Trayectoria
n_pasos = 50000
estado_actual = 0
conteo_estados = np.zeros(P.shape[0])

for _ in range(n_pasos):
    conteo_estados[estado_actual] += 1
    # Generar el siguiente estado según la fila correspondiente en P
    estado_actual = np.random.choice(len(P), p=P[estado_actual])

pi_simulada = conteo_estados / n_pasos

# 2. Cálculo Teórico de Distribución Estacionaria (Autovectores de P^T)
eigenvalues, eigenvectors = np.linalg.eig(P.T)
idx = np.argmin(np.abs(eigenvalues - 1.0))
pi_teorica = np.real(eigenvectors[:, idx])
pi_teorica /= np.sum(pi_teorica)

print("Distribución Estacionaria Simulada:", np.round(pi_simulada, 4))
print("Distribución Estacionaria Teórica: ", np.round(pi_teorica, 4))
