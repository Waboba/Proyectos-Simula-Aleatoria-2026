from os import environ
environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from scipy.optimize import curve_fit


# Indices aleatorios (i, j) de un sitio interior de la grilla.
# Recibe: el tamano N.
def getij(N):
    i=np.random.randint(1,N-1)
    j=np.random.randint(1,N-1)

    return i,j


# Lista de los indices (i, j) de los sitios del borde de una grilla N x N.
# Recibe: el tamano N.
def getBorderIndexes(N):
    l=np.append(np.append([[0,j] for j in range(N)],[[N-1,j] for j in range(N)],axis=0), np.append( [[i,0] for i in range(1,N)], [[i,N-1] for i in range(1,N)],axis=0),axis=0)
    return l


# Offsets a en forma de piramide: 1 en el centro y decreciendo hacia el borde, donde vale 0.
# Recibe: el tamano N.
def get_a_pyramid(N):
    if N%2 == 1:
        steps = (N-1)/2
        c=np.array([(N-1)/2, (N-1)/2])
    else:
        steps = N/2 - 1
        c=np.array([(N-1)/2, (N-1)/2])

    a=np.zeros((N,N))
    for i in range(1,N-1):
        for j in range(1,N-1):
            step = np.floor(max(abs(c - np.array([i,j]))))
            a[i,j] = 1 - step/steps

    return a


# Offsets a nulos (matriz N x N de ceros).
# Recibe: el tamano N.
def get_a_zero(N):
    return np.zeros((N,N))


# L matrices de offsets a uniformes en [0, 1), con el borde en 0.
# Recibe: L, el tamano N y la semilla. Devuelve un arreglo (L, N, N).
def get_a_random(L,N, semilla = None):
    rng = np.random.default_rng(semilla)
    a= rng.uniform(size=(L,N,N))
    for i,j in getBorderIndexes(N):
        a[:,i,j]=0
    return a


# Valores en el centro de cada configuracion.
# Recibe: sigma (L, N, N) y N. Devuelve un arreglo de largo L.
def get_middlelist(sigma,N):
    return sigma[:,N//2,N//2]


# Funcion cuya raiz es el lambda = log(1 - p) optimo de la propuesta geometrica simetrica (el que minimiza el numero
# esperado de intentos del rechazo).
# Recibe: lambd, mu y sigma2 (escalares o arreglos).
def fminimizar(lambd, mu, sigma2):
    return lambd*sigma2 + mu + 1/np.expm1(-lambd)


# Derivada de fminimizar respecto a lambda.
# Recibe: lambd y sigma2.
def derivadaf(lambd, sigma2):
    el = np.exp(lambd)
    return sigma2 + el/((1 - el)**2)


# Resuelve fminimizar = 0 con Newton-Raphson (vectorizado), partiendo de lambda = -1e-3, y devuelve p = 1 - exp(lambda).
# Recibe: arreglos mu y sigma2, la tolerancia relativa tol y el maximo de iteraciones max_iter.
def newton_raphson(mu, sigma2, tol=1e-12, max_iter=100):
    lambd = np.full(len(mu), -1e-3)
    with np.errstate(over="ignore"):
        for _ in range(max_iter):
            nuevo = lambd - fminimizar(lambd, mu, sigma2)/derivadaf(lambd, sigma2)
            listo = np.abs(nuevo - lambd) <= tol*np.maximum(1, np.abs(lambd))
            lambd = nuevo
            if listo.all():
                break
    return 1 - np.exp(lambd)


# Tabla de p optimos en una grilla de n valores de mu en [-bnds, bnds], para cada beta (sigma2 = 1/(4*beta)).
# Recibe: betas, bnds y n. Devuelve (mu_l, P), con P de forma (len(betas), n).
def get_plist(betas, bnds=1, n=100):
    betas = np.asarray(betas, dtype=float)
    B = len(betas)
    mu_l = np.linspace(-bnds, bnds, n)
    mu_all = np.tile(mu_l, B)
    sigma2_all = np.repeat(1/(4*betas), n)
    P = newton_raphson(mu_all, sigma2_all).reshape(B, n)
    return mu_l, P


# Interpola linealmente la tabla P en los valores mu, usando para cada elemento la fila (beta) indicada en row_idx.
# Recibe: mu, la grilla mu_l, la tabla P y row_idx.
def interp_rows(mu, mu_l, P, row_idx):
    n = len(mu_l)
    dx = mu_l[1] - mu_l[0]
    x = (mu - mu_l[0]) / dx
    k = np.minimum(x.astype(np.int64), n - 2)
    t = x - k
    return (1 - t) * P[row_idx, k] + t * P[row_idx, k + 1]


# Muestrea la normal discreta en Z (P(n) proporcional a exp(-(n - mu)^2/(2*sigma2))) para mu en [0, 0.5], por rechazo con una
# propuesta geometrica simetrica.
# Recibe: arreglos mu y sigma2, la tabla mu_P = (mu_l, P), row_idx y el generador rng.
def NormZ(mu, sigma2, mu_P, row_idx, rng):
    mu_l, P = mu_P

    p = interp_rows(mu, mu_l, P, row_idx)
    p = np.minimum(p, 0.999999)

    lg = np.log1p(-p)
    logk = 0.5 * lg**2 * sigma2 + mu * lg

    k = len(mu)
    result = np.empty(k, dtype=np.int64)
    active = np.arange(k)

    while active.size > 0:
        geom = rng.geometric(p[active])
        U = rng.random(active.size)
        n = np.where(U < 0.5, geom, 1 - geom)

        logprob = (
            -(n - mu[active])**2 / (2*sigma2[active])
            - logk[active]
            - lg[active] * (geom - 1)
        )
        accepted = np.log(rng.random(active.size)) < logprob
        result[active[accepted]] = n[accepted]
        active = active[~accepted]

    return result


# Muestrea la normal discreta en Z para cualquier mu: traslada mu a [0, 1), lo refleja a [0, 0.5] y llama a NormZ.
# Recibe: lo mismo que NormZ.
def NormZgeneral(mu, sigma2, mu_P, row_idx, rng):
    shift = np.floor(mu)
    r = mu - shift
    reflect = r > 0.5
    folded_mu = np.minimum(r, 1 - r)
    X = NormZ(folded_mu, sigma2, mu_P, row_idx, rng)
    return np.where(reflect, shift + 1 - X, shift + X)


# Grafica una matriz como superficie 3D.
# Recibe: la matriz, el titulo y argumentos extra para plot_surface (por ejemplo cmap).
def surface_plot(matrix, title, **kwargs):
    (x, y) = np.meshgrid(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x, y, matrix, **kwargs)

    ax.set_title(title)
    ax.set_xlabel('X (cols)')
    ax.set_ylabel('Y (rows)')
    ax.set_zlabel('Z (values)')

    plt.show()
    return


# Grafica cuantas veces aparece cada valor distinto de una lista.
# Recibe: la lista z_list.
def hist_plot(z_list):

    keys=[]
    for val in z_list:
        if val in keys:
            pass
        else:
            keys.append(val)

    z_dict =dict.fromkeys(keys,0)
    for val in z_list:
        z_dict[val]+=1

    vals=z_dict.values()

    plt.scatter(keys,vals)
    plt.grid()
    plt.show()
    return


# Grafica lado a lado las superficies de sigma, a y sigma - a.
# Recibe: a_matrix, sigma_matrix (N x N), el titulo (valor de beta como texto) y argumentos extra para plot_surface.
def surface_plotmany(a_matrix, sigma_matrix, title,**kwargs):

    fig, axs = plt.subplots(
        ncols=3,
        nrows=1,
        figsize=(14, 4),
        subplot_kw={"projection": "3d"},
        layout="constrained"
    )

    x, y = np.meshgrid(
        np.arange(sigma_matrix.shape[1]),
        np.arange(sigma_matrix.shape[0])
    )

    surf1 = axs[0].plot_surface(x, y, sigma_matrix, **kwargs)
    axs[0].set_title(r"$\sigma$")

    x, y = np.meshgrid(
        np.arange(a_matrix.shape[1]),
        np.arange(a_matrix.shape[0])
    )

    surf2 = axs[1].plot_surface(x, y, a_matrix, **kwargs)
    axs[1].set_title(r"$a$")

    surface = sigma_matrix - a_matrix

    surf3 = axs[2].plot_surface(x, y, surface, **kwargs)
    axs[2].set_title(r"$\sigma-a$")

    fig.suptitle(r"$\beta = $"+title)
    plt.show()

    return


# Ruta desde donde se leen los resultados para plotear (cambiala si corres el analisis desde otra carpeta)
ruta = r"Simulación Estocástica\SS-MinimarSurfaces\data_walo_cont"


# Carga los offsets a usados en la corrida guardada en ruta.
# Recibe: N. Devuelve un arreglo (K, N, N).
def get_a_used(N):
    a = np.load(f"{ruta}/a_N{N}.npy")
    return a


# Carga las configuraciones finales guardadas en ruta.
# Recibe: N. Devuelve sigma (B, K, L, N, N).
def get_fullsigma(N):
    sigmasT = np.array(np.load(f"{ruta}/sigma_N{N}.npy"))
    return sigmasT


# Carga los betas de la corrida guardada en ruta.
# Recibe: N.
def get_betas(N):
    betas = np.load(f"{ruta}/betas_N{N}.npy")
    return betas


# psi en el centro de cada configuracion final: centros - a del centro.
# Recibe: N. Devuelve un arreglo (B, K, L).
def get_sigma(N):
    sigma1 = np.load(f"{ruta}/centros_N{N}.npy")
    a = get_a_used(N)
    a = a[:, N//2, N//2]

    sigma1 = sigma1 - a[None, :, None]
    return sigma1


# Varianza total de psi_c para cada beta, juntando todos los desordenes y cadenas.
# Recibe: sigma (B, K, L).
def var(sigma):
    B, K, L = sigma.shape
    var = np.var(sigma.reshape(B, K*L), axis = 1, ddof = 1)
    return var


# Varianza termica: promedio sobre desordenes de la varianza entre las cadenas de un mismo a.
# Recibe: sigma (B, K, L).
def vartermica(sigma):
    v = np.var(sigma, axis = 2, ddof = 1)
    return np.mean(v, axis = 1)


# Varianza del desorden: varianza entre desordenes de la media de las cadenas, menos vartermica/L.
# Recibe: sigma (B, K, L).
def vardesorden(sigma):
    L = sigma.shape[2]
    m = np.mean(sigma, axis = 2)
    return np.var(m, axis = 1, ddof = 1) - vartermica(sigma)/L


# Media de psi_c para cada beta.
# Recibe: sigma (B, K, L).
def mean(sigma):
    return np.mean(sigma, axis = (1, 2))


# Grafica la varianza total contra beta en escala log-log.
# Recibe: la lista Ns y promediar (usar los promedios en el tiempo m, q).
def plotvar(Ns, promediar = False):
    for N in Ns:
        betas = get_betas(N)
        variations, termica, desorden = get_varianzas(N, promediar)
        plt.plot(betas, variations, label = 'Var N = ' + str(N))

    plt.loglog()
    plt.axvline(x=np.pi, linestyle = '--', color = 'black')
    plt.legend()
    plt.title(r"Varianza total con respecto a $\beta$")
    plt.xlabel("Beta")
    plt.ylabel(r"Var($\sigma _c$)")
    plt.grid()
    plt.show()
    return


# Grafica la media de psi_c contra beta.
# Recibe: la lista Ns.
def plotmean(Ns):
    for N in Ns:
        betas = get_betas(N)
        sigma=get_sigma(N)
        means=mean(sigma)
        plt.plot(betas, means, label = 'means_' + str(N))
    plt.axvline(x=np.pi, linestyle = '--', color = 'black')
    plt.legend()
    plt.xlabel("Beta")
    plt.grid()
    plt.show()
    return


# CLAIM: Estas funciones (de la linea 346 a la linea 680) fueron generadas mediante inteligencia artificial (Claude, de Anthropic). -------------------------------------------------
# G(c, c): elemento central de la inversa del laplaciano 2D con borde 0 (la varianza termica exacta es G(c, c)/beta).
# Recibe: N.
def Gcc(N):
    n = N - 2
    I = np.eye(n); T = 2*I - np.eye(n, k=1) - np.eye(n, k=-1)
    Lap = np.kron(T, I) + np.kron(I, T)
    c = (N//2 - 1)*n + (N//2 - 1)
    e = np.zeros(n*n); e[c] = 1
    return np.linalg.solve(Lap, e)[c]


# Carga los promedios en el tiempo m = <psi_c> y q = <psi_c^2>.
# Recibe: N. Devuelve m, q de forma (B, K, L).
def get_promedios(N):
    m = np.load(f"{ruta}/m_N{N}.npy")
    q = np.load(f"{ruta}/q_N{N}.npy")
    return m, q


# Promedio de los productos m_A * m_B sobre los pares distintos de cadenas de un mismo desorden.
# Recibe: m (B, K, L). Devuelve un arreglo (B, K).
def pares(m):
    L = m.shape[2]
    return (m.sum(axis = 2)**2 - (m**2).sum(axis = 2))/(L*(L-1))


# Varianza total a partir de los promedios en el tiempo: E[q] - E[m]^2.
# Recibe: m, q (B, K, L).
def var_t(m, q):
    return np.mean(q, axis = (1, 2)) - np.mean(m, axis = (1, 2))**2


# Varianza termica a partir de los promedios en el tiempo: E[q] - E[m_A m_B].
# Recibe: m, q (B, K, L).
def vartermica_t(m, q):
    return np.mean(q, axis = (1, 2)) - np.mean(pares(m), axis = 1)


# Varianza del desorden a partir de los promedios en el tiempo: E[m_A m_B] - E[m]^2.
# Recibe: m (B, K, L).
def vardesorden_t(m):
    return np.mean(pares(m), axis = 1) - np.mean(m, axis = (1, 2))**2


# Error estandar de la varianza del desorden para cada beta.
# Recibe: N y promediar.
def err_desorden(N, promediar = False):
    if promediar:
        m, q = get_promedios(N)
        return np.std(pares(m), axis = 1, ddof = 1)/np.sqrt(m.shape[1])
    sigma = get_sigma(N)
    K, L = sigma.shape[1], sigma.shape[2]
    return np.sqrt(2/(K-1))*(np.abs(vardesorden(sigma)) + vartermica(sigma)/L)


# Las tres varianzas (total, termica, desorden) para cada beta.
# Recibe: N y promediar (True: con m, q; False: con las configuraciones finales).
def get_varianzas(N, promediar = False):
    if promediar:
        m, q = get_promedios(N)
        return var_t(m, q), vartermica_t(m, q), vardesorden_t(m)
    sigma = get_sigma(N)
    return var(sigma), vartermica(sigma), vardesorden(sigma)


# Grafica la varianza del desorden con barras de error contra beta, desde bmin.
# Recibe: la lista Ns, promediar y bmin.
def plotdesorden(Ns, promediar = False, bmin = 0.5):
    for N in Ns:
        betas = get_betas(N)
        desorden = get_varianzas(N, promediar)[2]
        err = err_desorden(N, promediar)
        m = betas >= bmin
        plt.errorbar(betas[m], desorden[m], yerr = err[m], fmt = 'o-', ms = 3, capsize = 2, label = 'desorden_' + str(N))
    plt.axhline(y=0, color = 'gray')
    plt.axvline(x=np.pi, linestyle = '--', color = 'black')
    plt.legend()
    plt.xlabel("Beta")
    plt.grid()
    plt.show()
    return


# Grafica beta*H promedio a lo largo de los barridos para algunos betas, junto a (N-2)^2/2 (equiparticion).
# Recibe: N y cuantos (numero de betas a mostrar).
def plotbetaH(N, cuantos = 6):
    H = np.load(f"{ruta}/H_N{N}.npy")
    barridos = np.load(f"{ruta}/H_barridos_N{N}.npy")
    betas = get_betas(N)[-H.shape[1]:]
    for k in np.linspace(0, H.shape[1]-1, cuantos).astype(int):
        plt.plot(barridos, betas[k]*H[:, k], label = r'$\beta$ = ' + f"{betas[k]:.2f}")
    plt.axhline(y=(N-2)**2/2, linestyle = '--', color = 'black', label = r'$(N-2)^2/2$')
    plt.legend()
    plt.xlabel("Barrido")
    plt.ylabel(r"$\beta H$ promedio")
    plt.title("N = " + str(N))
    plt.grid()
    plt.show()
    return


# Grafica H/N^2 promedio a lo largo de los barridos para algunos betas.
# Recibe: N y cuantos.
def plotH(N, cuantos = 6):
    H = np.load(f"{ruta}/H_N{N}.npy")
    barridos = np.load(f"{ruta}/H_barridos_N{N}.npy")
    betas = get_betas(N)[-H.shape[1]:]
    for k in np.linspace(0, H.shape[1]-1, cuantos).astype(int):
        plt.plot(barridos, H[:, k]/N**2, label = r'$\beta$ = ' + f"{betas[k]:.2f}")
    plt.legend()
    plt.xlabel("Barrido")
    plt.ylabel(r"$H/N^2$ promedio")
    plt.title("N = " + str(N))
    plt.grid()
    plt.show()
    return


# Grafica la tasa de aceptacion de los intercambios entre betas vecinos.
# Recibe: la lista Ns.
def plotswaps(Ns):
    for N in Ns:
        tasa = np.load(f"{ruta}/swaps_N{N}.npy")
        betas = get_betas(N)[-len(tasa)-1:]
        plt.plot(betas[:-1], tasa, marker = 'o', label = 'N = ' + str(N))
    plt.xscale("log")
    plt.ylim(0, 1)
    plt.legend()
    plt.xlabel("Beta")
    plt.ylabel("Aceptacion de intercambios")
    plt.grid()
    plt.show()
    return


# Imprime y devuelve una de las varianzas y sus derivadas primera y segunda respecto a beta.
# Recibe: N, promediar y cual (0 total, 1 termica, 2 desorden).
def derivadas(N, promediar = False, cual = 2):
    nombres = ['total', 'termica', 'desorden']
    betas = get_betas(N)
    v = get_varianzas(N, promediar)[cual]
    b, inv = np.unique(betas, return_inverse = True)
    v = np.bincount(inv, weights = v)/np.bincount(inv)
    d1 = np.gradient(v, b)
    d2 = np.gradient(d1, b)
    print(f"N = {N}, varianza {nombres[cual]}")
    for k in range(len(b)):
        print(f"beta = {b[k]:6.3f}   var = {v[k]: .5f}   d1 = {d1[k]: .5f}   d2 = {d2[k]: .5f}")
    return b, v, d1, d2


# Grafica la varianza elegida y sus dos derivadas respecto a beta, desde bmin.
# Recibe: la lista Ns, promediar, cual y bmin.
def plotderivadas(Ns, promediar = False, cual = 2, bmin = 0.5):
    nombres = ['total', 'termica', 'desorden']
    fig, axs = plt.subplots(3, 1, sharex = True, figsize = (7, 9))
    for N in Ns:
        b, v, d1, d2 = derivadas(N, promediar, cual)
        m = b >= bmin
        axs[0].plot(b[m], v[m], 'o-', ms = 3, label = nombres[cual] + '_' + str(N))
        axs[1].plot(b[m], d1[m], 'o-', ms = 3, label = 'd1_' + str(N))
        axs[2].plot(b[m], d2[m], 'o-', ms = 3, label = 'd2_' + str(N))
    axs[0].set_ylabel("Varianza " + nombres[cual])
    axs[1].set_ylabel(r"$d/d\beta$")
    axs[2].set_ylabel(r"$d^2/d\beta^2$")
    for ax in axs:
        ax.axvline(x=np.pi, linestyle = '--', color = 'black')
        ax.axhline(y=0, color = 'gray')
        ax.grid()
        ax.legend()
    axs[2].set_xlabel("Beta")
    plt.show()
    return


# Ajusta V = C*beta^p en log-log usando los betas <= bfin y lo grafica junto a los datos.
# Recibe: la lista Ns, promediar, bfin y cual.
def plotajuste(Ns, promediar = False, bfin = 1.5, cual = 0):
    nombres = ['var', 'termica', 'desorden']
    for N in Ns:
        betas = get_betas(N)
        v = get_varianzas(N, promediar)[cual]
        m = (betas <= bfin) & (v > 0)
        pend, inter = np.polyfit(np.log(betas[m]), np.log(v[m]), 1)
        print(f"N = {N}: log V = {pend:.3f} log(beta) + {inter:.3f}  ->  V = {np.exp(inter):.3f} * beta^({pend:.3f})   (G(c,c) = {Gcc(N):.3f})")
        plt.plot(betas, v, 'o', ms = 3, label = nombres[cual] + ' N = ' + str(N))
        plt.plot(betas, np.exp(inter)*betas**pend, '--', label = f'ajuste_{N}: {np.exp(inter):.3f} beta^({pend:.2f})')
    plt.loglog()
    plt.axvline(x=bfin, linestyle = ':', color = 'gray')
    plt.legend()
    plt.xlabel("Beta")
    plt.title(r"Varianza total con respecto a $\beta$")
    plt.ylabel(r"Var($\sigma _c$)")
    plt.grid()
    plt.show()
    return


# Ajusta V = C*beta^p con los betas <= bfin y grafica los datos con el ajuste y, aparte, la resta varianza - ajuste.
# Recibe: la lista Ns, promediar, bfin y cual.
def plotresta(Ns, promediar = False, bfin = 1.5, cual = 0):
    nombres = ['variations', 'termica', 'desorden']
    ajustes = {}
    for N in Ns:
        betas = get_betas(N)
        v = get_varianzas(N, promediar)[cual]
        m = (betas <= bfin) & (v > 0)
        pend, inter = np.polyfit(np.log(betas[m]), np.log(v[m]), 1)
        ajustes[N] = (pend, inter)
        print(f"N = {N}: V_ajuste = {np.exp(inter):.3f} * beta^({pend:.3f})   (G(c,c) = {Gcc(N):.3f})")
        plt.plot(betas, v, 'o', ms = 3, label = nombres[cual] + '_' + str(N))
        plt.plot(betas, np.exp(inter)*betas**pend, '--', label = 'ajuste_' + str(N))
    plt.loglog()
    plt.axvline(x=bfin, linestyle = ':', color = 'gray')
    plt.legend()
    plt.xlabel("Beta")
    plt.grid()
    plt.show()

    for N in Ns:
        betas = get_betas(N)
        v = get_varianzas(N, promediar)[cual]
        pend, inter = ajustes[N]
        resta = v - np.exp(inter)*betas**pend
        plt.plot(betas, resta, 'o-', ms = 3, label = 'var - ajuste, N = ' + str(N))
    plt.axhline(y=0, color = 'gray')
    plt.axvline(x=np.pi, linestyle = '--', color = 'black')
    plt.title(r"Varianza - ajuste con respecto a $\beta$")
    plt.ylabel(r"Var($\sigma _c$) - ajuste")
    plt.legend()
    plt.xlabel("Beta")
    plt.grid()
    plt.show()
    return


# Para el beta mas cercano a beta_obj, ajusta Var_desorden = C*(ln N)^p entre los N dados y lo grafica.
# Recibe: la lista Ns, beta_obj y promediar.
def ajusteloglog(Ns, beta_obj, promediar = True):
    Ns = np.array(Ns)
    y, err, bs = [], [], []
    for N in Ns:
        betas = get_betas(N)
        k = np.argmin(np.abs(betas - beta_obj))
        y.append(get_varianzas(N, promediar)[2][k])
        err.append(err_desorden(N, promediar)[k])
        bs.append(betas[k])
    y, err = np.array(y), np.array(err)
    x = np.log(np.log(Ns))
    p, logc = np.polyfit(x, np.log(y), 1, w = y/err)
    print(f"beta ~ {beta_obj} (usados: {np.round(bs, 2)}): Var = {np.exp(logc):.4f} * (ln N)^{p:.2f}")
    plt.errorbar(x, np.log(y), yerr = err/y, fmt = 'o', capsize = 3, label = 'simulacion')
    xx = np.linspace(x.min(), x.max(), 50)
    plt.plot(xx, p*xx + logc, '--', label = f'pendiente p = {p:.2f}')
    plt.xlabel("log(ln N)")
    plt.ylabel("log(Var desorden)")
    plt.title(r"$\beta$ = " + str(beta_obj))
    plt.legend()
    plt.grid()
    plt.show()
    return p, np.exp(logc)


# Repite el ajuste Var_desorden = C*(ln N)^p en una grilla de n betas entre bmin y bmax y grafica el exponente p.
# Recibe: la lista Ns, promediar, bmin, bmax y n.
def ajusteloglog_betas(Ns, promediar = True, bmin = 3.5, bmax = 16, n = 20):
    Ns = np.array(Ns)
    bgrid = np.linspace(bmin, bmax, n)
    V = np.zeros((len(Ns), n))
    E = np.zeros((len(Ns), n))
    for i, N in enumerate(Ns):
        betas = get_betas(N)
        b, inv = np.unique(betas, return_inverse = True)
        cuenta = np.bincount(inv)
        v = np.bincount(inv, weights = get_varianzas(N, promediar)[2])/cuenta
        e = np.bincount(inv, weights = err_desorden(N, promediar))/cuenta
        V[i] = np.interp(bgrid, b, v)
        E[i] = np.interp(bgrid, b, e)

    x = np.log(np.log(Ns))
    ps, errp = [], []
    for k in range(n):
        y, err = V[:, k], E[:, k]
        coef, cov = np.polyfit(x, np.log(y), 1, w = y/err, cov = 'unscaled')
        ps.append(coef[0])
        errp.append(np.sqrt(cov[0, 0]))

    plt.errorbar(bgrid, ps, yerr = errp, fmt = 'o-', ms = 3, capsize = 2, label = 'p (Var ~ (ln N)^p)')
    plt.axhline(y=2, linestyle = ':', color = 'gray', label = 'p = 2 (asintotico)')
    plt.axhline(y=0, color = 'gray')
    plt.axvline(x=np.pi, linestyle = '--', color = 'black')
    plt.legend()
    plt.xlabel("Beta")
    plt.ylabel("p")
    plt.grid()
    plt.show()
    return bgrid, np.array(ps), np.array(errp)


# Ajusta a la resta varianza - C*beta^p (con C, p ajustados en beta <= bfin) dos modelos para beta >= bdesde:
# a*ln(beta/beta0) y C*(1 - beta_c/beta)^gamma, y grafica ambos.
# Recibe: N, promediar, bfin y bdesde.
def fitresta(N, promediar = True, bfin = 1.5, bdesde = 4):
    betas = get_betas(N)
    v = get_varianzas(N, promediar)[0]
    m = betas <= bfin
    pend, inter = np.polyfit(np.log(betas[m]), np.log(v[m]), 1)
    resta = v - np.exp(inter)*betas**pend

    s = betas >= bdesde
    b, d = betas[s], resta[s]

    def flog(beta, a, b0):
        return a*np.log(beta/b0)
    def ftrans(beta, C, bc, g):
        return C*np.clip(1 - bc/beta, 0, None)**g

    p1, c1 = curve_fit(flog, b, d, p0 = [0.07, 2.5])
    p2, c2 = curve_fit(ftrans, b, d, p0 = [0.2, np.pi, 1], maxfev = 10000)
    e1, e2 = np.sqrt(np.diag(c1)), np.sqrt(np.diag(c2))
    print(f"log:        a = {p1[0]:.4f} +- {e1[0]:.4f}, beta0 = {p1[1]:.3f} +- {e1[1]:.3f}")
    print(f"transicion: C = {p2[0]:.4f} +- {e2[0]:.4f}, beta_c = {p2[1]:.3f} +- {e2[1]:.3f}, gamma = {p2[2]:.3f} +- {e2[2]:.3f}")

    bb = np.linspace(bdesde, betas.max(), 200)
    plt.plot(betas, resta, 'o', ms = 3, label = 'var - ajuste, N = ' + str(N))
    plt.plot(bb, flog(bb, *p1), '-', label = f'{p1[0]:.3f} ln(beta/{p1[1]:.2f})')
    plt.plot(bb, ftrans(bb, *p2), '--', label = f'{p2[0]:.3f} (1 - {p2[1]:.2f}/beta)^{p2[2]:.2f}')
    plt.axhline(y=0, color = 'gray')
    plt.axvline(x=np.pi, linestyle = '--', color = 'black')
    plt.legend()
    plt.xlabel("Beta")
    plt.ylabel(r"Var($\sigma_c$) - ajuste")
    plt.grid()
    plt.show()
    return p1, p2


sigma = get_fullsigma(32)[6, 0, 0, :, :]
a = get_a_used(32)[6, :, :]
betas = get_betas(32)
b = betas[6]
surface_plotmany(a, sigma, f"{b:.3f}", cmap = 'coolwarm')
