import numpy as np
import tools as tp
import time
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

# Convencion: psi = sigma - a


# Suma de psi = sigma - a en los 4 vecinos del sitio (i, j), para todas las cadenas a la vez.
# Recibe: sigma y a de forma (C, N, N), y los indices i, j del sitio. (No se usa en la simulacion actual.)
def s(sigma,a,i,j):
    sval= sigma[:,i-1,j]-a[:,i-1,j]
    sval+= sigma[:,i+1,j]-a[:,i+1,j]
    sval+= sigma[:,i,j-1]-a[:,i,j-1]
    sval+= sigma[:,i,j+1]-a[:,i,j+1]
    return sval


# CLAIM: Estas funciones (de la linea 22 a la linea 310) fueron generadas mediante inteligencia artificial (Claude, de Anthropic).-------------------------



# Energia H = 1/2 * suma sobre aristas de (psi_x - psi_y)^2, con psi = sigma - a (el peso de cada configuracion es exp(-beta*H)).
# Recibe: sigma y a con forma (..., N, N). Devuelve H de cada configuracion (suma sobre los dos ultimos ejes).
def energia(sigma, a):
    psi = sigma - a
    dx = psi[..., 1:, :] - psi[..., :-1, :]
    dy = psi[..., :, 1:] - psi[..., :, :-1]
    return 0.5*((dx**2).sum(axis=(-2, -1)) + (dy**2).sum(axis=(-2, -1)))


# Escalera geometrica de betas para parallel tempering, con ceil(0.6*N*ln(bmax/bmin)) temperaturas.
# Recibe: el tamano N y los extremos bmin, bmax. Devuelve el arreglo de betas.
def escalera(N, bmin=2.5, bmax=11):
    Kbeta = int(np.ceil(0.6*N*np.log(bmax/bmin)))
    return np.geomspace(bmin, bmax, Kbeta)


# Simula C cadenas, cada una con una copia en cada beta. Un barrido actualiza todos los sitios interiores en tablero de ajedrez
# (bano termico con la normal discreta) y, si swaps = True, intenta un intercambio entre betas vecinos (parallel tempering).
# Recibe: N, betas, numero de barridos, a_c (C, N, N) con un a por cadena, seed, swaps, cada_H (cada cuantos barridos se
# registra H), inicio_medicion (barrido desde el que se promedian psi_c y psi_c^2; None no mide) y sigma0 (C, Kbeta, N, N)
# opcional para partir de esa configuracion (si es None se parte de round(a)).
# Devuelve: sigma final (C, Kbeta, N, N), registro de H promedio por beta, intercambios aceptados e intentados por par de
# betas, y los promedios en el tiempo m, q (C, Kbeta) (o None si no se mide).
def getsigmaPT(N, betas, barridos, a_c, seed=None, swaps=True, cada_H=10, inicio_medicion=None, sigma0=None):
    rng = np.random.default_rng(seed)
    betas = np.asarray(betas, dtype=float)
    Kbeta = len(betas)
    C = a_c.shape[0]
    c = N//2

    a = np.repeat(a_c, Kbeta, axis=0)
    if sigma0 is None:
        sigma = np.round(a)
    else:
        sigma = np.array(sigma0, dtype=float).reshape(C*Kbeta, N, N)
    var = np.tile(1/(4*betas), C)
    row_idx = np.tile(np.arange(Kbeta), C)
    mu_P = tp.get_plist(betas)

    R = C*Kbeta
    I, J = np.meshgrid(np.arange(1, N-1), np.arange(1, N-1), indexing="ij")
    colores = []
    for color in (0, 1):
        ii, jj = I[(I+J) % 2 == color], J[(I+J) % 2 == color]
        colores.append((ii, jj, np.repeat(var, len(ii)), np.repeat(row_idx, len(ii))))

    sigma4 = sigma.reshape(C, Kbeta, N, N)
    a4 = a.reshape(C, Kbeta, N, N)
    H_reg = []
    aceptados = np.zeros(max(Kbeta-1, 0))
    intentos = np.zeros(max(Kbeta-1, 0))
    suma_m = np.zeros((C, Kbeta))
    suma_q = np.zeros((C, Kbeta))
    n_med = 0

    for barrido in range(barridos):
        for ii, jj, var_c, row_c in colores:
            vecinos = (sigma[:, ii-1, jj] - a[:, ii-1, jj]) + (sigma[:, ii+1, jj] - a[:, ii+1, jj]) \
                    + (sigma[:, ii, jj-1] - a[:, ii, jj-1]) + (sigma[:, ii, jj+1] - a[:, ii, jj+1])
            mu = vecinos/4 + a[:, ii, jj]
            sigma[:, ii, jj] = tp.NormZgeneral(mu.ravel(), var_c, mu_P, row_c, rng).reshape(R, len(ii))

        if swaps and Kbeta > 1:
            H = energia(sigma4, a4)
            k0 = np.arange(barrido % 2, Kbeta-1, 2)
            if k0.size > 0:
                delta = (betas[k0+1] - betas[k0])*(H[:, k0+1] - H[:, k0])
                ok = np.log(rng.random(delta.shape)) < delta
                cs, ks = np.nonzero(ok)
                kk = k0[ks]
                tmp = sigma4[cs, kk].copy()
                sigma4[cs, kk] = sigma4[cs, kk+1]
                sigma4[cs, kk+1] = tmp
                aceptados[k0] += ok.sum(axis=0)
                intentos[k0] += C

        if inicio_medicion is not None and barrido >= inicio_medicion:
            psi_c = sigma4[:, :, c, c] - a4[:, :, c, c]
            suma_m += psi_c
            suma_q += psi_c**2
            n_med += 1

        if barrido % cada_H == 0:
            H_reg.append(energia(sigma4, a4).mean(axis=0))

    if n_med > 0:
        return sigma4, np.array(H_reg), aceptados, intentos, suma_m/n_med, suma_q/n_med
    return sigma4, np.array(H_reg), aceptados, intentos, None, None


# Tarea de un proceso: simula un bloque de desordenes, cada uno con L cadenas, en todos los betas.
# Recibe: una tupla (N, betas, barridos, a_bloque (Kb, N, N), L, seed, swaps, cada_H, inicio_medicion[, sigma_guardado]),
# donde sigma_guardado (Kbeta, Kb, L, N, N) es opcional y sirve para continuar una corrida anterior.
# Devuelve: sigma (Kbeta, Kb, L, N, N) en int8 o int16, H, aceptados, intentos y m, q (Kbeta, Kb, L).
def correr(args):
    N, betas, barridos, a_bloque, L, seed, swaps, cada_H, inicio_medicion = args[:9]
    sigma_guardado = args[9] if len(args) > 9 else None
    Kb = a_bloque.shape[0]
    Kbeta = len(betas)
    a_c = np.repeat(a_bloque, L, axis=0)
    sigma0 = None
    if sigma_guardado is not None:
        sigma0 = np.asarray(sigma_guardado).transpose(1, 2, 0, 3, 4).reshape(Kb*L, Kbeta, N, N)
    sg, H, aceptados, intentos, m, q = getsigmaPT(N, betas, barridos, a_c, seed=seed, swaps=swaps,
                                                  cada_H=cada_H, inicio_medicion=inicio_medicion, sigma0=sigma0)
    sg = sg.reshape(Kb, L, Kbeta, N, N).transpose(2, 0, 1, 3, 4)
    if sg.min() >= -128 and sg.max() <= 127:
        sg = sg.astype(np.int8)
    else:
        sg = sg.astype(np.int16)
    if m is not None:
        m = m.reshape(Kb, L, Kbeta).transpose(2, 0, 1)
        q = q.reshape(Kb, L, Kbeta).transpose(2, 0, 1)
    return sg, H, aceptados, intentos, m, q


# Corrida completa en paralelo. Para cada N genera y guarda los K desordenes y las semillas, corre Glauber en betas_bajos y
# parallel tempering en la escalera [bmin, bmax], reparte los desordenes en bloques entre n_proc procesos, guarda cada bloque
# al terminar (y se salta los que ya existen) y al final junta todo en la carpeta.
# Recibe: n_proc, K (desordenes), L (cadenas por desorden), Ns, betas_bajos, bmin, bmax, barridos_burn, barridos_med,
# factor_glauber (pasos de Glauber = factor_glauber*N^4), promediar, cada_H, carpeta, escalar_barridos (multiplicar los
# barridos por (N/N_ref)^2) y N_ref.
def simular(n_proc, K, L, Ns, betas_bajos, bmin, bmax, barridos_burn, barridos_med, factor_glauber,
            promediar, cada_H, carpeta, escalar_barridos=True, N_ref=16):
    os.makedirs(carpeta, exist_ok=True)
    Kb = int(np.ceil(K/n_proc))
    bloques = [np.arange(i, min(i + Kb, K)) for i in range(0, K, Kb)]

    for N in Ns:
        betas_pt = escalera(N, bmin, bmax)
        betas = np.concatenate([betas_bajos, betas_pt])
        np.save(f"{carpeta}/betas_N{N}.npy", betas)

        f = (N/N_ref)**2 if escalar_barridos else 1
        burn_N = max(1, int(round(barridos_burn*f)))
        med_N = max(1, int(round(barridos_med*f)))
        barridos_pt = burn_N + med_N
        barridos_gl = int(round(factor_glauber*N**4/(N-2)**2))
        med_pt = burn_N if promediar else None
        med_gl = min(N**2, barridos_gl - 1) if promediar else None

        print(f"N={N}: Glauber en {len(betas_bajos)} betas ({betas_bajos[0]} a {betas_bajos[-1]}, "
              f"{barridos_gl} barridos), parallel tempering en {len(betas_pt)} betas ({bmin} a {bmax}, "
              f"{burn_N}+{med_N} barridos), {K} desordenes x {L} cadenas, {len(bloques)} bloques, "
              f"promediar={promediar}", flush=True)

        f_sem = f"{carpeta}/semillas_a_N{N}.npy"
        f_din = f"{carpeta}/semilla_dinamica_N{N}.npy"
        if os.path.exists(f_sem):
            semillas_a = np.load(f_sem)
            semilla_din = int(np.load(f_din))
        else:
            semillas_a = np.random.SeedSequence().generate_state(K)
            semilla_din = int(np.random.SeedSequence().generate_state(1)[0])
            np.save(f_sem, semillas_a)
            np.save(f_din, semilla_din)

        a_todos = np.stack([tp.get_a_random(1, N, int(sem))[0] for sem in semillas_a])
        np.save(f"{carpeta}/a_N{N}.npy", a_todos)

        def archivo(tipo, j):
            return f"{carpeta}/{tipo}_N{N}_blk{j}.npz"

        tareas = []
        for j in range(len(bloques)):
            if not os.path.exists(archivo("pt", j)):
                tareas.append(("pt", j, betas_pt, barridos_pt, True, med_pt))
        for j in range(len(bloques)):
            if not os.path.exists(archivo("bajo", j)) and len(betas_bajos) > 0:
                tareas.append(("bajo", j, betas_bajos, barridos_gl, False, med_gl))
        tareas.sort(key=lambda t: -len(t[2])*t[3])
        print(f"N={N}: {len(tareas)} tareas pendientes", flush=True)

        t0 = time.time()
        with ProcessPoolExecutor(n_proc) as ex:
            futuros = {ex.submit(correr, (N, b, nb, a_todos[bloques[j]], L,
                                          [semilla_din, j, 0 if tipo == "pt" else 1], sw, cada_H, med)): (tipo, j)
                       for tipo, j, b, nb, sw, med in tareas}
            for n, fut in enumerate(as_completed(futuros), 1):
                tipo, j = futuros[fut]
                sg, H, aceptados, intentos, m, q = fut.result()
                extra = {} if m is None else {"m": m, "q": q}
                np.savez(archivo(tipo, j), sigma=sg, H=H, aceptados=aceptados, intentos=intentos, **extra)
                print(f"N={N} [{n}/{len(tareas)}] {tipo} bloque {j} listo ({time.time()-t0:.0f} s)", flush=True)

        def juntar(campo):
            partes = []
            if len(betas_bajos) > 0:
                partes.append(np.concatenate([np.load(archivo("bajo", j))[campo] for j in range(len(bloques))], axis=1))
            partes.append(np.concatenate([np.load(archivo("pt", j))[campo] for j in range(len(bloques))], axis=1))
            return np.concatenate(partes, axis=0)

        sigma = juntar("sigma")
        np.save(f"{carpeta}/sigma_N{N}.npy", sigma)
        np.save(f"{carpeta}/centros_N{N}.npy", sigma[:, :, :, N//2, N//2].astype(float))
        if promediar:
            np.save(f"{carpeta}/m_N{N}.npy", juntar("m"))
            np.save(f"{carpeta}/q_N{N}.npy", juntar("q"))

        pesos = np.array([len(bl) for bl in bloques], dtype=float)
        H = sum(w*np.load(archivo("pt", j))["H"] for j, w in enumerate(pesos))/pesos.sum()
        np.save(f"{carpeta}/H_N{N}.npy", H)
        np.save(f"{carpeta}/H_barridos_N{N}.npy", np.arange(H.shape[0])*cada_H)
        acc = sum(np.load(archivo("pt", j))["aceptados"] for j in range(len(bloques)))
        intn = sum(np.load(archivo("pt", j))["intentos"] for j in range(len(bloques)))
        tasa = acc/np.maximum(intn, 1)
        np.save(f"{carpeta}/swaps_N{N}.npy", tasa)
        print(f"N={N} listo: sigma {sigma.shape}, aceptacion de intercambios entre "
              f"{tasa.min():.2f} y {tasa.max():.2f}, {time.time()-t0:.0f} s", flush=True)


# Retoma el parallel tempering de una corrida anterior desde sus configuraciones finales (mismos a, escalera y cadenas),
# hace barridos_burn de equilibrado extra y una medicion nueva de barridos_med (escalados con (N/N_ref)^2). La parte de
# Glauber (betas bajos) se copia sin cambios.
# Recibe: n_proc, Ns, carpeta_vieja, carpeta_nueva, barridos_burn, barridos_med, cada_H, escalar_barridos y N_ref.
def continuar(n_proc, Ns, carpeta_vieja, carpeta_nueva, barridos_burn, barridos_med, cada_H=10,
              escalar_barridos=True, N_ref=16):
    import shutil, glob
    os.makedirs(carpeta_nueva, exist_ok=True)
    for N in Ns:
        f = (N/N_ref)**2 if escalar_barridos else 1
        burn_N = max(1, int(round(barridos_burn*f)))
        med_N = max(1, int(round(barridos_med*f)))

        betas = np.load(f"{carpeta_vieja}/betas_N{N}.npy")
        a_todos = np.load(f"{carpeta_vieja}/a_N{N}.npy")
        L = None
        archivos_pt = sorted(glob.glob(f"{carpeta_vieja}/pt_N{N}_blk*.npz"),
                             key=lambda s: int(s.split("blk")[-1].split(".")[0]))
        if len(archivos_pt) == 0:
            print(f"N={N}: no hay bloques de parallel tempering en {carpeta_vieja}, se salta", flush=True)
            continue

        bloques, ini = [], 0
        for arch in archivos_pt:
            sg = np.load(arch)["sigma"]
            Kbeta, Kb, L = sg.shape[0], sg.shape[1], sg.shape[2]
            bloques.append(np.arange(ini, ini + Kb))
            ini += Kb
        betas_pt = betas[-Kbeta:]

        for nombre in ["betas", "a", "semillas_a", "semilla_dinamica"]:
            src = f"{carpeta_vieja}/{nombre}_N{N}.npy"
            if os.path.exists(src):
                shutil.copy(src, f"{carpeta_nueva}/{nombre}_N{N}.npy")
        for src in glob.glob(f"{carpeta_vieja}/bajo_N{N}_blk*.npz"):
            shutil.copy(src, f"{carpeta_nueva}/{os.path.basename(src)}")

        def archivo(tipo, j):
            return f"{carpeta_nueva}/{tipo}_N{N}_blk{j}.npz"

        tareas = [j for j in range(len(bloques)) if not os.path.exists(archivo("pt", j))]
        print(f"N={N}: continuando {len(tareas)} bloques de parallel tempering ({len(betas_pt)} betas, "
              f"{burn_N}+{med_N} barridos extra) desde {carpeta_vieja}", flush=True)

        semilla = int(np.random.SeedSequence().generate_state(1)[0])
        t0 = time.time()
        with ProcessPoolExecutor(n_proc) as ex:
            futuros = {ex.submit(correr, (N, betas_pt, burn_N + med_N, a_todos[bloques[j]], L,
                                          [semilla, j, 2], True, cada_H, burn_N,
                                          np.load(archivos_pt[j])["sigma"])): j for j in tareas}
            for n, fut in enumerate(as_completed(futuros), 1):
                j = futuros[fut]
                sg, H, aceptados, intentos, m, q = fut.result()
                np.savez(archivo("pt", j), sigma=sg, H=H, aceptados=aceptados, intentos=intentos, m=m, q=q)
                print(f"N={N} [{n}/{len(tareas)}] pt bloque {j} listo ({time.time()-t0:.0f} s)", flush=True)

        def juntar(campo):
            partes = []
            if os.path.exists(archivo("bajo", 0)):
                partes.append(np.concatenate([np.load(archivo("bajo", j))[campo] for j in range(len(bloques))], axis=1))
            partes.append(np.concatenate([np.load(archivo("pt", j))[campo] for j in range(len(bloques))], axis=1))
            return np.concatenate(partes, axis=0)

        sigma = juntar("sigma")
        np.save(f"{carpeta_nueva}/sigma_N{N}.npy", sigma)
        np.save(f"{carpeta_nueva}/centros_N{N}.npy", sigma[:, :, :, N//2, N//2].astype(float))
        np.save(f"{carpeta_nueva}/m_N{N}.npy", juntar("m"))
        np.save(f"{carpeta_nueva}/q_N{N}.npy", juntar("q"))
        pesos = np.array([len(bl) for bl in bloques], dtype=float)
        H = sum(w*np.load(archivo("pt", j))["H"] for j, w in enumerate(pesos))/pesos.sum()
        np.save(f"{carpeta_nueva}/H_N{N}.npy", H)
        np.save(f"{carpeta_nueva}/H_barridos_N{N}.npy", np.arange(H.shape[0])*cada_H)
        acc = sum(np.load(archivo("pt", j))["aceptados"] for j in range(len(bloques)))
        intn = sum(np.load(archivo("pt", j))["intentos"] for j in range(len(bloques)))
        tasa = acc/np.maximum(intn, 1)
        np.save(f"{carpeta_nueva}/swaps_N{N}.npy", tasa)
        print(f"N={N} listo: aceptacion de intercambios entre {tasa.min():.2f} y {tasa.max():.2f}, "
              f"{time.time()-t0:.0f} s", flush=True)


if __name__ == "__main__":
    # Procesos: i5-9400 (6 nucleos) -> 5 ; Ryzen 9 4900HS (8 nucleos, 16 hilos) -> 7
    n_proc = 6
    K = 100                               # cantidad de a distintos (desordenes)
    L = 2                                 # cadenas por cada a (replicas A y B)
    Ns = [16, 32]
    promediar = True                      # True: promedios en el tiempo de psi_c ; False: solo la configuracion final
    betas_bajos = np.linspace(0.05, 2.5, 15)   # Glauber sin intercambios
    factor_glauber = 4                   # pasos de Glauber para los betas bajos = factor_glauber*N^4
    bmin, bmax = 2.5, 16                  # escalera de parallel tempering
    barridos_burn = 1000                  # barridos de equilibrado del parallel tempering (para N = N_ref)
    barridos_med = 2000                   # barridos de medicion (si promediar=False, se suman al equilibrado)
    escalar_barridos = True               # True: barridos * (N/N_ref)^2 (el tiempo de mezcla crece como N^2)
    N_ref = 16
    cada_H = 10                           # cada cuantos barridos se guarda H
    cont_burn = 1000                     # para "continuar": barridos extra de equilibrado (para N = N_ref)
    cont_med = 2000                       # para "continuar": barridos de la medicion nueva (para N = N_ref)
    carpeta = "data_pt"

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        simular(3, 6, 2, [8], np.linspace(0.05, 2.5, 3), bmin, bmax, 20, 20, 1, promediar, 5, "data_pt_test", False)
    elif len(sys.argv) > 1 and sys.argv[1] == "continuar":
        # python main.py continuar carpeta_vieja carpeta_nueva N1 N2 ...
        carpeta_vieja, carpeta_nueva = sys.argv[2], sys.argv[3]
        Ns_cont = [int(x) for x in sys.argv[4:]] or Ns
        continuar(n_proc, Ns_cont, carpeta_vieja, carpeta_nueva, cont_burn, cont_med, cada_H, escalar_barridos, N_ref)
    elif len(sys.argv) > 1 and sys.argv[1] == "plot":
        Ns_plot = [int(x) for x in sys.argv[2:]] or Ns
        tp.plotvar(Ns_plot, promediar)
        tp.plotdesorden(Ns_plot, promediar)
        tp.plotmean(Ns_plot)
        tp.plotswaps(Ns_plot)
        for N in Ns_plot:
            tp.plotH(N)
    else:
        simular(n_proc, K, L, Ns, betas_bajos, bmin, bmax, barridos_burn, barridos_med, factor_glauber,
                promediar, cada_H, carpeta, escalar_barridos, N_ref)
