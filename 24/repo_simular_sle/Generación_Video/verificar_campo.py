"""Diagnósticos de mapa inverso y derivada sobre una muestra espacial amplia."""
import json
from pathlib import Path
import numpy as np
from sle_gpu import GPUMaps
from calibrar import browniano_maestro
from preparar_trazas import STEPS

def verificar():
    gpu=GPUMaps();B=browniano_maestro();rng=np.random.default_rng(721)
    w=rng.uniform(-7,3,1024)+1j*np.exp(rng.uniform(np.log(.02),np.log(5),1024))
    delta=1e-5
    def evaluate(U):
        values=gpu.inverse(U,np.r_[w,w+delta,w-delta],1/(len(U)-1))
        f,plus,minus=np.split(values,3)
        return f,(plus-minus)/(2*delta)
    report=[]
    for k,n in STEPS.items():
        fref,dref=evaluate(np.sqrt(k)*B)
        f,d=evaluate(np.sqrt(k)*B[::(len(B)-1)//n])
        distance=np.abs(f-fref);scale=np.abs(np.log2(np.abs(d))-np.log2(np.abs(dref)))
        row={'kappa':k,'N':n,'source_points':len(w),'map_p95':float(np.quantile(distance,.95)),
             'map_max':float(distance.max()),'log2_derivative_p95':float(np.quantile(scale,.95)),
             'log2_derivative_max':float(scale.max()),'derivative_estimator':'centered difference h=1e-5',
             'reference_N':len(B)-1,'source_rectangle':[-7,3,.02,5]}
        report.append(row);print(row,flush=True)
    # Exact inverse of the zero-driver vertical slit at t=1.
    exact=np.sqrt(w*w-4);exact=np.where(exact.imag<0,-exact,exact)
    values=gpu.inverse(np.zeros(4097),w,1/4096)
    np.testing.assert_allclose(values,exact,rtol=1e-10,atol=1e-10)
    result={'samples':report,'zero_driver_inverse_max_error':float(np.abs(values-exact).max()),
            'scope':'Finite-point diagnostics at t=1, not a global bound and not a boundary trace guarantee.'}
    (Path(__file__).parent/'calibracion_campo.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result

if __name__=='__main__':verificar()
