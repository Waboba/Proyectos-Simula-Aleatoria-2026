import json
from pathlib import Path
import time
import numpy as np
from sle_gpu import GPUMaps,validate,cpu_forward

HERE=Path(__file__).resolve().parent

def browniano_maestro(seed=23, power=18):
    path=HERE/f'browniano_semilla{seed}_2pow{power}.npz'
    if path.exists():
        return np.load(path)['B']
    n=2**power
    B=np.r_[0.,np.cumsum(np.random.default_rng(seed).standard_normal(n)/np.sqrt(n))]
    np.savez_compressed(path,B=B)
    return B

def calibrar():
    gpu=validate()
    B=browniano_maestro()
    rng=np.random.default_rng(240)
    # Fixed source points: include a separate narrow band near the real boundary.
    w=rng.uniform(-3,2,512)+1j*np.exp(rng.uniform(np.log(.02),np.log(2.5),512))
    interior=w.imag>=.15
    report={'gpu':gpu.name,'seed':23,'master_steps':len(B)-1,
            'method':'right-endpoint vertical slits; shared nested Brownian samples',
            'spatial_tests':[],'benchmarks':[]}
    for kappa in (2,6,20):
        refU=np.sqrt(kappa)*B
        ref=gpu.inverse(refU,w,1/(len(B)-1))
        for power in (13,14,15,16,17):
            n=2**power
            U=refU[::(len(B)-1)//n]
            a=gpu.inverse(U,w,1/n)
            errors=np.abs(a-ref)
            row={'kappa':kappa,'N':n,'median':float(np.median(errors)),
                 'p95':float(np.quantile(errors,.95)),'max':float(errors.max()),
                 'interior_p95':float(np.quantile(errors[interior],.95)),
                 'near_boundary_p95':float(np.quantile(errors[~interior],.95))}
            report['spatial_tests'].append(row)
            print(json.dumps(row),flush=True)
    for width,height in ((640,360),(1280,720),(1920,1080)):
        yy=np.linspace(.002,3.4,height)
        z=np.linspace(-3,3,width)[None,:]+1j*yy[:,None]
        U=np.sqrt(6)*B[::128]
        gpu.initialize(z,U,1/2048)
        start=time.perf_counter();gpu.advance(2048);seconds=time.perf_counter()-start
        row={'width':width,'height':height,'steps':2048,'seconds':seconds,
             'million_pixel_steps_per_second':z.size*2048/seconds/1e6}
        report['benchmarks'].append(row);print(json.dumps(row),flush=True)
    # Whole-plane approximation is NOT certified by these finite samples.
    report['interpretation']='Differences against N=262144 on 512 source points; not a global error bound.'
    (HERE/'calibracion.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report

if __name__=='__main__':
    calibrar()
