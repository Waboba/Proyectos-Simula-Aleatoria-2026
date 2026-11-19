"""Muestreo adaptativo de puntas de la aproximación de rendijas verticales."""
from pathlib import Path
import json,time
import numpy as np
from sle_gpu import GPUMaps
from calibrar import browniano_maestro

HERE=Path(__file__).resolve().parent
STEPS={2:32768,6:65536,20:131072}

def prepare(kappa,force=False):
    path=HERE/f'traza_k{kappa}.npz'
    if path.exists() and not force:
        d=np.load(path)
        return {k:d[k] for k in d.files}
    start=time.perf_counter();gpu=GPUMaps();B=browniano_maestro();n=STEPS[kappa]
    U=np.sqrt(kappa)*B[::(len(B)-1)//n]
    ks=np.unique(np.linspace(0,n,2049).astype(np.int32))
    pts=gpu.trace(U,ks,1/n,batch=1024)
    # One common viewport preserves spatial scale across all three videos.
    bounds=np.array([-6.2,2.2,0.,4.725])
    tolerance=(bounds[1]-bounds[0])/3840*1.6
    for iteration in range(8):
        selected=(np.abs(np.diff(pts))>tolerance)&(np.diff(ks)>1)
        mids=(ks[:-1][selected]+ks[1:][selected])//2
        if not len(mids):break
        q=gpu.trace(U,mids,1/n,batch=1024)
        ks=np.r_[ks,mids];pts=np.r_[pts,q]
        order=np.argsort(ks);ks=ks[order];pts=pts[order]
        print(f'k={kappa}: ronda {iteration+1}, {len(ks)} puntas',flush=True)
    gaps=np.abs(np.diff(pts))
    stats={'kappa':kappa,'N':n,'trace_samples':len(ks),'seconds':time.perf_counter()-start,
           'max_tip_gap':float(gaps.max()),'gaps_over_8_pixels_at_4k':int((gaps>8*(bounds[1]-bounds[0])/3840).sum()),
           'max_y':float(pts.imag.max()),'min_x':float(pts.real.min()),'max_x':float(pts.real.max())}
    result={'indices':ks,'gamma':pts,'U':U,'bounds':bounds,'stats':np.array(json.dumps(stats))}
    np.savez_compressed(path,**result)
    print(stats,flush=True)
    return result

if __name__=='__main__':
    for k in STEPS:prepare(k)
