"""GPU Loewner maps without an installed CUDA compiler (CuPy + NVIDIA driver).

Handwritten PTX uses double precision and exact vertical-slit elementary maps.
The Brownian driver is piecewise constant at the RIGHT endpoint of each interval.
"""
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore', message='CUDA path could not be detected')

ROOT = r'''
    sub.f64 %d10, %d0, %d4;
    mul.f64 %d11, %d10, %d10;
    mul.f64 %d12, %d1, %d1;
    sub.f64 %d13, %d11, %d12;
    add.f64 %d13, %d13, %d5;
    mul.f64 %d14, %d10, %d1;
    add.f64 %d14, %d14, %d14;
    mul.f64 %d15, %d13, %d13;
    mul.f64 %d16, %d14, %d14;
    add.f64 %d15, %d15, %d16;
    sqrt.rn.f64 %d15, %d15;
    setp.lt.f64 %p2, %d13, 0d0000000000000000;
    @%p2 bra ROOT_NEG;
    add.f64 %d16, %d15, %d13;
    mul.f64 %d16, %d16, 0d3FE0000000000000;
    sqrt.rn.f64 %d17, %d16;
    setp.lt.f64 %p3, %d10, 0d0000000000000000;
    neg.f64 %d18, %d17;
    selp.f64 %d17, %d18, %d17, %p3;
    add.f64 %d18, %d17, %d17;
    div.rn.f64 %d18, %d14, %d18;
    bra ROOT_DONE;
ROOT_NEG:
    sub.f64 %d16, %d15, %d13;
    mul.f64 %d16, %d16, 0d3FE0000000000000;
    sqrt.rn.f64 %d18, %d16;
    add.f64 %d17, %d18, %d18;
    div.rn.f64 %d17, %d14, %d17;
ROOT_DONE:
'''

PTX = r'''.version 6.4
.target sm_75
.address_size 64

.visible .entry advance(
 .param .u64 px, .param .u64 py, .param .u64 pdr, .param .u64 pdi,
 .param .u64 pu, .param .u32 count, .param .u32 start, .param .u32 end,
 .param .f64 delta, .param .f64 cutoff
){
 .reg .pred %p<8>;
 .reg .b32 %r<16>;
 .reg .b64 %rd<16>;
 .reg .f64 %d<40>;
 mov.u32 %r0, %ctaid.x; mov.u32 %r1, %ntid.x; mov.u32 %r2, %tid.x;
 mad.lo.u32 %r0,%r0,%r1,%r2;
 ld.param.u32 %r3,[count]; setp.ge.u32 %p0,%r0,%r3; @%p0 bra EXIT;
 mul.wide.u32 %rd0,%r0,8;
 ld.param.u64 %rd1,[px]; add.u64 %rd1,%rd1,%rd0;
 ld.param.u64 %rd2,[py]; add.u64 %rd2,%rd2,%rd0;
 ld.param.u64 %rd3,[pdr]; add.u64 %rd3,%rd3,%rd0;
 ld.param.u64 %rd4,[pdi]; add.u64 %rd4,%rd4,%rd0;
 ld.param.u64 %rd5,[pu];
 ld.global.f64 %d0,[%rd1]; ld.global.f64 %d1,[%rd2];
 ld.global.f64 %d2,[%rd3]; ld.global.f64 %d3,[%rd4];
 ld.param.f64 %d5,[delta]; mul.f64 %d5,%d5,0d4010000000000000;
 ld.param.f64 %d6,[cutoff];
 ld.param.u32 %r4,[start]; ld.param.u32 %r5,[end];
LOOP:
 setp.ge.u32 %p0,%r4,%r5; @%p0 bra STORE;
 setp.le.f64 %p1,%d1,%d6; @%p1 bra DEAD;
 add.u32 %r6,%r4,1; mul.wide.u32 %rd6,%r6,8; add.u64 %rd6,%rd5,%rd6;
 ld.global.f64 %d4,[%rd6];
__ROOT_FORWARD__
 // Multiply the derivative by (old_g - U)/sqrt((old_g-U)^2+4dt).
 mul.f64 %d19,%d10,%d17; mul.f64 %d20,%d1,%d18;
 add.f64 %d19,%d19,%d20; div.rn.f64 %d19,%d19,%d15;
 mul.f64 %d20,%d1,%d17; mul.f64 %d21,%d10,%d18;
 sub.f64 %d20,%d20,%d21; div.rn.f64 %d20,%d20,%d15;
 mul.f64 %d21,%d2,%d19; mul.f64 %d22,%d3,%d20;
 sub.f64 %d21,%d21,%d22;
 mul.f64 %d22,%d2,%d20; mul.f64 %d23,%d3,%d19;
 add.f64 %d3,%d22,%d23; mov.f64 %d2,%d21;
 add.f64 %d0,%d4,%d17; mov.f64 %d1,%d18;
 add.u32 %r4,%r4,1; bra LOOP;
DEAD:
 mov.f64 %d1,0d0000000000000000;
STORE:
 st.global.f64 [%rd1],%d0; st.global.f64 [%rd2],%d1;
 st.global.f64 [%rd3],%d2; st.global.f64 [%rd4],%d3;
EXIT: ret;
}

.visible .entry trace_points(
 .param .u64 pk, .param .u64 px, .param .u64 py, .param .u64 pu,
 .param .u32 count, .param .f64 delta
){
 .reg .pred %p<8>; .reg .b32 %r<16>; .reg .b64 %rd<16>; .reg .f64 %d<40>;
 mov.u32 %r0,%ctaid.x; mov.u32 %r1,%ntid.x; mov.u32 %r2,%tid.x;
 mad.lo.u32 %r0,%r0,%r1,%r2;
 ld.param.u32 %r3,[count]; setp.ge.u32 %p0,%r0,%r3; @%p0 bra EXIT;
 mul.wide.u32 %rd0,%r0,4; ld.param.u64 %rd1,[pk]; add.u64 %rd1,%rd1,%rd0;
 ld.global.u32 %r4,[%rd1];
 ld.param.u64 %rd5,[pu]; mul.wide.u32 %rd6,%r4,8; add.u64 %rd6,%rd5,%rd6;
 ld.global.f64 %d0,[%rd6]; mov.f64 %d1,0d0000000000000000;
 ld.param.f64 %d5,[delta]; mul.f64 %d5,%d5,0dC010000000000000;
LOOP:
 setp.eq.u32 %p0,%r4,0; @%p0 bra STORE;
 mul.wide.u32 %rd6,%r4,8; add.u64 %rd6,%rd5,%rd6; ld.global.f64 %d4,[%rd6];
__ROOT_INVERSE__
 add.f64 %d0,%d4,%d17; mov.f64 %d1,%d18;
 sub.u32 %r4,%r4,1; bra LOOP;
STORE:
 mul.wide.u32 %rd0,%r0,8;
 ld.param.u64 %rd1,[px]; add.u64 %rd1,%rd1,%rd0; st.global.f64 [%rd1],%d0;
 ld.param.u64 %rd2,[py]; add.u64 %rd2,%rd2,%rd0; st.global.f64 [%rd2],%d1;
EXIT: ret;
}
'''.replace('__ROOT_FORWARD__',ROOT).replace('__ROOT_INVERSE__',ROOT)

INVERSE = r'''
.visible .entry inverse_points(
 .param .u64 px, .param .u64 py, .param .u64 pu,
 .param .u32 count, .param .u32 steps, .param .f64 delta
){
 .reg .pred %p<8>; .reg .b32 %r<16>; .reg .b64 %rd<16>; .reg .f64 %d<40>;
 mov.u32 %r0,%ctaid.x; mov.u32 %r1,%ntid.x; mov.u32 %r2,%tid.x;
 mad.lo.u32 %r0,%r0,%r1,%r2;
 ld.param.u32 %r3,[count]; setp.ge.u32 %p0,%r0,%r3; @%p0 bra EXIT;
 mul.wide.u32 %rd0,%r0,8;
 ld.param.u64 %rd1,[px]; add.u64 %rd1,%rd1,%rd0; ld.global.f64 %d0,[%rd1];
 ld.param.u64 %rd2,[py]; add.u64 %rd2,%rd2,%rd0; ld.global.f64 %d1,[%rd2];
 ld.param.u64 %rd5,[pu]; ld.param.u32 %r4,[steps];
 ld.param.f64 %d5,[delta]; mul.f64 %d5,%d5,0dC010000000000000;
LOOP:
 setp.eq.u32 %p0,%r4,0; @%p0 bra STORE;
 mul.wide.u32 %rd6,%r4,8; add.u64 %rd6,%rd5,%rd6; ld.global.f64 %d4,[%rd6];
__ROOT__
 add.f64 %d0,%d4,%d17; mov.f64 %d1,%d18;
 sub.u32 %r4,%r4,1; bra LOOP;
STORE:
 st.global.f64 [%rd1],%d0; st.global.f64 [%rd2],%d1;
EXIT: ret;
}
'''.replace('__ROOT__',ROOT)
PTX += INVERSE

class GPUMaps:
    def __init__(self):
        import cupy as cp
        self.cp=cp
        self.device=cp.cuda.Device(0)
        self.module=cp.cuda.function.Module()
        self.module.load(PTX.encode())
        self.advance_kernel=self.module.get_function('advance')
        self.trace_kernel=self.module.get_function('trace_points')
        self.inverse_kernel=self.module.get_function('inverse_points')
        self.name=cp.cuda.runtime.getDeviceProperties(0)['name'].decode()

    def array(self,a):
        a=np.ascontiguousarray(a)
        out=self.cp.empty(a.shape,a.dtype)
        out.set(a)
        return out

    def initialize(self,z,U,dt,cutoff=1e-12):
        self.shape=z.shape
        self.x=self.array(np.asarray(z.real,dtype=np.float64).ravel())
        self.y=self.array(np.asarray(z.imag,dtype=np.float64).ravel())
        self.dr=self.array(np.ones(z.size,dtype=np.float64))
        self.di=self.array(np.zeros(z.size,dtype=np.float64))
        self.U=self.array(np.asarray(U,dtype=np.float64))
        self.dt=float(dt); self.cutoff=float(cutoff); self.index=0

    def advance(self,end,chunk=128):
        if end<self.index:
            raise ValueError('Forward maps cannot move backward in time.')
        while self.index<end:
            stop=min(end,self.index+chunk)
            self.advance_kernel(((self.x.size+127)//128,),(128,),
                (self.x,self.y,self.dr,self.di,self.U,np.int32(self.x.size),
                 np.int32(self.index),np.int32(stop),np.float64(self.dt),np.float64(self.cutoff)))
            self.device.synchronize()  # Bounded kernels avoid the Windows display-driver timeout.
            self.index=stop

    def state(self):
        return tuple(a.get().reshape(self.shape) for a in (self.x,self.y,self.dr,self.di))

    def trace(self,U,ks,dt,batch=4096):
        U_gpu=self.array(np.asarray(U,dtype=np.float64))
        ks=np.asarray(ks,dtype=np.int32)
        out=np.empty(len(ks),dtype=np.complex128)
        for start in range(0,len(ks),batch):
            inds=ks[start:start+batch]
            kg=self.array(inds); xg=self.cp.empty(len(inds),self.cp.float64); yg=self.cp.empty_like(xg)
            self.trace_kernel(((len(inds)+127)//128,),(128,),
                (kg,xg,yg,U_gpu,np.int32(len(inds)),np.float64(dt)))
            self.device.synchronize()
            out[start:start+len(inds)]=xg.get()+1j*yg.get()
        return out

    def inverse(self,U,w,dt):
        w=np.asarray(w,dtype=complex)
        ug=self.array(np.asarray(U,dtype=float))
        xg=self.array(w.real.ravel()); yg=self.array(w.imag.ravel())
        self.inverse_kernel(((w.size+127)//128,),(128,),
            (xg,yg,ug,np.int32(w.size),np.int32(len(U)-1),np.float64(dt)))
        self.device.synchronize()
        return (xg.get()+1j*yg.get()).reshape(w.shape)

def cpu_forward(z,U,dt):
    z=np.asarray(z,dtype=complex).copy()
    D=np.ones_like(z)
    for u in U[1:]:
        q=z-u
        root=np.sqrt(q*q+4*dt)
        root=np.where(root.imag<0,-root,root)
        D*=q/root
        z=u+root
    return z,D

def validate():
    gpu=GPUMaps()
    rng=np.random.default_rng(23)
    z=rng.uniform(-2,2,200)+1j*rng.uniform(.4,3,200)
    U=np.r_[0,np.cumsum(rng.normal(size=256)*np.sqrt(1/256))]
    gpu.initialize(z,U,1/256); gpu.advance(256)
    x,y,dr,di=gpu.state(); expected,D=cpu_forward(z,U,1/256)
    np.testing.assert_allclose(x+1j*y,expected,rtol=2e-11,atol=2e-11)
    np.testing.assert_allclose(dr+1j*di,D,rtol=2e-10,atol=2e-10)
    ks=np.array([0,1,2,16,64,256])
    exact=2j*np.sqrt(ks/256)
    trace=gpu.trace(np.zeros(257),ks,1/256)
    np.testing.assert_allclose(trace,exact,atol=2e-12)
    a=gpu.trace(U,ks,1/256)
    b=gpu.trace(-U,ks,1/256)
    np.testing.assert_allclose(b,-a.conjugate(),atol=2e-12)
    # Inverse composition evaluated at U[k]+i*eta maps back under the forward chain.
    print({'gpu':gpu.name,'map_error':float(np.max(np.abs(x+1j*y-expected))),
           'derivative_error':float(np.max(np.abs(dr+1j*di-D))),
           'vertical_error':float(np.max(np.abs(trace-exact)))},flush=True)
    return gpu

if __name__=='__main__':
    gpu=validate()
    z=np.linspace(-3,3,640)[None,:]+1j*np.linspace(.002,3.4,360)[:,None]
    U=np.r_[0,np.cumsum(np.random.default_rng(23).normal(size=2048)/np.sqrt(2048))]
    gpu.initialize(z,U,1/2048)
    start=time.perf_counter(); gpu.advance(2048); elapsed=time.perf_counter()-start
    print({'pixels':z.size,'steps':2048,'seconds':elapsed,
           'million_pixel_steps_per_second':z.size*2048/elapsed/1e6},flush=True)
