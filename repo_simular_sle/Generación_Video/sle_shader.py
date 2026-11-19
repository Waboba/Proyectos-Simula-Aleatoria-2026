"""Analytic texture shader; only drawing uses float32, maps remain float64."""
import struct
import numpy as np

class Builder:
    def __init__(self): self.lines=[]; self.n=0
    def val(self,x): return x if isinstance(x,V) else V(self, f'0f{struct.unpack("I",struct.pack("f",float(x)))[0]:08X}')
    def op(self,ins,*args):
        name=f'%f{self.n}';self.n+=1
        self.lines.append(f'{ins} {name}, '+', '.join(self.val(a).r for a in args)+';')
        return V(self,name)

class V:
    def __init__(self,b,r):self.b=b;self.r=r
    def __add__(self,x):return self.b.op('add.f32',self,x)
    __radd__=__add__
    def __sub__(self,x):return self.b.op('sub.f32',self,x)
    def __rsub__(self,x):return self.b.op('sub.f32',x,self)
    def __mul__(self,x):return self.b.op('mul.f32',self,x)
    __rmul__=__mul__
    def __truediv__(self,x):return self.b.op('div.approx.f32',self,x)
    def __neg__(self):return self.b.op('neg.f32',self)

def shader_ptx():
    b=Builder()
    x,y,dr,di,pixel,blend=(V(b,s) for s in ('%x','%y','%dr','%di','%pixel','%blend'))
    def clamp(v,lo=0,hi=1):return b.op('min.f32',b.op('max.f32',v,lo),hi)
    def abs_(v):return b.op('abs.f32',v)
    def floor(v):return b.op('cvt.rmi.f32.f32',v)
    def sqrt(v):return b.op('sqrt.approx.f32',v)
    def sin(v):return b.op('sin.approx.f32',v)
    def cos(v):return b.op('cos.approx.f32',v)
    def exp2(v):return b.op('ex2.approx.f32',v)
    def log2(v):return b.op('lg2.approx.f32',v)
    def smooth(v):v=clamp(v);return v*v*(3-2*v)
    def fract(v):return v-floor(v)
    norm=sqrt(dr*dr+di*di+1e-30)
    footprint=norm*pixel
    # Footprint filtering suppresses unresolved grid frequencies rather than inventing detail.
    fade=clamp(1-footprint/0.12)
    width=0.003+0.7*footprint
    distx=abs_(fract(x/0.24+0.5)-0.5)*0.24
    disty=abs_(fract(y/0.24+0.5)-0.5)*0.24
    gx=exp2(-3*(distx/width)*(distx/width))*fade
    gy=exp2(-3*(disty/width)*(disty/width))*fade
    xx=x-2.2*floor(x/2.2+0.5)
    yy=y-0.98-2.2*floor(y/2.2)
    rr=sqrt(xx*xx+yy*yy+1e-12)
    c=xx/rr;s=yy/rr
    c2=2*c*c-1;s2=2*s*c
    c4=2*c2*c2-1;s4=2*s2*c2
    c8=2*c4*c4-1;s8=2*s4*c4
    radius=0.70*(1+0.20*c8)
    edge=0.009+footprint
    inside=smooth((radius-rr)/edge+0.5)
    rim=exp2(-2*((rr-radius)/(0.007+footprint))*((rr-radius)/(0.007+footprint)))
    ring=exp2(-12*sin(35*rr+1.7*c8)*sin(35*rr+1.7*c8))
    spoke=exp2(-25*s8*s8)
    core=exp2(-30*rr*rr)
    detail=clamp(1-footprint/0.04)
    light=0.45+detail*(0.34*ring+0.34*spoke)+0.30*core
    red=clamp(0.57+0.36*c+0.10*cos(14*rr))
    green=clamp(0.53+0.34*(-0.5*c+0.8660254*s)+0.10*cos(14*rr+2.094))
    blue=clamp(0.62+0.34*(-0.5*c-0.8660254*s)+0.10*cos(14*rr+4.188))
    bg=(0.010+0.025*gx+0.10*gy,0.017+0.18*gx+0.025*gy,0.040+0.23*gx+0.16*gy)
    rose=(red*light,green*light,blue*light)
    rgb=[bg[i]*(1-inside)+rose[i]*inside+rim*(0.15+0.45*rose[i]) for i in range(3)]
    # log2 |f'_t(g_t(z))| = -log2 |g'_t(z)|, fixed [-3,3] color scale.
    sigma=-log2(norm)
    hot=clamp(sigma/3);cold=clamp(-sigma/3)
    neutral=(0.10,0.14,0.22)
    heat=[neutral[i]+hot*((0.98,0.55,0.09)[i]-neutral[i])+
          cold*((0.04,0.52,0.97)[i]-neutral[i])+(gx+gy)*0.055 for i in range(3)]
    rgb=[clamp(rgb[i]*(1-blend)+heat[i]*blend) for i in range(3)]
    for i,v in enumerate(rgb):
        v=v*255
        b.lines.append(f'cvt.rni.u32.f32 %q{i}, {v.r};')
        b.lines.append(f'st.global.u8 [%out+{i}], %q{i};')
    prefix=r'''.version 6.4
.target sm_75
.address_size 64
.visible .entry shade(
 .param .u64 px,.param .u64 py,.param .u64 pdr,.param .u64 pdi,
 .param .u64 pout,.param .u32 count,.param .f32 pix,.param .f32 mix
){
 .reg .pred %p<4>; .reg .b32 %i<6>; .reg .b32 %q<3>; .reg .b64 %a<8>;
 .reg .b64 %out; .reg .f64 %d<4>;
 .reg .f32 %x,%y,%dr,%di,%pixel,%blend;
 .reg .f32 %f<NUM_REGS>;
 mov.u32 %i0,%ctaid.x;mov.u32 %i1,%ntid.x;mov.u32 %i2,%tid.x;
 mad.lo.u32 %i0,%i0,%i1,%i2;ld.param.u32 %i3,[count];
 setp.ge.u32 %p0,%i0,%i3;@%p0 bra EXIT;
 mul.wide.u32 %a0,%i0,8;
 ld.param.u64 %a1,[px];add.u64 %a1,%a1,%a0;ld.global.f64 %d0,[%a1];
 ld.param.u64 %a1,[py];add.u64 %a1,%a1,%a0;ld.global.f64 %d1,[%a1];
 ld.param.u64 %a1,[pdr];add.u64 %a1,%a1,%a0;ld.global.f64 %d2,[%a1];
 ld.param.u64 %a1,[pdi];add.u64 %a1,%a1,%a0;ld.global.f64 %d3,[%a1];
 mul.wide.u32 %a0,%i0,3;ld.param.u64 %out,[pout];add.u64 %out,%out,%a0;
 setp.le.f64 %p1,%d1,0d3D719799812DEA11;@%p1 bra DARK;
 setp.nan.f64 %p1,%d0,%d2;@%p1 bra DARK;
 cvt.rn.f32.f64 %x,%d0;cvt.rn.f32.f64 %y,%d1;
 cvt.rn.f32.f64 %dr,%d2;cvt.rn.f32.f64 %di,%d3;
 ld.param.f32 %pixel,[pix];ld.param.f32 %blend,[mix];
'''.replace('NUM_REGS',str(b.n+1))
    return prefix+'\n'.join(b.lines)+r'''
 bra EXIT;
DARK:
 mov.u32 %q0,2;st.global.u8 [%out],%q0;
 mov.u32 %q0,4;st.global.u8 [%out+1],%q0;
 mov.u32 %q0,9;st.global.u8 [%out+2],%q0;
EXIT:ret;
}
'''

class Shader:
    def __init__(self,gpu):
        self.gpu=gpu;self.module=gpu.cp.cuda.function.Module()
        self.module.load(shader_ptx().encode())
        self.kernel=self.module.get_function('shade')
        self.rgb=None
    def draw(self,pixel_size,blend=0):
        g=self.gpu
        if self.rgb is None or self.rgb.shape[:2]!=g.shape:
            self.rgb=g.cp.empty((*g.shape,3),g.cp.uint8)
        self.kernel(((g.x.size+255)//256,),(256,),
            (g.x,g.y,g.dr,g.di,self.rgb,np.int32(g.x.size),np.float32(pixel_size),np.float32(blend)))
        g.device.synchronize()
        return self.rgb.get()

if __name__=='__main__':
    from pathlib import Path
    from PIL import Image
    from sle_gpu import GPUMaps
    from calibrar import browniano_maestro
    gpu=GPUMaps();B=browniano_maestro();n=16384
    U=np.sqrt(6)*B[::(len(B)-1)//n]
    xmin,xmax,ymin,ymax=-3.5,2.1,0,3.15
    w,h=1280,720
    z=np.linspace(xmin,xmax,w)[None,:]+1j*np.linspace(ymax,ymin+0.002,h)[:,None]
    gpu.initialize(z,U,1/n)
    shader=Shader(gpu)
    Image.fromarray(shader.draw((xmax-xmin)/w)).save(Path(__file__).parent/'textura_inicial.png')
    gpu.advance(n)
    Image.fromarray(shader.draw((xmax-xmin)/w)).save(Path(__file__).parent/'textura_final.png')
    Image.fromarray(shader.draw((xmax-xmin)/w,1)).save(Path(__file__).parent/'dilatacion_final.png')
    print('Shader listo.',flush=True)
