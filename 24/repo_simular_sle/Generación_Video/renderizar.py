"""Tres películas reproducibles: textura conforme, dilatación y relieve escalar."""
from pathlib import Path
import argparse,json,time,subprocess,os
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg
from sle_gpu import GPUMaps
from sle_shader import Shader
from preparar_trazas import prepare,STEPS

HERE=Path(__file__).resolve().parent
BG=(3,5,12)

def font(size,bold=False):
    candidates=[Path('C:/Windows/Fonts')/('segoeuib.ttf' if bold else 'segoeui.ttf'),
                Path('/usr/share/fonts/truetype/dejavu')/('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf')]
    for p in candidates:
        if p.exists():return ImageFont.truetype(str(p),max(8,int(size)))
    return ImageFont.load_default(size=max(8,int(size)))

class Writer:
    def __init__(self,path,width,height,fps,bitrate='3600k',codec='h264_nvenc'):
        self.path=path;self.frames=0
        self.log=open(path.with_suffix('.ffmpeg.log'),'w',encoding='utf-8')
        quality=['-preset','p5','-rc','vbr','-b:v',bitrate,'-maxrate',bitrate,'-bufsize','7200k'] if codec=='h264_nvenc' else ['-preset','fast','-b:v',bitrate]
        cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','warning','-y','-f','rawvideo','-vcodec','rawvideo',
             '-s',f'{width}x{height}','-pix_fmt','rgb24','-r',str(fps),'-i','-','-an','-c:v',codec,*quality,
             '-pix_fmt','yuv420p','-movflags','+faststart',str(path)]
        self.process=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=self.log,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    def add(self,im):
        self.process.stdin.write(np.asarray(im,dtype=np.uint8).tobytes());self.frames+=1
    def close(self):
        self.process.stdin.close();code=self.process.wait();self.log.close()
        if code:raise RuntimeError(f'FFmpeg falló ({code}): {self.path.with_suffix(".ffmpeg.log")}')

class TraceLayer:
    def __init__(self,data,w,h):
        self.w=w;self.h=h;self.data=data;self.done=0
        self.layer=Image.new('RGBA',(w,h),(0,0,0,0));self.draw=ImageDraw.Draw(self.layer)
        xmin,xmax,ymin,ymax=data['bounds'];z=data['gamma']
        self.points=np.c_[(z.real-xmin)/(xmax-xmin)*w,(ymax-z.imag)/(ymax-ymin)*h]
        self.gaps=np.abs(np.diff(z));self.limit=8*(xmax-xmin)/3840
    def update(self,k):
        end=int(np.searchsorted(self.data['indices'],k,side='right'))
        scale=self.w/1920
        # No long straight bridges across jumps of the finite slit approximation.
        for j in range(max(1,self.done),end):
            a=tuple(self.points[j-1]);b=tuple(self.points[j])
            if self.gaps[j-1]<=self.limit:
                for thickness,color in ((5,(60,133,178,70)),(2.4,(140,218,243,160)),(.9,(246,251,255,245))):
                    self.draw.line((a,b),fill=color,width=max(1,round(thickness*scale)))
            else:
                r=max(.5,.5*scale);x,y=b
                self.draw.ellipse((x-r,y-r,x+r,y+r),fill=(231,249,255,230))
        self.done=end
    def compose(self,rgb,k):
        self.update(k);im=Image.fromarray(rgb).convert('RGBA');im.alpha_composite(self.layer)
        if self.done:
            x,y=self.points[self.done-1];r=4*self.w/1920
            ImageDraw.Draw(im).ellipse((x-r,y-r,x+r,y+r),fill=(255,212,108,255))
        return im.convert('RGB')

def decorate(im,kappa,t,phase,heat=False):
    w,h=im.size;s=w/1920;d=ImageDraw.Draw(im)
    d.rectangle((0,0,w,88*s),fill=BG)
    d.text((38*s,15*s),f'SLE   κ = {kappa}',font=font(33*s,True),fill=(243,248,255))
    d.text((330*s,23*s),phase,font=font(23*s),fill=(150,191,218))
    d.text((w-170*s,24*s),f't = {t:0.3f}',font=font(22*s),fill=(236,242,248))
    d.rectangle((0,h-52*s,w,h),fill=BG)
    if heat:
        label='Dilatación local:  azul = contracción · naranja = expansión'
        d.text((38*s,h-39*s),label,font=font(19*s),fill=(193,209,229))
        # Fixed scale, matching shader; clipped tails are denoted explicitly.
        x0=int(w-540*s);y0=int(h-37*s);bw=int(220*s);bh=max(3,int(8*s))
        for j in range(bw):
            v=6*j/(bw-1)-3;base=np.array([.10,.14,.22]);end=np.array([.98,.55,.09]) if v>=0 else np.array([.04,.52,.97])
            color=tuple((255*(base+abs(v)/3*(end-base))).astype(int))
            d.line((x0+j,y0,x0+j,y0+bh),fill=color)
        d.text((x0,y0+10*s),'≤ −3          0          ≥ +3',font=font(13*s),fill=(187,201,219))
        d.text((w-270*s,h-35*s),'log₂ |f′|',font=font(18*s),fill=(232,238,248))
    else:
        d.text((38*s,h-39*s),'Cada color conserva su punto de origen · fₜ = gₜ⁻¹',font=font(19*s),fill=(193,209,229))
        d.text((w-355*s,h-39*s),'Misma trayectoria browniana',font=font(18*s),fill=(132,159,184))
    return im

def crop_zoom(im,amount,data):
    if amount<=0:return im.copy()
    w,h=im.size;xmin,xmax,ymin,ymax=data['bounds'];z=data['gamma']
    cx=float((np.quantile(z.real,.45)-xmin)/(xmax-xmin)*w)
    cy=float((ymax-np.quantile(z.imag,.65))/(ymax-ymin)*h)
    scale=1+0.8*amount;cw=w/scale;ch=h/scale
    left=np.clip(cx-cw/2,0,w-cw);top=np.clip(cy-ch/2,0,h-ch)
    return im.crop((round(left),round(top),round(left+cw),round(top+ch))).resize((w,h),Image.Resampling.LANCZOS)

def relief_frames(field,data,width,height,count):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x,y,dr,di=field
    sigma=-np.log2(np.maximum(np.hypot(dr,di),1e-30));valid=(y>1e-12)&np.isfinite(sigma)
    hh,ww=x.shape;xmin,xmax,ymin,ymax=data['bounds']
    X,Y=np.meshgrid(np.linspace(xmin,xmax,ww),np.linspace(ymax,ymin,hh))
    Z=np.where(valid,np.clip(sigma,-3,3),np.nan)
    q=np.nan_to_num(Z);neutral=np.array([.10,.14,.22]);cold=np.array([.04,.52,.97]);hot=np.array([.98,.55,.09])
    colors=neutral+np.maximum(q[...,None],0)/3*(hot-neutral)+np.maximum(-q[...,None],0)/3*(cold-neutral)
    fig=plt.figure(figsize=(width/100,height/100),dpi=100,facecolor=np.array(BG)/255)
    ax=fig.add_axes([.01,.015,.98,.91],projection='3d',facecolor=np.array(BG)/255)
    ax.plot_surface(X,Y,Z,facecolors=np.clip(colors,0,1),rstride=1,cstride=1,linewidth=.12,
                    edgecolor=(.25,.45,.60,.13),antialiased=True,shade=False)
    ax.set(xlim=(xmin,xmax),ylim=(ymin,ymax),zlim=(-3.1,3.1))
    ax.set_box_aspect((xmax-xmin,ymax-ymin,2.7));ax.set_proj_type('persp',focal_length=1.2)
    ax.set_xlabel('Re z',color='#acc0d5',fontsize=12*width/1920,labelpad=9)
    ax.set_ylabel('Im z',color='#acc0d5',fontsize=12*width/1920,labelpad=9)
    ax.set_zlabel('log₂ |f′|',color='#acc0d5',fontsize=12*width/1920,labelpad=9)
    for a in (ax.xaxis,ax.yaxis,ax.zaxis):
        a.set_pane_color((.025,.04,.07,.35));a._axinfo['grid']['color']=(.22,.33,.44,.24)
        a.line.set_color('#496175')
    ax.tick_params(colors='#8199b0',labelsize=10*width/1920);ax.set_zticks([-3,0,3])
    for j in range(count):
        a=j/max(1,count-1);ax.view_init(elev=32+5*np.sin(np.pi*a),azim=-67+55*a)
        fig.canvas.draw();im=Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy())
        yield im
    plt.close(fig)

def render(kappa,width=3840,height=2160,fps=24,duration=30,preview=False,codec='h264_nvenc'):
    if width*9!=height*16 or width%2 or height%2:
        raise ValueError('Usa dimensiones pares con relación 16:9 para conservar la escala del plano complejo.')
    if fps<=0 or duration<6:
        raise ValueError('Se requiere fps > 0 y una duración de al menos 6 segundos.')
    start=time.perf_counter();data=prepare(kappa);n=STEPS[kappa]
    stem=f'SLE_kappa_{kappa}'+('_prueba' if preview else '_4K')
    path=HERE/(stem+'.mp4');tmp=HERE/(stem+'.partial.mp4')
    gpu=GPUMaps();xmin,xmax,ymin,ymax=data['bounds'];pixel=(xmax-xmin)/width
    z=(xmin+(np.arange(width)+.5)*pixel)[None,:]+1j*(ymax-(np.arange(height)+.5)*pixel)[:,None]
    gpu.initialize(z,data['U'],1/n);del z
    shader=Shader(gpu);trace=TraceLayer(data,width,height)
    writer=Writer(tmp,width,height,fps,bitrate='1400k' if preview else '3600k',codec=codec)
    total=int(round(duration*fps));initial=round(total/30);grow=round(total*18/30);zoom=round(total*3/30);fade=round(total*2/30)
    three=total-initial-grow-zoom-fade
    print(f'INICIO k={kappa}, {width}x{height}, {total} fotogramas, N={n}',flush=True)
    rgb=shader.draw(pixel)
    initial_frame=decorate(trace.compose(rgb,0),kappa,0,'Geometría conforme')
    for _ in range(initial):writer.add(initial_frame)
    last_report=time.perf_counter()
    field_seconds=0
    for j in range(grow):
        k=round(n*(j+1)/grow);tic=time.perf_counter();gpu.advance(k,chunk=64)
        field_seconds+=time.perf_counter()-tic
        rgb=shader.draw(pixel);im=decorate(trace.compose(rgb,k),kappa,k/n,'Geometría conforme')
        writer.add(im)
        if j in (grow//3,2*grow//3,grow-1):
            im.resize((1920,1080),Image.Resampling.LANCZOS).save(HERE/f'{stem}_t{k/n:.2f}.jpg',quality=94)
        if time.perf_counter()-last_report>20 or j==grow-1:
            print(f'k={kappa} crecimiento {(j+1)/grow:.0%} · {time.perf_counter()-start:.1f} s',flush=True);last_report=time.perf_counter()
    texture=trace.compose(shader.draw(pixel),n)
    heat=trace.compose(shader.draw(pixel,1),n)
    decorate(texture.copy(),kappa,1,'Geometría conforme').save(HERE/f'{stem}_poster.png')
    for j in range(zoom):
        a=(j+1)/max(1,zoom);a=a*a*(3-2*a)
        writer.add(decorate(crop_zoom(texture,a,data),kappa,1,'Detalle de la deformación'))
    tz=crop_zoom(texture,1,data);hz=crop_zoom(heat,1,data)
    for j in range(fade):
        a=(j+1)/max(1,fade)
        writer.add(decorate(Image.blend(tz,hz,a),kappa,1,'Cambio local de escala',heat=True))
    decorate(hz.copy(),kappa,1,'Cambio local de escala',heat=True).resize((1920,1080),Image.Resampling.LANCZOS).save(HERE/f'{stem}_dilatacion.jpg',quality=94)
    # A coarser surface is intentional: the 3D height is a scalar visualization,
    # not an additional spatial dimension of the SLE trace.
    indices_y=np.linspace(0,height-1,85).astype(int);indices_x=np.linspace(0,width-1,151).astype(int)
    full=gpu.state();field=tuple(a[np.ix_(indices_y,indices_x)] for a in full)
    del full
    np.savez_compressed(HERE/f'campo_k{kappa}.npz',x=field[0],y=field[1],dr=field[2],di=field[3],bounds=data['bounds'])
    print(f'k={kappa}: relieve 3D ({three} fotogramas)',flush=True)
    for j,frame in enumerate(relief_frames(field,data,width,height,three)):
        frame=decorate(frame,kappa,1,'Relieve de la dilatación',heat=True)
        d=ImageDraw.Draw(frame);d.text((width*.32,height*.13),'La altura representa el cambio de escala',font=font(20*width/1920),fill=(186,209,230))
        if j==three//2:frame.resize((1920,1080),Image.Resampling.LANCZOS).save(HERE/f'{stem}_relieve.jpg',quality=94)
        writer.add(frame)
        if time.perf_counter()-last_report>20:
            print(f'k={kappa} relieve {(j+1)/three:.0%} · {time.perf_counter()-start:.1f} s',flush=True);last_report=time.perf_counter()
    writer.close();tmp.replace(path)
    report={'kappa':kappa,'N':n,'width':width,'height':height,'fps':fps,'frames':writer.frames,'duration':writer.frames/fps,
            'field_seconds':field_seconds,'total_seconds':time.perf_counter()-start,'size_bytes':path.stat().st_size,
            'spatial_resolution':'native pixel grid; no upscaling of field','trace':json.loads(str(data['stats']))}
    path.with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('TERMINADO '+json.dumps(report),flush=True)
    return path

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kappa',type=int,nargs='+',default=[2,6,20],choices=[2,6,20])
    p.add_argument('--preview',action='store_true');p.add_argument('--codec',default='h264_nvenc',choices=['h264_nvenc','libx264']);p.add_argument('--force',action='store_true')
    args=p.parse_args()
    for k in args.kappa:
        target=HERE/(f'SLE_kappa_{k}'+('_prueba' if args.preview else '_4K')+'.mp4')
        if target.exists() and not args.force:print(f'Existe {target.name}; se conserva.',flush=True);continue
        render(k,width=960 if args.preview else 3840,height=540 if args.preview else 2160,
               fps=12 if args.preview else 24,duration=6 if args.preview else 30,preview=args.preview,codec=args.codec)
