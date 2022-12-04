import numpy as np
import matplotlib.pyplot as plt
import model
from mpl_toolkits.mplot3d import axes3d
import pandas as pd
import math

#パラメータ
ix,jx=model.ix,model.jx
kb=model.kb
Ti=model.Ti
year=3.16e7
tend=9.e0*year*1.e6
dtout=0.1e0*tend
nstop=1000000
t=0.0e0
tt=0.0e0
nd=1
safety=0.4e0
dtmin=1.e-10*tend
gm=5.e0/3.e0
t=0.e0
mH=1.67e-24
x=model.x
z=model.z
dx=model.dx
dxi=model.dxi
dz=model.dz
dzi=model.dzi
r=model.r
cosr=x/r
sinr=z/r
dfdx=np.zeros((ix,jx))
dfdz=np.zeros((ix,jx))
xm=x+dx/2.e0
#物理量初期条件
ro=model.ro
pr=model.pr
vx=model.vx
vz=model.vz

#cfl条件
def cfl(dt,ro,pr,vx,vz):
    dtq=np.zeros((ix,jx))
    cs2=np.zeros((ix,jx))
    v2=vx**2+vz**2
    cs2=gm*pr/ro
    dtq=safety*math.sqrt(dx**2+dz**2)/np.sqrt(v2+cs2)
    dt=np.amin(dtq)
    return dt

#流体計算
def culc(ro,vx,vz,pr):
    dro=np.zeros((ix,jx))
    dee=np.zeros((ix,jx))
    drx=np.zeros((ix,jx))
    drz=np.zeros((ix,jx))
    vv=vx**2+vz**2
    ee=pr/(gm-1.e0)+0.5e0*ro*vv
    fx=ro*vx
    fz=ro*vz
    ss=-fx/xm
    roh,dro=halflax(dro,ro,fx,fz,dt,ss)
    vv=vx**2+vz**2
    ep=pr*gm/(gm-1.e0)+0.5e0*ro*vv
    fx=ep*vx
    fz=ep*vz
    rx=ro*vx
    rz=ro*vz
    ss=-fx/xm
    eeh,dee=halflax(dee,ee,fx,fz,dt,ss)
    fx=ro*vx**2+pr
    fz=ro*vx*vz
    ss=-(ro*vx**2)/xm
    rxh,drx=halflax(drx,rx,fx,fz,dt,ss)
    fx=ro*vz*vx
    fz=ro*vz**2+pr
    ss=-fx/xm
    rzh,drz=halflax(drz,rz,fx,fz,dt,ss)
    vxh=np.zeros((ix,jx))
    vzh=np.zeros((ix,jx))
    vxh[0:-1,0:-1]=rxh[0:-1,0:-1]/roh[0:-1,0:-1]
    vzh[0:-1,0:-1]=rzh[0:-1,0:-1]/roh[0:-1,0:-1]
    vv=vxh**2+vzh**2
    prh=(gm-1)*(eeh-0.5e0*roh*vv)
    fx=roh*vxh
    fz=roh*vzh
    ss=-fx/xm
    dro=fulllax(dro,dt,fx,fz,ss)
    vv=vxh**2+vzh**2
    ep=prh*gm/(gm-1.e0)+0.5e0*roh*vv
    fx=ep*vxh
    fz=ep*vzh
    ss=-fx/xm
    dee=fulllax(dee,dt,fx,fz,ss)
    fx=roh*vxh**2+prh
    fz=roh*vxh*vzh
    ss=-roh*vxh**2/xm
    drx=fulllax(drx,dt,fx,fz,ss)
    fx=roh*vzh*vxh
    fz=roh*vzh**2+prh
    ss=-fx/xm
    drz=fulllax(drz,dt,fx,fz,ss)
    #print(dro)
    dro=visco(ro,dro,dt)
    dee=visco(ee,dee,dt)
    drx=visco(rx,drx,dt)
    drz=visco(rz,drz,dt)    
    rx=rx+drx
    rz=rz+drz
    ro[1:-1,1:-1]=ro[1:-1,1:-1]+dro[1:-1,1:-1]
    ee[1:-1,1:-1]=ee[1:-1,1:-1]+dee[1:-1,1:-1]
    vx[1:-1,1:-1]=rx[1:-1,1:-1]/ro[1:-1,1:-1]
    vz[1:-1,1:-1]=rz[1:-1,1:-1]/ro[1:-1,1:-1]
    vv=vx**2+vz**2
    pr[1:-1,1:-1]=(gm-1.e0)*(ee[1:-1,1:-1]-0.5e0*ro[1:-1,1:-1]*vv[1:-1,1:-1])
    return ro,vx,vz,pr

#lax-wendroff法１段階目
def halflax(du,u,fx,fz,dt,ss):
    uh=np.zeros((ix,jx))
    sh=np.zeros((ix,jx))
    sh[:-1,:-1]=0.5e0*(ss[1:,:-1]+ss[:-1,:-1]+ss[1:,1:]+ss[:-1,1:])/4.e0
    du[1:-1,1:-1]=du[1:-1,1:-1]-0.5e0*dt*(0.5e0*dxi*(fx[2:,1:-1]-fx[0:-2,1:-1]))-0.5e0*dt*(0.5e0*dzi*(fz[1:-1,2:]-fz[1:-1,0:-2]))+0.5e0*dt*ss[1:-1,1:-1]
    uh[:-1,:-1]=0.25e0*(u[1:,:-1]+u[:-1,:-1]+u[1:,1:]+u[:-1,1:])
    dfdx[:-1,:-1] = dxi*(fx[1:,:-1]-fx[:-1,:-1]+fx[1:,1:]-fx[:-1,1:])/2.e0
    dfdz[:-1,:-1] = dzi*(fz[:-1,1:]-fz[:-1,:-1]+fz[1:,1:]-fz[1:,:-1])/2.e0
    un=uh-dt*dfdx-dt*dfdz+dt*sh
    return un,du

#lax-wendroff法２段階目
def fulllax(du,dt,fx,fz,ss):
    sh=np.zeros((ix,jx))
    sh[1:-1,1:-1]=0.5e0*(0.5e0*ss[:-2,:-2]+0.5e0*ss[1:-1,:-2])+0.5e0*(0.5e0*ss[:-2,1:-1]+0.5e0*ss[1:-1,1:-1])
    dfdx[1:-1,1:-1] = dxi*(0.5e0*(fx[1:-1,:-2]-fx[:-2,:-2])+0.5e0*(fx[1:-1,1:-1]-fx[:-2,1:-1]))
    dfdz[1:-1,1:-1] = dzi*(0.5e0*(fz[:-2,1:-1]-fz[:-2,:-2])+0.5e0*(fz[1:-1,1:-1]-fz[1:-1,:-2]))
    du[1:-1,1:-1]= du[1:-1,1:-1]-0.5e0*dt*dfdx[1:-1,1:-1]-0.5e0*dt*dfdz[1:-1,1:-1]+0.5e0*dt*sh[1:-1,1:-1]
    return du
#人工粘性
def visco(u,du,dt):
    qav=0.3e0
    qx=np.zeros((ix,jx))
    qz=np.zeros((ix,jx))
    qx[1:-1,1:-1]=qav*dx*np.maximum(0,abs(vx[2:,1:-1]-vx[1:-1,1:-1])-1.0e-4)
    qz[1:-1,1:-1]=qav*dz*np.maximum(0,abs(vz[1:-1,2:]-vz[1:-1,1:-1])-1.0e-4)
    du[1:-1,1:-1]=du[1:-1,1:-1]+dt*(dxi*(qx[1:-1,1:-1]*dxi*(u[2:,1:-1]-u[1:-1,1:-1])-qx[:-2,1:-1]*dxi*(u[1:-1,1:-1]-u[:-2,1:-1])))+dt*(dzi*(qz[1:-1,1:-1]*dzi*(u[1:-1,2:]-u[1:-1,1:-1])-qz[1:-1,:-2]*dzi*(u[1:-1,1:-1]-u[1:-1,:-2])))
    return du

#境界条件
def bnd(u):
    u[0,:]=u[1,:]
    u[ix-1,:]=u[ix-2,:]
    u[:,0]=u[:,1]
    u[:,jx-1]=u[:,jx-2]
    return u

#時間積分
for i in range(nstop):
    nd=nd+1
    dt=1.e20
    dt=cfl(dt,ro,pr,vx,vz)
    print(dt)
    if dt<dtmin:
        break
    tt=t
    t=t+dt
    nt1=int(tt/dtout)
    nt2=int(t/dtout)
    #銀河中心から供給するガス
    ro[0,0]=9.3*1.67e-27
    v=1.1e8
    vx[0,0]=v*cosr[0,0]
    vz[0,0]=v*sinr[0,0]
    pr[0,0]=ro[0,0]*kb*Ti/(1.67e-24)
    #流体計算
    ro,vx,vz,pr=culc(ro,vx,vz,pr)
    #境界条件
    ro=bnd(ro)
    vx=bnd(vx)
    vz=bnd(vz)
    pr=bnd(pr)
    #終わる時間の1/10ごとに出力
    if nt1<nt2:
        T=pr/(ro*kb)
        df=pd.DataFrame(data=pr,columns=z[0,:],index=x[:,0])
        df.to_csv('pr'+str(nd)+'Myr.csv')
        df=pd.DataFrame(data=vx,columns=z[0,:],index=x[:,0])
        df.to_csv('vx'+str(nd)+'Myr.csv')
        df=pd.DataFrame(data=vz,columns=z[0,:],index=x[:,0])
        df.to_csv('vz'+str(nd)+'Myr.csv')
        df=pd.DataFrame(data=T,columns=z[0,:],index=x[:,0])
        df.to_csv('T'+str(nd)+'Myr.csv')
        df=pd.DataFrame(data=ro,columns=z[0,:],index=x[:,0])
        df.to_csv('ro'+str(nd)+'Myr.csv')
        nd=nd+1
        #tがtendを超えたら終了
        if int(t/tend)==1:
            break

#出力
ro=ro/mH
T=pr/(ro*kb)
df=pd.DataFrame(data=ro,columns=z[0,:],index=x[:,0])
df.to_csv('')
df=pd.DataFrame(data=pr,columns=z[0,:],index=x[:,0])
df.to_csv('')
df=pd.DataFrame(data=vx,columns=z[0,:],index=x[:,0])
df.to_csv('')
df=pd.DataFrame(data=vz,columns=z[0,:],index=x[:,0])
df.to_csv('')
df=pd.DataFrame(data=T,columns=z[0,:],index=x[:,0])
df.to_csv('')
print(t)
plt.contour(x,z,ro)
plt.gca().set_aspect('equal')
plt.show()
