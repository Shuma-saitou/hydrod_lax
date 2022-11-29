import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd

pc=3.09e18
pi=3.14159265359e0
mp=1.67e-24
kb=1.38e-16
gm=5.e0/3.e0
Etot=1.e51
ix,jx=20002,3
gm=5/3e0
mH=1.67e-24   
Ti=0.2e3*1.6e-12/kb 
vhalo=131.5e5
Grav=6.67e-8
dh = 12.*1.e3*3.09e18
Mb = 3.4e10*1.989e33
db = 0.7*1.e3*3.09e18
Md = 1.e11*1.989e33
ad = 6.5*1.e3*3.09e18
bd = 0.26*1.e3*3.09e18
a=kb*Ti
xmin=1.e3*3.09e18
xmax=15e3*3.09e18
zmin=xmin
zmax=xmax
dx=(xmax-xmin)/(ix-1)
dz=(zmax-zmin)/(jx-1)
dxi=1/dx
dzi=1/dz
xx=np.linspace(xmin,xmax,ix)
zz=np.linspace(zmin,zmax,jx)
z,x=np.meshgrid(zz,xx)
ro=np.zeros((ix,jx))
pr=np.zeros((ix,jx))
vx=np.zeros((ix,jx))
vz=np.zeros((ix,jx))
potential=np.zeros((ix,jx))
ro0=0.024e0*1.67e-24
potential=(vhalo**2)*np.log(x**2+dh**2)-Grav*Mb/(x+db)-Grav*Md/(ad+np.sqrt(x**2+bd**2))
ro=ro0*np.exp(-mH*(potential-potential[0,0])*0.6/a)
pr=ro*kb*Ti/(mH*0.6)
#print(ro)
ss=np.sqrt(x**2+z**2)
#print(ss)
df=pd.DataFrame(data=pr,columns=z[0,:],index=x[:,0])
df.to_csv('/home/theoretical/ダウンロード/canspython/initial'+'.csv')#,columns=[1.236000e+17])


#plt.contour(x,z,pr)
#plt.gca().set_aspect('equal')
#plt.show()
#for i in range(ix):
#    for j in range(jx):
#        ro[i,j]=1.e0
#        vx[i,j]=0.e0
#        vz[i,j]=0.e0
#        ss=math.sqrt(x(i)**2+z(j)**2)
#        pr[i,j]=prism+(1.e0-prism)*math.exp(-(ss/wexp)**2)
#df=pd.DataFrame(data=pr,columns=x[0,:],index=z[:,1])
#print(df)
