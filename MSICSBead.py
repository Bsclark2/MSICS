# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 15:01:18 2023

@author: bencl
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from matplotlib import cm
from matplotlib.colors import hsv_to_rgb
from mpl_toolkits.mplot3d import Axes3D
#%%
imagesize=100 #number of pixels per side of image
rowstart=0 #the starting row of the region of intrest
collumnstart=0 #the starting column region of intrest
ROIheight=100 #the number of rows in the region of intrest
ROIlength=100 #the number of collumns in the region of intrest
Sfactor=15
Cfactor=15
#%%
pixelsize=30.0e-9
taup=1.7e-6
ecc=3.333
w0=0.353e-6
def GDL(inp,D,N):
    xi,psi,taud=inp
    return 0.3535*1.0/N*np.exp(-1.0*((pixelsize*xi)**2+(pixelsize*psi)**2)/(w0**2+4*D*(np.abs(taud))))/(1.0+4*D*(np.abs(taud))/w0**2)/np.sqrt(1.0+4*D*(np.abs(taud))/ecc**2/w0**2)
def flat(lis):
    flatList = []
    # Iterate with outer list
    for element in lis:
        if type(element) is np.ndarray:
            # Check if type is list than iterate through the sublist
            for item in element:
                flatList.append(item)
        else:
            flatList.append(element)
    return flatList
#%%
Gmstack=[]
for k in range(1,501):
    filename='E:\Ben\MSICS Frames\Bead\Bead_'+f'{k:03}'+'.txt'
    intdata = pd.read_csv(filename,sep="\t",header=None,skiprows=3,skipfooter=imagesize+3,engine='python',encoding='unicode_escape')
    intdata=intdata.to_numpy()
    Odata=intdata
    Gmstack.append(Odata)
Gmstack=np.array(Gmstack)
#%%
shape=np.shape(Gmstack)
immobilefactor=10
immobilefactor2=5
AF1=np.array([])
ACFlen=len(Gmstack)-immobilefactor
for i in range(immobilefactor2,ACFlen+immobilefactor2):
    AF1=np.append(AF1,np.mean(Gmstack[i-immobilefactor2:i+immobilefactor2,:,:],axis=0))
AF1=np.reshape(AF1,(shape[0]-immobilefactor,shape[1],shape[2]))
#%%
Gm=[]
hp=int(len(Odata)/2)
braF=np.average(np.average(Gmstack[immobilefactor2:ACFlen+immobilefactor2,:,:],axis=1),axis=1)
variancearr=[]
#%%
for i in range(-Sfactor,Sfactor+1):
    for j in range(-Sfactor,Sfactor+1):
        if i%2==1:
            if i>=0:
                if j>=0:
                    i=abs(i)
                    j=abs(j)
                    Sdata=Gmstack[immobilefactor2:ACFlen+immobilefactor2,rowstart:rowstart+ROIheight-i,collumnstart:collumnstart+ROIlength-j]
                    NSdata=Gmstack[immobilefactor2:ACFlen+immobilefactor2,rowstart+i:rowstart+ROIheight,collumnstart+j:collumnstart+ROIlength]
                    Sdata[:,1::2, :] = Sdata[:,1::2, ::-1]
                    NSdata[:,1::2, :] = NSdata[:,1::2, ::-1]
                    AF2=np.average(np.multiply(Sdata,NSdata),axis=1)
                    for k in range(len(braF)):
                        AF2[k,:]=AF2[k,:]/braF[k]**2
                    variancearr.append(AF2)
                    AF2=np.average(AF2,axis=0)
                else:
                    i=abs(i)
                    j=abs(j)
                    Sdata=Gmstack[immobilefactor2:ACFlen+immobilefactor2,rowstart:rowstart+ROIheight-i,collumnstart+j:collumnstart+ROIlength]
                    NSdata=Gmstack[immobilefactor2:ACFlen+immobilefactor2,rowstart+i:rowstart+ROIheight,collumnstart:collumnstart+ROIlength-j]
                    Sdata[:,1::2, :] = Sdata[:,1::2, ::-1]
                    NSdata[:,1::2, :] = NSdata[:,1::2, ::-1]
                    AF2=np.average(np.multiply(Sdata,NSdata),axis=1)
                    for k in range(len(braF)):
                        AF2[k,:]=AF2[k,:]/braF[k]**2
                    variancearr.append(AF2)
                    AF2=np.average(AF2,axis=0)
            else:
                if j>=0:
                    i=abs(i)
                    j=abs(j)
                    Sdata=Gmstack[immobilefactor2:ACFlen+immobilefactor2,rowstart+i:rowstart+ROIheight,collumnstart:collumnstart+ROIlength-j]
                    NSdata=Gmstack[immobilefactor2:ACFlen+immobilefactor2,rowstart:rowstart+ROIheight-i,collumnstart+j:collumnstart+ROIlength]
                    Sdata[:,1::2, :] = Sdata[:,1::2, ::-1]
                    NSdata[:,1::2, :] = NSdata[:,1::2, ::-1]
                    AF2=np.average(np.multiply(Sdata,NSdata),axis=1)
                    for k in range(len(braF)):
                        AF2[k,:]=AF2[k,:]/braF[k]**2
                    variancearr.append(AF2)
                    AF2=np.average(AF2,axis=0)
                else:
                    i=abs(i)
                    j=abs(j)
                    Sdata=Gmstack[immobilefactor2:ACFlen+immobilefactor2,rowstart+i:rowstart+ROIheight,collumnstart+j:collumnstart+ROIlength]
                    NSdata=Gmstack[immobilefactor2:ACFlen+immobilefactor2,rowstart:rowstart+ROIheight-i,collumnstart:collumnstart+ROIlength-j]
                    Sdata[:,1::2, :] = Sdata[:,1::2, ::-1]
                    NSdata[:,1::2, :] = NSdata[:,1::2, ::-1]
                    AF2=np.average(np.multiply(Sdata,NSdata),axis=1)
                    for k in range(len(braF)):
                        AF2[k,:]=AF2[k,:]/braF[k]**2
                    variancearr.append(AF2)
                    AF2=np.average(AF2,axis=0)
            Gm.append(AF2)
        else:
            if i>0:
                if j>0:
                    i=abs(i)
                    j=abs(j)
                    Sdata=Gmstack[immobilefactor2:ACFlen+immobilefactor2,rowstart:rowstart+ROIheight-i,collumnstart:collumnstart+ROIlength-j]
                    NSdata=Gmstack[immobilefactor2:ACFlen+immobilefactor2,rowstart+i:rowstart+ROIheight,collumnstart+j:collumnstart+ROIlength]
                    AF2=np.average(np.average(np.multiply(Sdata,NSdata),axis=1),axis=1)
                    AF2=np.divide(AF2,braF**2)
                    variancearr.append(AF2)
                    AF2=np.average(AF2,axis=0)
                else:
                    i=abs(i)
                    j=abs(j)
                    Sdata=Gmstack[immobilefactor2:ACFlen+immobilefactor2,rowstart:rowstart+ROIheight-i,collumnstart+j:collumnstart+ROIlength]
                    NSdata=Gmstack[immobilefactor2:ACFlen+immobilefactor2,rowstart+i:rowstart+ROIheight,collumnstart:collumnstart+ROIlength-j]
                    AF2=np.average(np.average(np.multiply(Sdata,NSdata),axis=1),axis=1)
                    AF2=np.divide(AF2,braF**2)
                    variancearr.append(AF2)
                    AF2=np.average(AF2,axis=0)
            else:
                if j>0:
                    i=abs(i)
                    j=abs(j)
                    Sdata=Gmstack[immobilefactor2:ACFlen+immobilefactor2,rowstart+i:rowstart+ROIheight,collumnstart:collumnstart+ROIlength-j]
                    NSdata=Gmstack[immobilefactor2:ACFlen+immobilefactor2,rowstart:rowstart+ROIheight-i,collumnstart+j:collumnstart+ROIlength]
                    AF2=np.average(np.average(np.multiply(Sdata,NSdata),axis=1),axis=1)
                    AF2=np.divide(AF2,braF**2)
                    variancearr.append(AF2)
                    AF2=np.average(AF2,axis=0)
                else:
                    i=abs(i)
                    j=abs(j)
                    Sdata=Gmstack[immobilefactor2:ACFlen+immobilefactor2,rowstart+i:rowstart+ROIheight,collumnstart+j:collumnstart+ROIlength]
                    NSdata=Gmstack[immobilefactor2:ACFlen+immobilefactor2,rowstart:rowstart+ROIheight-i,collumnstart:collumnstart+ROIlength-j]
                    AF2=np.average(np.average(np.multiply(Sdata,NSdata),axis=1),axis=1)
                    AF2=np.divide(AF2,braF**2)
                    variancearr.append(AF2)
                    AF2=np.average(AF2,axis=0)
            Gm.append(AF2)
Gm=np.reshape(Gm,(Sfactor*2+1,Sfactor*2+1))
#%%
Datastack=Gm
Timestack=np.empty_like(Gm)
for i in np.arange(0,Sfactor*2+1,2):
    for j in range(0,Sfactor*2+1):
        Timestack[i,j]=np.linspace((np.abs(i-Sfactor)//2+1)*2*imagesize-(1+np.abs(j-Sfactor))-(imagesize-np.abs(j-Sfactor)-1)*2,(np.abs(i-Sfactor)//2+1)*2*imagesize-(1+np.abs(j-Sfactor)),imagesize-np.abs(j-Sfactor))[::-1]
for i in np.arange(1,2*Sfactor,2):
    for j in range(0,Sfactor*2+1):
        Timestack[i,j]=(np.abs(i-Sfactor)//2)*2*imagesize+np.abs(j-Sfactor)
xstack=np.empty_like(Gm)
for i in np.arange(0,Sfactor*2+1,2):
    for j in range(0,Sfactor*2+1):
        xstack[i,j]=np.ones(imagesize-np.abs(j-Sfactor))*(j-Sfactor)
for i in np.arange(1,Sfactor*2,2):
    for j in range(0,Sfactor*2+1):
        xstack[i,j]=1.0*(j-Sfactor)
ystack=np.empty_like(Gm)
for i in np.arange(0,Sfactor*2+1,2):
    for j in range(0,Sfactor*2+1):
        ystack[i,j]=np.ones(imagesize-np.abs(j-Sfactor))*(i-Sfactor)
for i in np.arange(1,Sfactor*2,2):
    for j in range(0,Sfactor*2+1):
        ystack[i,j]=1.0*(i-Sfactor)
noyx=xstack[Sfactor,Sfactor+1:]
noyx=np.array(flat(noyx))
noyy=ystack[Sfactor,Sfactor+1:]
noyy=np.array(flat(noyy))
noyt=Timestack[Sfactor,Sfactor+1:]
noyt=np.array(flat(noyt))*taup
noyd=Datastack[Sfactor,Sfactor+1:]
noyd=np.array(flat(noyd))
noxxo=xstack[Sfactor:,Sfactor][1::2][::-1]
noxxo=np.array(flat(noxxo))
noxyo=ystack[Sfactor:,Sfactor][1::2][::-1]
noxyo=np.array(flat(noxyo))
noxto=Timestack[Sfactor:,Sfactor][1::2][::-1]
noxto=np.array(flat(noxto))*taup
noxdo=Datastack[Sfactor:,Sfactor][1::2][::-1]
noxdo=np.array(flat(noxdo))
noxxe=xstack[Sfactor+1:,Sfactor][1::2]
noxxe=np.array(flat(noxxe))
noxye=ystack[Sfactor+1:,Sfactor][1::2]
noxye=np.array(flat(noxye))
noxte=Timestack[Sfactor+1:,Sfactor][1::2]
noxte=np.array(flat(noxte))*taup
noxde=Datastack[Sfactor+1:,Sfactor][1::2]
noxde=np.array(flat(noxde))
Tubedata=np.dstack((xstack[Sfactor+1,Sfactor],ystack[Sfactor+1,Sfactor],Timestack[Sfactor+1,Sfactor],Datastack[Sfactor+1,Sfactor]))
np.savetxt('TubeData.csv',Tubedata[0],delimiter=",",fmt='%.18e')
Timearray=np.concatenate(Timestack)
Timearray=np.array(flat(Timearray))*taup
xstack=np.concatenate(xstack)
xstack=np.array(flat(xstack))
ystack=np.concatenate(ystack)
ystack=np.array(flat(ystack))
Datastack=np.concatenate(Datastack)
Datastack=np.array(flat(Datastack))
midpoint=int((len(Timearray)-1)/2)
Zero=Datastack[midpoint]
Datastack[midpoint]=(Datastack[midpoint-1]+Datastack[midpoint+1])/2
GData=np.dstack((xstack,ystack,Timearray,Datastack))
np.savetxt('GData.csv',GData[0],delimiter=",",fmt='%.18e')
Datastack[midpoint]=Zero
#%%
midpoint=int((len(Timearray)-1)/2)
Zero=Datastack[midpoint]
Datastack[midpoint]=(Datastack[midpoint-1]+Datastack[midpoint+1])/2
# fig=plt.figure(dpi=600,figsize = (12,10))
# ax=fig.gca(projection='3d')
# ax.scatter(xstack,ystack,Timearray,c=Datastack,cmap=plt.cm.inferno)
# plt.xlabel('xi')
# plt.ylabel('psi')
# ax.set_zlabel('Correlation Time')
# plt.title('QD ROC')
# plt.show()
Datastack[midpoint]=Zero
#%%
Na=6.022e23
Veff=1.5e-15
midpoint=int((len(Timearray)-1)/2)
fitarray=np.concatenate((Timearray[:midpoint],Timearray[midpoint+1:]))
fitx=np.concatenate((xstack[:midpoint],ystack[midpoint+1:]))
fity=np.concatenate((ystack[:midpoint],ystack[midpoint+1:]))
fitdata=np.concatenate((Datastack[:midpoint],Datastack[midpoint+1:]))
# popt,pcov=curve_fit(GDL, (noxxo,noxyo,noxto),noxdo,p0=[2.0e-12, 1.19193849],method='lm')
# Con=1/popt[1]/Na/Veff
# print('total fit para '+str(popt[0]/10.0**(-12)))
# print('total fit para '+str(popt[1]))
# print('total fit cov '+str(pcov))
# print(Con)
# popt,pcov=curve_fit(GDL, (noxxe,noxye,noxte),noxde,p0=[2.0e-12, 1.19193849],method='lm')
# Con=1/popt[1]/Na/Veff
# print('total fit para '+str(popt[0]/10.0**(-12)))
# print('total fit para '+str(popt[1]))
# print('total fit cov '+str(pcov))
# print(Con)
popt,pcov=curve_fit(GDL, (noyx,noyy,noyt),noyd,p0=[0, 0.019193849],method='lm')
Con=1/popt[1]/Na/Veff
print('total fit para '+str(popt[0]/10.0**(-12)))
print('total fit para '+str(popt[1]))
print('total fit cov '+str(pcov))
print(Con)
# popt,pcov=curve_fit(GDL, (fitx,fity,fitarray),fitdata,p0=[2.0e-12, 1.19193849],method='lm')
# Con=1/popt[1]/Na/Veff
# print('total fit para '+str(popt[0]/10.0**(-12)))
# print('total fit para '+str(popt[1]))
# print('total fit cov '+str(pcov))
# print(Con)
#%%
# fig=plt.figure(dpi=600,figsize = (12,10))
# ax=fig.gca(projection='3d')
# ax.scatter(xstack,ystack,Timearray,c=GDL((xstack,ystack,Timearray),popt[0],popt[1]),cmap=plt.cm.inferno)
# plt.xlabel('xi')
# plt.ylabel('psi')
# ax.set_zlabel('Correlation Time')
# plt.title('QD ROC Fit')
# plt.show()
#%%
# fig=plt.figure(dpi=600,figsize = (12,10))
# ax=fig.gca(projection='3d')
# ax.scatter(xstack,ystack,Timearray,c=GDL((xstack,ystack,Timearray),popt[0],popt[1])-Datastack,cmap=plt.cm.inferno)
# plt.xlabel('xi')
# plt.ylabel('psi')
# ax.set_zlabel('Correlation Time')
# plt.title('QD ROC Residuals')
# plt.show()
print(max(GDL((xstack,ystack,Timearray),popt[0],popt[1])-Datastack))
#%%
plt.figure(dpi=1000)
plt.scatter(noyt,noyd,c='r',alpha=0.2,label='($\\xi$,0)')
plt.scatter(noxto,noxdo,c='b',alpha=0.2,label='(0,$\psi$=odd)')
plt.scatter(noxte,noxde,c='orange',label='(0,$\psi$=even)')
plt.xscale('log')
plt.ylabel('G($\\tau$)')
plt.xlabel('Correlation Time (s)')
plt.title('MSICS Spatially Uncorrected ACF')
# plt.ylim(0.0,0.32)
plt.legend()
plt.show()
#%%
plt.figure(dpi=1000)
plt.scatter(noyt,GDL((noyx,noyy,noyt),popt[0],popt[1]),c='r',alpha=0.2,label='($\\xi$,0)')
plt.scatter(noxto,GDL((noxxo,noxyo,noxto),popt[0],popt[1] ),c='b',alpha=0.2,label='(0,$\psi$=odd)')
plt.scatter(noxte,GDL((noxxe,noxye,noxte),popt[0],popt[1]),c='orange',label='(0,$\psi$=even)')
plt.xscale('log')
plt.ylabel('G(t)')
plt.xlabel('Correlation Time (s)')
plt.title('MSICS Spatially Uncorrected ACF')
plt.legend()
plt.show()
#%%
plt.figure(dpi=1000)
plt.scatter(noyt,noyd,c='r',alpha=0.2,label='($\\xi$,0)')
plt.scatter(noxto,noxdo,c='b',alpha=0.2,label='(0,$\psi$=odd)')
plt.scatter(noxte,noxde,c='orange',label='(0,$\psi$=even)')
plt.plot(noyt,GDL((noyx,noyy,noyt),popt[0],popt[1]),c='r',label='($\\xi$,0)')
plt.plot(noxto[0:100],GDL((noxxo[0:100],noxyo[0:100],noxto[0:100]),popt[0],popt[1]),c='black',label='(0,$\psi$=odd)')
plt.plot(noxto[100:200],GDL((noxxo[100:200],noxyo[100:200],noxto[100:200]),popt[0],popt[1]),c='black',label='(0,$\psi$=odd)')
plt.scatter(noxte,GDL((noxxe,noxye,noxte),popt[0],popt[1]),c='orange',label='(0,$\psi$=even)',marker='s')
plt.xscale('log')
plt.ylabel('G(t)')
plt.xlabel('Correlation Time (s)')
plt.title('MSICS Spatially Uncorrected ACF')
plt.show()
#%%
plt.figure(dpi=1000)
plt.imshow(Gmstack[50,:,:],cmap="plasma")
plt.colorbar()
plt.xlabel('x (pixels)')
plt.ylabel('y (pixels)')
plt.title('1 Frame Intensity Map')
plt.show()
#%%
plt.figure(dpi=1000)
plt.imshow(np.sum(Gmstack,axis=0),cmap="plasma")
plt.colorbar()
plt.xlabel('x (pixels)')
plt.ylabel('y (pixels)')
plt.title('QD 580 1000 Frame Intensity Map')
plt.show()
#%%
plt.figure(dpi=600)
plt.hist(variancearr[511][:,1],bins=100)
plt.title('ACF Frame by Frame Histogram G(0,1) $\\tau\{1\}$')
plt.xlabel('ACF Value')
plt.ylabel('# Frames')
plt.show()
#%%
plt.figure(dpi=600)
plt.hist(variancearr[489],bins=100)
plt.title('ACF Frame by Frame Histogram G(8,0)')
plt.xlabel('ACF Value')
plt.ylabel('# Frames')
plt.show()
#%%
plt.figure(dpi=600)
plt.hist(variancearr[937][:,1],bins=100)
plt.title('ACF Frame by Frame Histogram G(15,1) $\\tau\{1\}$')
plt.xlabel('ACF Value')
plt.ylabel('# Frames')
plt.show()
#%%
plt.figure(dpi=1000)
plt.scatter(noyt,noyd,c='r',alpha=0.2,label='($\\xi$,0)')
# plt.scatter(noxto,noxdo,c='b',alpha=0.2,label='(0,$\psi$=odd)')
# plt.scatter(noxte,noxde,c='orange',label='(0,$\psi$=even)')
plt.plot(noyt,GDL((noyx,noyy,noyt),popt[0],popt[1]),c='r',label='($\\xi$,0)')
# plt.plot(noxto,GDL((noxxo,noxyo,noxto),popt[0],popt[1]),c='black',label='(0,$\psi$=odd)')

# plt.scatter(noxte,GDL((noxxe,noxye,noxte),popt[0],popt[1]),c='orange',label='(0,$\psi$=even)',marker='s')
plt.xscale('log')
plt.ylabel('G(t)')
plt.xlabel('Correlation Time (s)')
plt.title('MSICS Spatially Uncorrected ACF')
plt.show()
#%%
def Sn(inp,omega,N):
    xi,psi,taud=inp
    return 0.3535*1.0/N*np.exp(-1.0*((pixelsize*xi)**2+(pixelsize*psi)**2)/(omega**2))
popt,pcov=curve_fit(Sn, (noyx,noyy,noyt),noyd,p0=[50.0e-9, 0.0019193849],method='lm')
Con=1/popt[1]/Na/Veff
print('total fit para '+str(popt[0]/10.0**(-9)))
print('total fit para '+str(popt[1]))
print('total fit cov '+str(pcov))
print(Con)
plt.figure(dpi=1000)
plt.scatter(noyt,noyd,c='r',alpha=0.2,label='($\\xi$,0)')
# plt.scatter(noxto,noxdo,c='b',alpha=0.2,label='(0,$\psi$=odd)')
# plt.scatter(noxte,noxde,c='orange',label='(0,$\psi$=even)')
plt.plot(noyt,Sn((noyx,noyy,noyt),popt[0],popt[1]),c='r',label='($\\xi$,0)')
# plt.plot(noxto,Sn((noxxo,noxyo,noxto),popt[0],popt[1]),c='black',label='(0,$\psi$=odd)')

# plt.scatter(noxte,GDL((noxxe,noxye,noxte),popt[0],popt[1]),c='orange',label='(0,$\psi$=even)',marker='s')
plt.xscale('log')
plt.ylabel('G(t)')
plt.xlabel('Correlation Time (s)')
plt.title('MSICS Spatially Uncorrected ACF')
plt.show()