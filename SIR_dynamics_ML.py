# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 15:29:54 2020

@author: subrata
"""

import numpy as np
from numpy import *
from scipy import stats
from scipy.integrate import odeint
import pylab as pl
import random
import math
import scipy.io
import networkx as nx
import pandas as pd
from scipy.signal import savgol_filter

# create time series of SIR model
N = 1000

def f(y,t):
    dy = np.empty(2*N,float)
    u = y[0:2*N:2]
    v = y[1:2*N+1:2]
    for i in range(0,N):
        dy[2*i] = (-B[i]*u[i]*v[i] ) 
        dy[2*i+1] =( B[i]*u[i]*v[i] - P*v[i] ) 
    return dy
B =np.random.uniform(0,0.6,N)
P =1./14
tf = 300
y0 = np.zeros(2*N,float)
y0[1:2*N+1:2]= np.random.uniform(0.000001,0.000008,N) 
y0[0:2*N:2]= 1.- y0[1:2*N+1:2] 
t1= np.linspace(0, tf, tf*100 )
[sol1,info] = odeint(f,y0,t1,full_output=1,printmessg=1) 

Y1=sol1[:,1:2*N+1:2]
trainlength=20000
testlength=5000
initlength=0

insize=950
outsize=50
resize = 1000
a = 0.5

Win = (np.random.rand(resize,1+insize)-0.5)*1
W= (np.random.rand(resize,resize)-0.5)*1

indata=Y1[:,0:insize]
outdata=Y1[:,insize:outsize+insize]

X= zeros((1+insize+resize,trainlength-initlength))
Yt= outdata[initlength:trainlength,:]
x= zeros((resize,1))
for t in range(trainlength):
    u =indata[t,:]
    u = np.reshape(u,(insize,1))
    x = (1-a)*x +a*tanh(dot(Win,vstack((1,u)) ) + dot(W,x))
    if t > initlength:
        X[:,t-initlength] = vstack((1,u,x))[:,0]
        
reg= 1e-10
X_T = X.T
Wout = dot(dot(Yt.T,X_T),linalg.inv(dot(X,X_T)+reg*eye(1+insize+resize)))

Y =zeros((outsize,testlength))  
u = indata[trainlength,:] 
u = np.reshape(u,(insize,1)) 
for t in range(testlength):
    x = (1-a)*x + a*tanh( dot( Win, vstack((1,u)) ) + dot( W, x ) )
    y = dot( Wout, vstack((1,u,x)) )
    Y[:,t] = y.T
    u = indata[trainlength+t+1,:] 
    u = np.reshape(u,(insize,1))    

ind=(0,20000,25000)
ind1=(0,0.1,0.2)
#plot a single time series
pl.figure(1)
pl.plot(Y1[0:25000,201],color='r',linewidth=10)
pl.axvline(x = 20000,linewidth = 3, linestyle ="--", color ='black')
pl.yticks(fontsize=0)
pl.xticks(ind,('',r'$t_{r}$',r'$t_{f}$'),fontsize=60)
pl.xlabel(r'$t$',fontsize=60)
pl.ylabel(r'$\mathcal{I}$',fontsize=60)
pl.tight_layout()
# plot some predicted time series

days = np.arange(0,25000)
for i in range(30,50):
    fig,ax = pl.subplots()
    ax.plot(days[0:25000],outdata[0:25000,i],'blue',lw=10)
    ax.scatter(days[20000:25000:700],Y[i,0:5000:700],s=400,color='none',edgecolors='darkorange',lw=5)
    pl.axvline(x = 20000,linewidth = 3, linestyle ="--", color ='black')
    pl.yticks(fontsize=0)
    pl.xticks(ind,('',r'$t_{r}$',r'$t_{f}$'),fontsize=60)
    pl.xlabel(r'$t$',fontsize=60)
    pl.ylabel(r'$\mathcal{I}$',fontsize=60)
    pl.tight_layout()