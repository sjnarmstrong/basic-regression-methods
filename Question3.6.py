import numpy as np
import scipy.io as spio
import Plotter as plt
from scipy.stats import multivariate_normal, norm

mat = spio.loadmat('./Datasets/Dataset_2.mat', squeeze_me=True);
alpha=5*10**-3
beta=11.1

def createSQM(x,M):
    vfunc = np.vectorize(lambda x,i: x**i)
    ilist=np.arange(M)
    xlist=np.repeat(x, M)
    return vfunc(xlist,ilist)

def getXMatrix(Xlist,M):
    out = np.empty((0,M))
    for x in Xlist:
        out = np.append(out,createSQM(x,M).reshape((1,M)),axis=0)
        
    return out

def f_WandX(W,x):
    return np.dot(x,W)
    
def getMnandSn(M,xlist,tlist):
    xMat=getXMatrix(xlist,M)
    Sninv=alpha*np.eye(M)+beta*np.dot(xMat.T,xMat)
    MnEqns=beta*np.dot(xMat.T,tlist)
    Mn=np.linalg.solve(Sninv,MnEqns)
    
    return Mn, np.linalg.inv(Sninv)
    
tlist=np.array(mat['y']).reshape((-1,1))
xlist=np.arange(0,1+1.0/(len(tlist)),1.0/(len(tlist)-1.0))

p_DGMlist=[]
for sudoM in range(1,11):
    Mn, Sn= getMnandSn(sudoM,xlist,tlist)
    
    p_DGM=1
    for x,t in zip(xlist,tlist):
        phi=createSQM(x,sudoM).reshape((-1,1))
        sigma_2n=1/beta+np.dot(phi.T,np.dot(Sn,phi))
        p_DGM*=100*norm.pdf(t[0],loc=np.dot(Mn.T,phi)[0,0],scale=sigma_2n[0,0])
    p_DGMlist+=[p_DGM]
    
p_DGMlist=100.0*np.array(p_DGMlist)/np.sum(p_DGMlist)
        
    
    


