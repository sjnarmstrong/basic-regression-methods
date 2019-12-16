import numpy as np
import scipy.io as spio
import Plotter as plt
from scipy.stats import multivariate_normal

mat = spio.loadmat('./Datasets/Dataset_1.mat', squeeze_me=True);

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

alpha=5*10**-3
beta=11.1
M=4

sudoM=M+1

tlist=np.array(mat['y']).reshape((-1,1))
xlist=np.arange(0,1+1.0/(len(tlist)),1.0/(len(tlist)-1.0))


#tlist=np.delete(tlist,[1,2,3,4,5])
#xlist=np.delete(xlist,[1,2,3,4,5])

xMat=getXMatrix(xlist,sudoM)

Sninv=alpha*np.eye(sudoM)+beta*np.dot(xMat.T,xMat)

MnEqns=beta*np.dot(xMat.T,tlist)

Mn=np.linalg.solve(Sninv,MnEqns)

xFineList=np.arange(xlist[0],xlist[-1],0.001)
xFineMat=getXMatrix(xFineList,sudoM)

Sn=np.linalg.inv(Sninv)
sigmaxlist=[]
for xp in xFineList:
    phi=createSQM(xp,sudoM).reshape((-1,1))
    sigma_2n=1/beta+np.dot(phi.T,np.dot(Sn,phi))
    sigmaxlist+=[np.sqrt(sigma_2n[0,0])]

axarr = plt.createaxis()
#plt.setupaxis(axarr, "X Values", "Y Values", "Plot of Gaussian curve fitting for M="+str(M)+" with half of the points")
plt.setupaxis(axarr, "X Values", "Y Values", "Plot of Gaussian curve fitting for M="+str(M))
plt.plot(xlist,tlist,axarr, graphtype="scatter")
f_W=f_WandX(Mn,xFineMat).reshape(-1)
plt.plot(xFineList,f_W,axarr)
plt.plot(xFineList,f_W+sigmaxlist,axarr, ls='--')
plt.plot(xFineList,f_W-sigmaxlist,axarr, ls='--')
#plt.showOutput(FileName="Q3P3_minus_a_few_points")
#plt.showOutput(FileName="Q3P3")
plt.showOutput()


axarr = plt.createaxis()
#plt.setupaxis(axarr, "X Values", "Y Values", "Plot of 10 randomly drawn curves from the posterior distribution with half of the points")
plt.setupaxis(axarr, "X Values", "Y Values", "Plot of 10 randomly drawn curves from the posterior distribution")
plt.plot(xlist,tlist,axarr, graphtype="scatter")

SN=np.linalg.inv(Sninv)

for i in range(10):
    randomW=np.random.multivariate_normal(Mn.reshape(-1),SN)
    plt.plot(xFineList,f_WandX(randomW.reshape((-1,1)),xFineMat),axarr)
#plt.showOutput(FileName="Q3P3RandomSample")
#plt.showOutput(FileName="Q3P3RandomSample_minus_a_few_points")
plt.showOutput()

maxaposteroiri=multivariate_normal.pdf(Mn.reshape(-1), mean=Mn.reshape(-1), cov=SN);
