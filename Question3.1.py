import numpy as np
import scipy.io as spio
import Plotter as plt

def createSQM(x,M):
    vfunc = np.vectorize(lambda x,i: x**i)
    ilist=np.arange(M+1)
    xlist=np.repeat(x, M+1)
    return vfunc(xlist,ilist)

def getXMatrix(Xlist,M):
    out = np.empty((0,M+1))
    for x in Xlist:
        out = np.append(out,createSQM(x,M).reshape((1,M+1)),axis=0)
        
    return out

def f_WandX(W,x):
    return np.dot(x,W)

def Error(W,x,t):
    return 0.5*np.sum(np.square(np.dot(x,W)-t))

def getLogLiErr(W,x,t,Beta):
    return 0.5*Beta*np.sum(np.square(np.dot(x,W)-t))+0.5*len(t)*np.log(Beta/(2.0*np.pi))

def getOptimalW(x,t):
    return np.linalg.solve(np.dot(x.T,x),np.dot(x.T,t))



mat = spio.loadmat('Datasets\Dataset_1.mat', squeeze_me=True);

tlist=np.array(mat['y']).reshape((-1,1))
xlist=np.arange(0,1+1.0/(len(tlist)),1.0/(len(tlist)-1.0))

xFineList=np.arange(xlist[0],xlist[-1],0.001)

errorList=[]
lierrorList=[]

for Order in range(0,10):
    
    print("Running with order of: "+str(Order))
    
    xMat=getXMatrix(xlist,Order)
    xFineMat=getXMatrix(xFineList,Order)
    Wlist=getOptimalW(xMat,tlist)
    
    print("Weigths are: "+str(Wlist.reshape(-1)))
    
    err=Error(Wlist,xMat,tlist)
    errorList.append(err)
    print("Error is: "+str(err))
    
    err=getLogLiErr(Wlist,xMat,tlist,11.1)
    lierrorList.append(err)
    print("Log Error is: "+str(err))

    axarr = plt.createaxis()
    plt.setupaxis(axarr, "X Values", "Y Values", "Plot of least squares curve fitting for M="+str(Order))
    plt.plot(xlist,tlist,axarr, graphtype="scatter")
    plt.plot(xFineList,f_WandX(Wlist,xFineMat),axarr)
    plt.showOutput(FileName="Q3P1Order_"+str(Order))
    #plt.showOutput()
    
axarr = plt.createaxis()
plt.setupaxis(axarr, "M value", "Error", "Plot of error versus M")
plt.plot(range(0,10),errorList,axarr)
plt.showOutput(FileName="ErrvsM")
#plt.showOutput()

axarr = plt.createaxis()
plt.setupaxis(axarr, "M value", "Log Error", "Plot of log error versus M")
plt.plot(range(0,10),lierrorList,axarr)
plt.showOutput(FileName="LogErrvsM")
#plt.showOutput()


