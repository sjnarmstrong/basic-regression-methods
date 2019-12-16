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
    
def getMnandSninv(M,tlist,xMat):
    Sninv=alpha*np.eye(M)+beta*np.dot(xMat.T,xMat)
    MnEqns=beta*np.dot(xMat.T,tlist)
    Mn=np.linalg.solve(Sninv,MnEqns)
    
    return Mn, Sninv
    
def getE_mn(beta,Phi,Mn,t,alpha):
    return (beta*(np.linalg.norm(t-np.dot(Phi,Mn))**2)/2.0)+(alpha*np.dot(Mn.T,Mn)/2.0)
    
tlist=np.array(mat['y']).reshape((-1,1))
xlist=np.arange(0,1+1.0/(len(tlist)),1.0/(len(tlist)-1.0))
xFineList=np.arange(xlist[0],xlist[-1]+(xlist[-1]-xlist[0])/(1000),(xlist[-1]-xlist[0])/(1000-1))


N=len(tlist)

p_DGMlist=[]
Ap_DGMlist=[]
MnList=[]
AList=[]

snholdasdf=0

predictorList=np.array([])
sigmaxlist=np.array([])
for sudoM in range(1,11):    
    Phi=getXMatrix(xlist,sudoM)
    Mn, A= getMnandSninv(sudoM,tlist,Phi)
    MnList+=[Mn]
    AList+=[A]
    E_Mn=getE_mn(beta,Phi,Mn,tlist,alpha)
    p_DGM=((sudoM*np.log(alpha)+N*np.log(beta)-np.log(np.linalg.det(A))-N*np.log(2*np.pi))/2.0)-E_Mn        
    p_DGMlist+=[p_DGM[0,0]]
    
    
    Ap_DGM=(beta/(2*np.pi))**(N/2)\
        *(alpha)**(sudoM/2)\
        *(np.linalg.det(A)**(-0.5))\
        *np.exp(-E_Mn)
    Ap_DGMlist+=[Ap_DGM[0,0]]
    
    
    xFineMat=getXMatrix(xFineList,sudoM)
    f_W=f_WandX(Mn,xFineMat).reshape(-1)
    predictorList=np.append(predictorList,f_W)
    
    Sn=np.linalg.inv(A)
    for xp in xFineList:
        phi=createSQM(xp,sudoM).reshape((-1,1))
        sigma_2n=1/beta+np.dot(phi.T,np.dot(Sn,phi))
        sigmaxlist=np.append(sigmaxlist,sigma_2n[0,0])
        
        
predictorList=predictorList.reshape((10,-1)) 
sigmaxlist=sigmaxlist.reshape((10,-1))   
    
axarr = plt.createaxis()
plt.setupaxis(axarr, "Model Complexity", "Evidance", "Plot of Evidence vs M")
plt.plot(range(0,10),Ap_DGMlist,axarr)
#plt.showOutput(FileName="Q3P6LogEvidance")
plt.showOutput()
    
    
#bestOrder=3
bestOrder=np.argmax(p_DGMlist)

for order in range(10): 
    axarr = plt.createaxis()
    #plt.setupaxis(axarr, "X Values", "Y Values", "Plot of Gaussian curve fitting for M="+str(M)+" with half of the points")
    plt.setupaxis(axarr, "X Values", "Y Values", "Plot of Gaussian curve fitting for M="+str(order))
    plt.plot(xlist,tlist,axarr, graphtype="scatter")
    f_W=predictorList[order]
    plt.plot(xFineList,f_W,axarr)
    plt.plot(xFineList,f_W+np.sqrt(sigmaxlist[order]),axarr, ls='--')
    plt.plot(xFineList,f_W-np.sqrt(sigmaxlist[order]),axarr, ls='--')
    #plt.showOutput(FileName="Q3P6Best")
    #plt.showOutput(FileName="Q3P3_minus_a_few_points")
    #plt.showOutput(FileName="Q3P3")
    plt.showOutput()
    
    
    




###############Question 3.7
Ap_DGMlist=Ap_DGMlist/np.sum(Ap_DGMlist)
FCombined=np.dot(Ap_DGMlist,predictorList)

sigma2=0
for mui,sigmai,w in zip(predictorList,sigmaxlist,Ap_DGMlist):
    sigma2+=w*((mui-FCombined)**2+sigmai)

sigma=np.sqrt(sigma2)

axarr = plt.createaxis()
#plt.setupaxis(axarr, "X Values", "Y Values", "Plot of Gaussian curve fitting for M="+str(M)+" with half of the points")
plt.setupaxis(axarr, "X Values", "Y Values", "Plot of Gaussian curve fitting for a mixture of complexities")
plt.plot(xlist,tlist,axarr, graphtype="scatter")
plt.plot(xFineList,FCombined,axarr)
plt.plot(xFineList,f_W+sigma,axarr, ls='--')
plt.plot(xFineList,f_W-sigma,axarr, ls='--')
plt.showOutput()