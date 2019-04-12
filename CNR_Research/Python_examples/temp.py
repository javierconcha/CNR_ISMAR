#%%
from scipy.spatial.distance import mahalanobis
import scipy as sp
import pandas as pd
import numpy as np
#%%
x = pd.read_csv('/Users/javier/Desktop/Javier/2019_ROMA/CNR_Research/Python_examples/iris2.csv')
x = x.iloc[:,1:]

Sx = x.cov().values
Sx = sp.linalg.inv(Sx)

mean = x.mean().values

def mahalanobisR(X,meanCol,IC):
    m = []
    for i in range(X.shape[0]):
        m.append(mahalanobis(X.iloc[i,:],meanCol,IC) ** 2)
    return(m)

mR = mahalanobisR(x,mean,Sx)
#%%
from scipy.spatial.distance import cdist

x = pd.read_csv('/Users/javier/Desktop/Javier/2019_ROMA/CNR_Research/Python_examples/iris2.csv')
x = x.iloc[:,1:]

mean = x.mean(axis=0).reshape(1, -1)  # make sure 2D
vi = np.linalg.inv(np.cov(x.T))

cdist(mean, x, 'mahalanobis', VI=vi)