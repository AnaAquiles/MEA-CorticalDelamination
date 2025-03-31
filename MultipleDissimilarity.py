# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 14:09:49 2024

@author: aaquiles

          MATRIX DISSIMILARITY

"""

from scipy.signal import hilbert
import matplotlib.pyplot as plt 
import math
import pandas as pd
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay


def RandMat(n):
    n = 5
    r = np.random.rand(n*(n+1)//2)
    sym = np.zeros((n,n))
    for i in range(n):
        t = i*(i+1)//2
        sym[i,0:i+1] = r[t:t+i+1]
        sym[0:i,i] = r[t:t+i]
    return sym

MatARANGE = []
for i in range(45):
    MatARANGE.append(RandMat(i))
    
Mats = np.array(MatARANGE)

### essay 

def DistKnnVECT(MatARANGE, MatARANGE1):
    
    
    X = MatARANGE
    X_train = MatARANGE1
    
    m = X.shape[0]
    n = X_train.shape[0]
    d = X.shape[1]  ## just if the mtrix have not the same shape 

    dists = np.zeros((len(X), len(X_train)))
    
    for j in range(m):
        dists[j, :] = np.sqrt(np.sum((X[j] - X_train) ** 2, axis=1))
    return dists

Distance = []
for j in range(len(Mats)):
    Distance.append(DistKnnVECT(Mats[j,:,:], Mats[j+1,:,:]))

Distance = np.array(Distance)
Int = len(Distance)

DissVals = []
for i in range(Int):
    DissVals.append(Distance[i,0,:])
    
DissVals = np.array(DissVals)
