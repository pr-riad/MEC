# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.spatial.distance import norm
from math import exp
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# /////////////////////////////////////////////////////////////////////////////
#
# Mutual equidistant-scattering criterion (MEC) index validation using python
#
"""
Reference: Flexa, C., Santos, R., Gomes, W., Sales, C., & Costa, J. C. (2019). 
Mutual equidistant-scattering criterion: A new index for crisp clustering. 
Expert Systems with Applications, 128, 225-245. 
https://doi.org/10.1016/j.eswa.2019.03.027
"""
"""
Created on 08/19/2019

@author: R. RIAD
"""
def GetCluster(X, lab, c):
    Clust=[]
    for i in range(len(X)):
        if lab[i] == c:
            Clust.append(X[i])
    return np.asarray(Clust)

def GetCenters(X, lab):
    n_clusters = int(max(lab)+1)
    centers = []
    for c in range(n_clusters):
        centers.append(GetCluster(X, lab, c).mean(axis=0))
    return np.asarray(centers)

def MEC(X, labels):
    centers = GetCenters(X, labels)
    K = len(centers) # Number of centers or clusters
    sumK = 0.0
    for i in range(K):
        x = []
        for idx in range(len(labels)):
           if labels[idx]==i:
            x.append(X[idx])
        if (len(x) > 1):
            d = pdist(x, 'euclidean') # Euclidean dist computed between all samples
            L = len(d) # Number of computed distances
        elif (len(x)==1):
            d = norm(x, 2)
            L = 1
        else:
            d = 0
            L = 1
        if (L > 1):
            # Sum of absolute difference:
            d.sort()
            c = [0]*(L - 1)
            c[len(c) - 1] = d[len(d) - 1]
            for j in range(2, L):
               c[L - j - 1] = d[L - j] + c[L - j]
            # Number of absolute differences computed:
            eta = ((L * (L - 1)) / 2)
          
          # Mean Absolute Difference (MAD)
            sumAD = 0
            for j in range(L - 1):
                sumAD += (c[j] - d[j] * (L - j -1))
            MAD = eta**(-1)*sumAD
            # Nonlinear penalization on compactness:
            s = mean(d**2) - mean(d)**2
            if (exp(-s)!=0):
                sig = (1.0 - exp(-s))/exp(-s)
            else:
                sig = 1.0
            sumK += sig * MAD
        else:
            sumK += d
    # Penalization on separation
    if (K > 1): # Eq_17 paper MEC
        lamb = K * 1.0 / max(pdist(centers))
    else:
        lamb = 1.0
    y = lamb * sumK
    return y

"""
Created on 13/08/2021

@author: R. RIAD
"""
def mec_eval(X, k_min = 1, k_max = 10, disp = False):    # 
    
    mec_list = [];
    n_clusters_list = [];

    for n_clusters in range(k_min, k_max+1):
        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        KM = KMeans(n_clusters=n_clusters,init='k-means++',n_init=10,
                         max_iter=700,random_state=10)
        
        cluster_labels = KM.fit_predict(X)
        scor_mec = MEC(X, cluster_labels)
       # print("For n_clusters =", n_clusters, "The MEC scor is :", scor_mec)
       
        mec_list.append(scor_mec)
        n_clusters_list.append(n_clusters)

    # Compute the optimale number of clusters acording to Eq_18 
    #                    K_hat = argmin(MEC(K))             (18)
    Min_mec = min(mec_list)
    idx_min_mec = mec_list.index(Min_mec)
    K_hat = n_clusters_list[idx_min_mec]
    # Display
    if (disp):
        plt.plot(n_clusters_list, mec_list, 'bo--')
        plt.plot(K_hat, Min_mec, 'ro')
        plt.text(K_hat, Min_mec+max(mec_list)/10, 'K_opt = %d' % K_hat)
        plt.ylabel("MEC")
        plt.xlabel("Number of clusters K")
        plt.show()
    return K_hat
#//////////////////////////////////////////////////////////////////////////////
