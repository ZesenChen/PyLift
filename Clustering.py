import numpy as np
from mlab.releases import latest_release as mb

def KMeans(data,k):
    index = mb.kmeans(data,k,'EmptyAction',  
                      'singleton','OnlinePhase', 
                      'off','Display','off').reshape((data.shape[0],))
    centers = np.zeros((k,data.shape[1]))
    for i in range(1,k+1):
        centers[i-1] = np.mean(data[index==i,:],0)
    return index,centers    
