import numpy as np
from sklearn.cluster import KMeans, DBSCAN, MeanShift, estimate_bandwidth
from sklearn.mixture import GaussianMixture

"""
Cluster method to get the centers
"""


# get the centers
def ClusterCenters(data, label):
    KList = list(set(label))
    centers = []
    for k in KList:
        centers.append(np.mean(data[label==k, :], axis=0))
    centers = np.array(centers)
    return centers


# DBSCAN
def MyDBSCAN(data, eps=20, min_samples=20):  
    model = DBSCAN(eps, min_samples=min_samples)
    label = model.fit_predict(np.array(data))
    centers = ClusterCenters(data, label)
    return centers


# K-means
def MyKMeans(data, k=8): 
    kmeans = KMeans(n_clusters=k).fit(np.array(data))
    return kmeans.cluster_centers_


# Gaussian Mixture Model
def MyGMM(data, k=8):  
    gmm = GaussianMixture(n_components=k).fit(np.array(data))
    return gmm.means_


# Mean-Shift
def MyMeanShift(data, bw=200):
    ms = MeanShift(bandwidth=bw, bin_seeding=True).fit(data)
    return ms.cluster_centers_