"""
Clustering by scale-space filtering

Author: Hongtao Wang
---
cite: 
Yee Leung, Jiang-She Zhang and Zong-Ben Xu, "Clustering by scale-space filtering," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 22, no. 12, pp. 1396-1410, Dec. 2000, doi: 10.1109/34.895974.
"""

import sklearn.datasets
from time import time
import numpy as np
import copy


def Norm2(x, y):
    return np.sqrt(np.sum((x-y)**2, axis=1))


class SSF:
    def __init__(self, DistanceThreshold=1e-4, ClusterThreshold=1e-4, Sigma0=0.05, MinNum=5):
        self.DisThre = DistanceThreshold
        self.CluThre = ClusterThreshold
        self.Sig = Sigma0
        self.MinNum = MinNum
        self.X = None
    
    def cluster(self, X):
        start = time()
        # init 
        CluDict, self.X, count, CenterNum = {}, X, 1, []
        # each point in datum are the init center points 
        CenterPoints = copy.deepcopy(X)

        while True:
            CenterN, IterTime = self.IterateBlob(CenterPoints)
            # while True:
            while True:
                MergedCenters = self.MergeBlob(CenterN)
                if MergedCenters.shape[0] == CenterN.shape[0] or MergedCenters.shape[0] <= self.MinNum:
                    CenterPoints = copy.deepcopy(MergedCenters)
                    break
                else:
                    CenterN = copy.deepcopy(MergedCenters)
            
            CluDict.setdefault(count, {'sigma': copy.deepcopy(self.Sig),
                                       'center': CenterPoints,
                                       'IterTime': IterTime})

            CenterNum.append(CenterPoints.shape[0])
            # print('Count: %d\tSigma: %.5f\tCenter Num: %d\tIterTime: %d' 
            #       % (count, self.Sig, CenterN.shape[0], IterTime))
            
            # break situation
            if CenterPoints.shape[0] <= self.MinNum:
                break
            else:
                # update sigma and count number
                self.Sig *= 1.029
                count += 1

        CenterNum = np.array(CenterNum)
        BestCluster = np.min(np.where(CenterNum==np.argmax(np.bincount(CenterNum)))[0]) + 1
        # print('Sigma Change Time: %d\tCost Time: %.2fs' % (count, time()-start))
        # print('Best Sigma: %.5f\tCenter Number: %d' % (CluDict[BestCluster]['sigma'], len(CluDict[BestCluster]['center'])))
        return CluDict[BestCluster]['center'], CluDict[BestCluster]['sigma'], CluDict
    
    def IterateBlob(self, CenterPoints):
        IterTime, NewCenterPoints = 1, copy.deepcopy(CenterPoints)
        while True:
            FinalCenterPoints = self.ShiftBlob(NewCenterPoints)
            # judge the distance of these iteration < Distance Threshold
            if np.max(Norm2(FinalCenterPoints, NewCenterPoints)) > self.DisThre:
                NewCenterPoints = FinalCenterPoints
                IterTime += 1
            else:
                break
        return FinalCenterPoints, IterTime

    def ShiftBlob(self, CenterPoints):
        NewCenterPoints = np.zeros_like(CenterPoints)
        for ind, x in enumerate(CenterPoints):
            GauKer = np.exp(-Norm2(self.X, x)/(2*self.Sig**2))
            NewCenterPoints[ind, :] = np.dot(GauKer.reshape(1, -1), self.X) / np.sum(GauKer)
        
        return NewCenterPoints

    def MergeBlob(self, CenterPoints):
        Index, MergeIndex = list(np.arange(CenterPoints.shape[0])), []
        for ind, x in enumerate(CenterPoints):
            Index.remove(ind)
            Distance = Norm2(CenterPoints[Index, :], x)
            MergeInd = np.array(Index)[Distance < self.CluThre]
            if len(MergeInd) > 0:
                MergeIndex += [[ind, indM] for indM in MergeInd]
        
        if len(MergeIndex) == 0:
            MergeResults = CenterPoints
        else:
            MergeSetList = []
            for ind1, ind2 in MergeIndex:
                Insert = 0
                for SetK in MergeSetList:
                    if ind1 in SetK or ind2 in SetK:
                        SetK.add(ind1)
                        SetK.add(ind2)
                        Insert = 1
                if not Insert:
                   MergeSetList.append({ind1, ind2}) 
            AllIndex, NewCenter = [], []
            for SetMerge in MergeSetList:
                ListMerge = list(SetMerge)
                NewCenter.append(np.mean(CenterPoints[ListMerge, :], axis=0))
                AllIndex += ListMerge
            SaveIndex = list(set(np.arange(CenterPoints.shape[0])) - set(AllIndex))
            MergeResults = np.vstack((CenterPoints[SaveIndex, :], np.array(NewCenter)))

        return MergeResults


if __name__ == "__main__":
    Data_general, Data_labels = sklearn.datasets.make_blobs(
    n_samples=300,
    n_features=2,
    centers=10,
    cluster_std=0.1,
    random_state=0,
    center_box=(-1, 1))

    ssf = SSF(Sigma0=0.05)
    result = ssf.cluster(Data_general)