"""
Clustering by attention scale-space filtering

Author: Hongtao Wang
---
cite: 
Yee Leung, Jiang-She Zhang and Zong-Ben Xu, "Clustering by scale-space filtering," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 22, no. 12, pp. 1396-1410, Dec. 2000, doi: 10.1109/34.895974.
"""

from time import time
import numpy as np
import copy


def DistLam(x, y, lam=1):
    diff = x - y
    if lam != 1:
        diff[:, 1] *= lam
    return np.sqrt(np.sum(diff**2, axis=1))


def Norm2(x, y):
    return np.sum((x-y)**2, axis=1)


class AttSSF:
    def __init__(self, DistanceThreshold=60, ClusterThreshold=40, Sigma0=40, alpha=1, lam=1.5, minNum=3):
        self.DisThre = DistanceThreshold
        self.CluThre = ClusterThreshold
        self.Sig = Sigma0
        self.alpha = alpha
        self.minNum = minNum
        self.lam = lam
        self.X, self.semW = None, None

    def cluster(self, X, values):
        # init 
        self.semW = 1 + copy.deepcopy(values)
        CluDict, self.X, count, CenterNum, IterList = {}, X, 1, [], []

        # each point in datum are the init center points 
        CenterPoints = copy.deepcopy(self.X)

        while True:
            CenterN, IterTime = self.IterateBlob(CenterPoints)
            # merge the blobs
            while True:
                MergedCenters = self.MergeBlob(CenterN)
                if MergedCenters.shape[0] == CenterN.shape[0] or MergedCenters.shape[0] == 1:
                    CenterPoints = copy.deepcopy(MergedCenters)
                    break
                else:
                    CenterN = copy.deepcopy(MergedCenters)
                    
            CluDict.setdefault(count, {'sigma': copy.deepcopy(self.Sig),
                                       'center': CenterPoints,
                                       'IterTime': IterTime})
            CenterNum.append(CenterPoints.shape[0])
            IterList.append([copy.deepcopy(self.Sig), CenterPoints])
                
            # break situation
            if CenterPoints.shape[0] <= self.minNum:
                break
            else:
                # update sigma and count number
                self.Sig *= 1.029
                count += 1

        CenterNum = np.array(CenterNum)
        BestCluster = np.min(np.where(CenterNum==np.argmax(np.bincount(CenterNum)))[0]) + 1
        # print('Sigma Change Time: %d\tCost Time: %.2fs' % (count, time()-start))
        # print('Best Sigma: %.5f\tCenter Number: %d' % (CluDict[BestCluster]['sigma'], len(CluDict[BestCluster]['center'])))
        # , CluDict[BestCluster]['sigma'], CluDict, IterList
        return CluDict[BestCluster]['center']
    
    def IterateBlob(self, CenterPoints):
        IterTime, NewCenterPoints = 1, copy.deepcopy(CenterPoints)
        while True:
            FinalCenterPoints = self.ShiftBlob(NewCenterPoints)
            # judge the distance of these iteration < Distance Threshold
            if np.max(DistLam(FinalCenterPoints, NewCenterPoints, self.lam)) > self.DisThre:
                NewCenterPoints = FinalCenterPoints
                IterTime += 1
            else:
                break
        return FinalCenterPoints, IterTime
    
    def ShiftBlob(self, CenterPoints):
        NewCenterPoints = np.zeros_like(CenterPoints)
        for ind, x in enumerate(CenterPoints):
            GauKer = np.exp(-Norm2(self.X, x)/(2*self.Sig**2))
            semW = self.semW
            Weight = GauKer * semW ** self.alpha
            NewCenterPoints[ind, :] = np.dot(Weight.reshape(1, -1), self.X) / np.sum(Weight)
        
        return NewCenterPoints

    def MergeBlob(self, CenterPoints):
        Index, MergeIndex = list(np.arange(CenterPoints.shape[0])), []
        for ind, x in enumerate(CenterPoints):
            Index.remove(ind)
            Distance = DistLam(CenterPoints[Index, :], x, self.lam)
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

