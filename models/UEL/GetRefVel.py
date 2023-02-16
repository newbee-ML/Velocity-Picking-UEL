"""
Get reference velocity prior from the spectra of the near samples

Author: Hongtao Wang | stolzpi@163.com
"""

import cv2
import copy
import numpy as np


def MakeSeedIndex(LabelIndex, rate=0.1):
    LineList = np.array(sorted(map(int, list(set([elm.split('-')[0] for elm in LabelIndex])))))
    CdpList = np.array(sorted(map(int, list(set([elm.split('-')[1] for elm in LabelIndex])))))
    LineSample = LineList[np.linspace(0, len(LineList)-1, int(len(LineList)*np.sqrt(rate))).astype(np.int32)]
    CdpSample = CdpList[np.linspace(0, len(CdpList)-1, int(len(CdpList)*np.sqrt(rate))).astype(np.int32)]
    SampleList = []
    for line in LineSample:
        for cdp in CdpSample:
            SampleIndex = '%d-%d' % (line, cdp)
            if SampleIndex in LabelIndex:
                SampleList.append(SampleIndex)
    return SampleList


# get the nearest the seef reference velocity
def GetSeedIndex(index, SeedIndexList, scale=(20, 20)):
    indexNew = np.array(index.split('-'), dtype=np.int32).reshape(1, 2)
    IndexArray = np.array([indexs.split('-') for indexs in  SeedIndexList], dtype=np.int32).reshape(-1, 2)
    distance = np.sum(np.abs(IndexArray - indexNew)/np.array(scale), axis=1)
    SeedIndex = SeedIndexList[np.argmin(distance)]
    return SeedIndex

##################################################################
# Get the low frequency spectrum information form near spectra
##################################################################

# Compute the scale parameter of line and cdp index
def GetScale(AllIndex):
    IndexNew = np.array([indexs.split('-') for indexs in sorted(AllIndex)], dtype=np.int32).reshape(-1, 2)
    LineIndex, cdpIndex = IndexNew[:, 0], IndexNew[:, 1]
    cdpScale = cdpIndex[1] - cdpIndex[0]
    LineNo = sorted(list(set(LineIndex)))
    LineScale = LineNo[1] - LineNo[0]
    return (LineScale, cdpScale)


# Find the near sample
def GetNearIndex(index, AllIndex, scale=(20, 20), k=2):
    indexNew = np.array(index.split('-'), dtype=np.int32).reshape(1, 2)
    IndexArray = np.array([indexs.split('-') for indexs in  AllIndex], dtype=np.int32).reshape(-1, 2)
    distance = np.sum(np.abs(IndexArray - indexNew)/np.array(scale), axis=1)
    AllIndex = np.array(AllIndex)
    NearIndex = list(AllIndex[np.where(distance <= k)[0]])
    NearIndex.remove(index)
    return sorted(NearIndex)


# Keep the same size among the near spectra and the target spectrum
def KeepSize(TargetDict, NearDict):
    """
    Keep the same size among the near spectra and the target spectrum
    According to the velocity vectors
    ---
    Para:
        - TargetDict: the data dict of the target specturm
        - NearDict: the data dict list of the near spectra
    """
    # Load target info
    vVecT = copy.deepcopy(TargetDict['scale']['v'])
    vVecD = vVecT[1] - vVecT[0]

    # Keep the size of the near spectra
    NewNearSpec = []
    for DataDict in NearDict:
        # Load basic info of near spectrum
        SpecNear = copy.deepcopy(DataDict['spectrum'])
        vVecNear = copy.deepcopy(DataDict['scale']['v'])

        # compare the vVecNear and the vVecT
        LeftLoc = int((vVecNear[0] - vVecT[0]) / vVecD)
        RightLoc = int((vVecNear[-1] - vVecT[-1]) / vVecD)

        # padding, clip or no change
        if LeftLoc == 0:   # no change
            NewSpec = SpecNear
        elif LeftLoc < 0:  # clip
            NewSpec = SpecNear[:, -LeftLoc:]
        else:              # padding
            NewSpec = np.hstack((np.zeros((SpecNear.shape[0], LeftLoc)), SpecNear))

        if RightLoc == 0:  # no change
            NewSpec = NewSpec
        elif RightLoc > 0: # clip
            NewSpec = NewSpec[:, : -RightLoc]
        else:              # padding
            NewSpec = np.hstack((NewSpec, np.zeros((SpecNear.shape[0], -RightLoc))))
        NewNearSpec.append(NewSpec)
            
    NewNearSpec = np.array(NewNearSpec)
    return NewNearSpec


# Get the low frequency map
def GetLowFrequencyMap(NearSpec, k=5, time=1):

    def LPFilters(img):
        imgcp = copy.deepcopy(img)
        for _ in range(time):
            imgcp = cv2.blur(imgcp, (k, k))
        return imgcp
    
    for j in range(NearSpec.shape[0]):
        NearSpec[j] = LPFilters(NearSpec[j])
    NearSpec = np.sum(NearSpec, axis=0)

    return NearSpec


################################################################
# Get the velocity prior from the low frequency map
################################################################
# Plan 1: Get the velocity prior from the low frequency map directly
def VelPriorMap2Weight(NearSpec, OriSpecPoints, t0Vec, vVec):
    X, NearCp = copy.deepcopy(OriSpecPoints), copy.deepcopy(NearSpec)
    NearCp /= np.max(NearCp)
    X[:, 0] = (X[:, 0] - t0Vec[0]) / (t0Vec[1] - t0Vec[0])
    X[:, 1] = (X[:, 1] - vVec[0]) / (vVec[1] - vVec[0])
    X = X.astype(np.int32)
    Weight = NearCp[X[:, 0], X[:, 1]]
    return Weight


# Plan 2: Get the reference velocity curve
class ILWLR:
    def __init__(self, x, y, v, k=200, lmd=1, minSum=10):
        self.x = x
        self.k = k
        self.v = v
        self.lmd = lmd
        self.minSum = minSum
        self.X = np.mat(np.hstack((x.reshape(-1, 1), np.ones((len(x), 1)))))
        self.Y = np.mat(y.reshape(-1, 1))
    
    def SinglePredict(self, xNew):
        XNew = np.mat([xNew, 1])
        # weight matrix
        WVec = self.v ** self.lmd * np.exp((self.x-xNew)**2/(-2*self.k**2))
        Wscale = WVec / np.max(WVec)

        # select the necessary x
        SelectIndex = np.where(Wscale > 0.01)[0]
        if len(SelectIndex) < self.minSum:
            return -1000
        WS, XS, YS = np.mat(np.diag(Wscale[SelectIndex])), self.X[SelectIndex, :], self.Y[SelectIndex, :]

        # compute theta for xNew
        theta = np.linalg.inv(XS.T * WS * XS) * XS.T * WS * YS

        # predict y
        yPred = XNew * theta 
        return yPred.item()

    def predict(self, xNewList):
        yPredList = [self.SinglePredict(xNew) for xNew in xNewList]
        return np.array(yPredList)
        

def EnergyCorrect(VelPick, spectrum, threshold=0.5):
    # keep the high energy picking
    EnergyEst = np.sum(spectrum, axis=1)
    EnergyEst /= np.max(EnergyEst)
    KeepIndex1 = np.where(EnergyEst > threshold)[0]
    # remove nonsense picking
    KeepIndex2 = np.where(VelPick > 0)[0]

    KeepList = sorted(list(set(KeepIndex1) & set(KeepIndex2)))
    
    return VelPick[KeepList]
