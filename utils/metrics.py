import numpy as np


# Euclidean distance
def Distance(x, y):
    return np.sqrt(np.sum((x-y)**2, axis=1))


# compute the VMAE
def VMAE(AutoCurve, ManualCurve):
    return np.mean(np.abs(AutoCurve[:, 1] - ManualCurve[:, 1]))


# compute the VMRE
def VMRE(AutoCurve, ManualCurve):
    return np.mean(np.abs(AutoCurve[:, 1] - ManualCurve[:, 1])/ManualCurve[:, 1])*100


# Mean deviation
def MeanDeviation(AP, MP, Threshold=200):
    MinDist = np.array([np.min(Distance(AP, mp)) for mp in MP])
    return np.mean(MinDist[MinDist<Threshold])


# Picking rate
def PickRate(AP, MP, Threshold=200):
    MinDist = np.array([np.min(Distance(AP, mp)) for mp in MP])
    return len(np.where(MinDist<Threshold)[0])/len(MP)*100


# statistics result picking result
def StatPick(AP, MP, Threshold=200):
    MinDist = np.array([np.min(Distance(AP, mp)) for mp in MP])
    PickRight = np.zeros_like(MinDist)
    PickRight[MinDist < Threshold] = 1
    return np.array([MP[:, 0], MinDist, PickRight]).T