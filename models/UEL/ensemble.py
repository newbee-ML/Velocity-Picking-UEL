"""
Ensemble Learning for velocity picking

Author: Hongtao Wang | stolzpi@163.com
"""

import copy
import numpy as np




# interval velocity compute to the rms velocity picking points
def IntervalVel(tv1, tv2):
    t1, v1 = tv1
    t2, v2 = tv2
    if t1 == t2:
        return 1e6
    else:
        diff = v2**2 * t2 - v1**2 * t1
        if diff < 0:
            return np.nan
        else:
            return np.sqrt((diff)/(t2 - t1))


class EnsemblePicking:
    """
    Ensemble Learning Class

    Params:
    ---
    NearRef: near reference velocity curve
    SeedRef: seed reference velocity curve
    vVec: velocity vector
    bw: the bandwidth of the realiable interval 
    """
    def __init__(self, NearRef, SeedRef, vVec, bwNear=300, bwSeed=100, MinIntv=300, AblMode=0, UseVC=1):
        self.NearRef = NearRef
        self.SeedRef = SeedRef
        self.vVec = vVec
        self.bwNear = bwNear
        self.bwSeed = bwSeed
        self.MinIntv = MinIntv
        self.AblMode = AblMode
        self.UseVC = UseVC


    # interval velocity constraint
    def IntVelConst(self, VelPick):
        vMin, vMax = 1000, 8000
        VelPickCp = copy.deepcopy(VelPick).tolist()
        VelPickCp = np.array(sorted(VelPickCp, key=lambda t_v: t_v[0]))
        VelPickNew = copy.deepcopy(VelPickCp).tolist()
        for i in range(VelPickCp.shape[0]-1):
            # print(VelPickCp[i, 0])
            if VelPickCp[i+1, 0] - VelPickCp[i, 0] > self.MinIntv:  # the dix formulation only fits the time diff under 200ms
                VInti = IntervalVel(VelPickCp[i, :], VelPickCp[i+1, :])
                if VInti < vMin or VInti > vMax or np.isnan(VInti):  # interval velocity unreasonable
                    # print('remove', VelPickCp[self.RemoveBad(i, VelPickCp)])
                    VelPickNew.pop(self.RemoveBad(i, VelPickCp))
                    return self.IntVelConst(np.array(VelPickNew))
                
            else:  # remove the bad one
                # print('remove', VelPickCp[self.RemoveBad(i, VelPickCp)])
                VelPickNew.pop(self.RemoveBad(i, VelPickCp))
                return self.IntVelConst(np.array(VelPickNew))

        return np.array(VelPickNew)
            
    # choose one between two picking
    def RemoveBad(self, i, VelPickCp):
        tv1NearDist, tv2NearDist = self.SeedRefDistance(VelPickCp[i, :]), self.SeedRefDistance(VelPickCp[i+1, :])
        if tv1NearDist > tv2NearDist:
            remove = i
        else:
            remove = i + 1
        return remove

    # compute the distance between the point and the near reference velocity
    def NearRefDistance(self, point): 
        return np.min(np.sqrt(np.sum((point - self.NearRef)**2, axis=1)))

    # compute the distance between the point and the seed reference velocity
    def SeedRefDistance(self, point):
        return np.min(np.sqrt(np.sum((point - self.SeedRef)**2, axis=1)))

    # the main function to ensemble learning
    def select(self, VelPick):
        # compute the distance between velocity picking points and reference velocity curve
        NearRefDist = np.array([self.NearRefDistance(point) for point in VelPick])
        SeedRefDist = np.array([self.SeedRefDistance(point) for point in VelPick])

        # select the realiable point
        SelectNear = np.where(NearRefDist < self.bwNear)[0]
        SelectSeed = np.where(SeedRefDist < self.bwSeed)[0]
        
        # ablation setting
        if self.AblMode==0:
            FinalSelect = list(set(SelectNear) & set(SelectSeed))
        elif self.AblMode==1:
            FinalSelect = list(SelectNear)
        elif self.AblMode==2:
            FinalSelect = list(SelectSeed)

        SaveVelPick = VelPick[FinalSelect, :]

        # # interval velocity constraint
        if self.UseVC:
            FinalVelPick = self.IntVelConst(SaveVelPick)
        else:
            FinalVelPick = SaveVelPick 
        return FinalVelPick
