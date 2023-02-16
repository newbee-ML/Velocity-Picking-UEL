"""
Main flow of UEL for velocity picking

------
Author: Hongtao Wang | stolzpi@163.com
"""


import numpy as np
from models.Postprocessing import interpolation, interpolation2
from models.Preprocess import Img2Point, LocalScale
from models.cluster.ClusterMethod import MyDBSCAN, MyKMeans
from models.cluster.SSF import SSF
from models.UEL.ASSF import AttSSF as ASSFM
from models.UEL.ensemble import EnsemblePicking
from models.UEL.GetRefVel import ILWLR, GetLowFrequencyMap, KeepSize


class UELPick:
    # load hyperparameters and dataloader class
    def __init__(self, opt):
        self.opt = opt
        self.ASSF = ASSFM(opt.threD, opt.threM, opt.sig0, opt.alp, opt.lam, opt.minNum)
    
    def pick(self, DataDict):
        #######################################
        # load data  
        #######################################
        SpecData, NearData, SeedLabel = DataDict['Current'], DataDict['Near']['spectrum'], DataDict['Seed']
        t0Vec, vVec = SpecData['scale']['t0'], SpecData['scale']['v']

        #######################################
        # estimate near reference RMS velocity 
        #######################################
        # Get the info of near samples
        if self.opt.UseGain:  # ablation setting
            NearSpec = NearData
            NearSpec = np.array([LocalScale(SingleOne, width=self.opt.wh) for SingleOne in NearSpec])
        else:
            NearSpec = np.array(NearData)

        # Generate the low freq info of the near samples
        NearSpec = GetLowFrequencyMap(NearSpec, k=self.opt.blurK, time=self.opt.blurT)
        # Get the velocity prior
        NearPoints = Img2Point(NearSpec, t0Vec, vVec, False, threshold=0.05)
        lwlr = ILWLR(NearPoints[:, 0], NearPoints[:, 1], NearPoints[:, 2], 300, lmd=5, minSum=2)
        NearPredVel = lwlr.predict(t0Vec)
        NearRefVel = np.hstack((t0Vec.reshape(-1, 1), NearPredVel.reshape(-1, 1)))

        #########################
        # load seed RMS velocity 
        #########################
        SeedRefVel = interpolation(SeedLabel, t0Vec, vVec) 

        ################################################
        # estimate current spec RMS velocity using ASSF
        ################################################
        if self.opt.UseGain:  # ablation setting
            LMSpec = LocalScale(SpecData['spectrum'], width=self.opt.wh)
        else:
            LMSpec = np.array(SpecData['spectrum'])

        OriPoints = Img2Point(LMSpec, t0Vec, vVec, False, threshold=self.opt.therT)

        CluVelPick = self.ASSF.cluster(OriPoints[:, :2], OriPoints[:, 2])
        ##########################################################
        # ensemble picking method based on above three velocities 
        ##########################################################
        Ensemer = EnsemblePicking(NearRefVel, SeedRefVel, vVec, bwNear=self.opt.EnBwNear, bwSeed=self.opt.EnBwNear, MinIntv=self.opt.EnIntV, AblMode=self.opt.EnsAblM, UseVC=self.opt.UseVC)
        FinalCenters = Ensemer.select(CluVelPick)
        AP = interpolation2(FinalCenters, t0Vec, vVec, SeedRefVel)

        return FinalCenters, AP, CluVelPick, NearRefVel, SeedRefVel, LMSpec, NearPoints
        