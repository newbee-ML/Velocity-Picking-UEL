"""
Visual the results of EP1

"""

import argparse
import os
import random

import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter

from ASSF.ASSF import AttSSF as ASSFM
from ASSF.ensemble import EnsembleFlow
from ASSF.GetRefVel import *
from ASSF.Pastprocessing import interpolation, interpolation2
from ASSF.Preprocess import Img2Point, LocalScale
from utils.CheckPath import CheckFolder
from utils.LoadData import FindIndex, LoadData, LoadFile
from utils.LogPara import LogParameters
from utils.LogWriter import Logger
from utils.metrics import VMAE
from utils.PlotTools import *


# LN method visual
def PrePolot():
    # Original CMP gather
    gth = DataDict['gth']
    W_Plot(gth, DataDict['scale']['offset'], DataDict['scale']['t-gth'], 'Offset (m)',
           SavePath=os.path.join('PaperPlot', '0-1-CMP.png'))
    # Original Spectrum
    pwr = DataDict['spectrum']
    PlotSpec(pwr, DataDict['scale']['t0'], DataDict['scale']['v'], 
             save_path=os.path.join('PaperPlot', '0-2-OriPwr.png'))
    # Gain Spectrum
    GainPwr = LocalScale(pwr, width=15)
    PlotSpec(GainPwr, DataDict['scale']['t0'], DataDict['scale']['v'], 
             save_path=os.path.join('PaperPlot', '0-3-GainPwr.png'))
    

# near reference velocity visual
def NearPlot():
    NearIndex = GetNearIndex(index, AllIndex, scale=IndexScale, k=2)
    NearSpec = [(LoadClass.SingleSpec(ind)) for ind in NearIndex]
    NearSpec = KeepSize(DataDict, NearSpec)
    PlotSpec(NearSpec[0], DataDict['scale']['t0'], DataDict['scale']['v'], 
             save_path=os.path.join('PaperPlot', '1-1-NearOriPwr.png'))
    NearSpec = np.array([LocalScale(SingleOne, width=15) for SingleOne in NearSpec])
    PlotSpec(NearSpec[0], DataDict['scale']['t0'], DataDict['scale']['v'], 
             save_path=os.path.join('PaperPlot', '1-2-NearGainPwr.png'))
    # Generate the low freq info of the near samples
    NearSpec = GetLowFrequencyMap(NearSpec, k=5, time=1)
    PlotSpec(NearSpec, DataDict['scale']['t0'], DataDict['scale']['v'], 
             save_path=os.path.join('PaperPlot', '1-3-NearPriorMap.png'))
    # Get the velocity prior
    NearPoints = Img2Point(NearSpec, t0Vec, vVec, False)
    lwlr = ILWLR(NearPoints[:, 0], NearPoints[:, 1], NearPoints[:, 2], 200, lmd=3, minSum=2)
    NearPredVel = lwlr.predict(t0Vec)
    NearRefVel = np.hstack((t0Vec.reshape(-1, 1), NearPredVel.reshape(-1, 1)))
    NearRefVel = EnergyCorrect(NearRefVel, NearSpec)  # "NearRefVel" is we need
    

# ep 2 
def EP2Plot():
    # seed distribution
    SampleDict = {'Seed': SeedIndexList, 'Other': TestIndexList}
    PlotSampleDistributions(SampleDict, SavePath=os.path.join('PaperPlot', '3-1-SeedDistribution.png'))
    # # near distribution
    # NearIndex = GetNearIndex(index, AllIndex, scale=IndexScale, k=2)
    # SampleS = [elm.split('_') for elm in GetNearIndex(index, AllIndex, scale=IndexScale, k=3)]
    # SampleA = np.array(SampleS, dtype=np.int).reshape((-1, 2))
    # LineList, CdpList = list(set(SampleA[:, 0])), list(set(SampleA[:, 1]))
    # OtherIndex = []
    # for line in LineList:
    #     for cdp in CdpList:
    #       OtherIndex.append('%d_%d'%(line, cdp))   
    # OtherIndex = list(set(OtherIndex) - set(NearIndex) - set([index]))
    # SampleDict2 = {'Current Spectrum': [index], 'Near Spectra': NearIndex, 'Other Spectra': OtherIndex}
    # PlotSampleDistributions2(SampleDict2, SavePath=os.path.join('PaperPlot', '3-2-NearDistribution.png'))
    # # near velocity map


if __name__ == '__main__':
    k = 200
    root = '/home/colin/data/Spectrum/hade'
    SegyDict, H5Dict, LabelDict = LoadFile(root)
    AllIndex, LabeledIndex = FindIndex(H5Dict, LabelDict)
    SeedIndexList = MakeSeedIndex(LabeledIndex, rate=0.1)
    print(len(SeedIndexList), len(AllIndex))
    LoadClass = LoadData(SegyDict, H5Dict, LabelDict)
    TestIndexList = sorted(list(set(AllIndex) - set(SeedIndexList)))
    index = TestIndexList[k]
    DataDict = LoadClass.SingleDataDict(index, mode='test')
    IndexScale = GetScale(AllIndex)
    t0Vec, vVec = DataDict['scale']['t0'], DataDict['scale']['v']
    EP2Plot()
    pass