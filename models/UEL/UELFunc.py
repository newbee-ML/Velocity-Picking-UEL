import argparse
import os

import matplotlib
import numpy as np
import pandas as pd
from models.UEL.UEL import UELPick

matplotlib.use('Agg')
from data.config import SaveRoot
from data.LoadData import LoadData
from models.Postprocessing import interpolation, NMOCorr
from models.UEL.UEL import UELPick
from utils.CheckPath import CheckFolder
from utils.LogPara import LogParameters
from utils.LogWriter import Logger
from utils.metrics import VMAE, VMRE, MeanDeviation, PickRate
from utils.PlotTools import PlotSpec, PlotGth, PlotNearPWR, RefVelPlot



def GetPara():
    parser = argparse.ArgumentParser()
    parser.add_argument('--SetName', type=str, default='hade', help='Dataset Name')
    parser.add_argument('--SaveFoldName', type=str, default='UEL', help='Dataset Name')
    parser.add_argument('--EpName', type=str, default='Ep-1', help='Dataset Root Path')
    parser.add_argument('--TestNum', type=int, default=200)
    parser.add_argument('--Resave', type=int, default=0)
    parser.add_argument('--wh', type=int, default=30, help="the height of the local mean window")
    parser.add_argument('--visual', type=int, default=1)
    parser.add_argument('--VisualNum', type=int, default=8)

    ##########################################
    # near spectra velocity prior parameters
    ##########################################
    parser.add_argument('--therT', type=float, default=0.3, help="the threshold of transform img to point set")
    parser.add_argument('--nearK', type=int, default=3, help="the distance of near samples")
    parser.add_argument('--blurK', type=int, default=5, help="the blur size of mean blur")
    parser.add_argument('--blurT', type=int, default=2, help="the blur time of mean blur")

    ######################
    # ASSF parameters
    ######################
    parser.add_argument('--minNum', type=int, default=20)
    parser.add_argument('--lam', type=float, default=1)
    parser.add_argument('--threD', type=int, default=30, help="the threshold of convergence")
    parser.add_argument('--threM', type=int, default=50, help="the threshold of merge")
    parser.add_argument('--sig0', type=int, default=2, help="the init sigma")
    parser.add_argument('--alp', type=float, default=1)

    #######################
    # ensemble parameters
    #######################
    parser.add_argument('--EnBwNear', type=float, default=250)
    parser.add_argument('--EnBwSeed', type=float, default=100)
    parser.add_argument('--EnIntV', type=float, default=100)  # 200

    #####################
    # ablation setting
    #####################
    # velocity constraints
    parser.add_argument('--UseVC', type=int, default=1) 
    # use gain method
    parser.add_argument('--UseGain', type=int, default=1)  
    # 0 all use 1 w/o seed reference 2 w/o near reference
    parser.add_argument('--EnsAblM', type=int, default=0)  

    optN = parser.parse_args()
    return optN


def MainFunc(opt):
    ###################################
    # preparation for estimation
    ###################################
    # Check Save File
    SaveRootN = os.path.join(SaveRoot, opt.SaveFoldName)
    CheckFolder(SaveRootN, opt.EpName, 'logs', opt.Resave)
    CheckFolder(SaveRootN, opt.EpName, 'figs', opt.Resave)
    SaveFig, SaveLog = os.path.join(SaveRootN, opt.EpName, 'figs'), os.path.join(SaveRootN, opt.EpName, 'logs')
    # set Logger
    log = Logger(os.path.join(SaveLog, 'VMAE.log'), level='info')
    PickDict = {'line': [], 'trace': [], 'time': [], 'velocity': []}
    ResultList = []
    # save the hyperparameters
    LogParameters(opt, os.path.join(SaveLog, 'Parameters.csv'))
    # get dataloader
    DataLoader = LoadData(opt.SetName)
    
    ###################################
    # main flow
    ###################################
    IndexList = DataLoader.TestIndexList[:opt.TestNum]
    Picker = UELPick(opt)
    PlotCount = 1

    for ind, index in enumerate(IndexList):
        # get datadict
        DataDict = DataLoader.GetDataDict(index)
        # picking processing
        try:
            Finalpicks, AP, CluPick, NearVel, SeedVel, LMSpec, NearPoints = Picker.pick(DataDict)
        except:
            continue
        # compute metrics and save results
        line, cdp = index.split('-')
        t0Vec, vVec = DataDict['Current']['scale']['t0'], DataDict['Current']['scale']['v']
        MP = interpolation(DataDict['Current']['label'], t0Vec, vVec)
        try:
            vmae = VMAE(AP, MP)
        except:
            continue
        try:
            vmre = VMRE(AP, MP)
        except:
            continue
        try:
            MD = MeanDeviation(AP, DataDict['Current']['label'])
        except:
            continue
        try:
            PR = PickRate(AP, DataDict['Current']['label'])
        except:
            continue

        # log1: index + VMAE
        log.logger.info('Line %s\tCDP %s\tVMAE %.3f\tVMER %.3f\tPR %.3f\tMD %.3f\tCenter Num %d' % (line, cdp, vmae, vmre, PR, MD, Finalpicks.shape[0]))
        ResultList.append([line, cdp, vmae, vmre, PR, MD, Finalpicks.shape[0]])
        # log2: write in the DF: line | cdp | time | velocity
        PickDict['line'] += Finalpicks.shape[0] * [int(line)]
        PickDict['trace'] += Finalpicks.shape[0] * [int(cdp)]
        PickDict['time'] += Finalpicks[:, 0].astype(np.int32).tolist()
        PickDict['velocity'] += Finalpicks[:, 1].astype(np.int32).tolist()
        def PlotModule():
            PlotSpec(DataDict['Current']['spectrum'], t0Vec, vVec, save_path=os.path.join(SaveFig, '%s-0Ori-%s.png'%(index, opt.EpName)))
            PlotSpec(LMSpec, t0Vec, vVec, save_path=os.path.join(SaveFig, '%s-0LM-%s.png'%(index, opt.EpName)))
            PlotSpec(LMSpec, t0Vec, vVec, VelCurve=[AP, MP], VelPick=Finalpicks, save_path=os.path.join(SaveFig, '%s-1PWRwMPick-%s.png'%(index, opt.EpName)))
            tVec, OffsetVec = DataDict['Current']['scale']['t-gth'], DataDict['Current']['scale']['offset']
            gth = DataDict['Current']['gth']
            APL = interpolation(AP, tVec, vVec)
            MPL = interpolation(DataDict['Current']['label'], tVec, vVec)
            NMOGth = NMOCorr(gth, tVec, OffsetVec, APL[:, 1])
            NMOGthGT = NMOCorr(gth, tVec, OffsetVec, MPL[:, 1])
            PlotGth(gth, tVec, OffsetVec, os.path.join(SaveFig, '%s-2OriGth-%s.png'%(index, opt.EpName)))
            PlotGth(NMOGth, tVec, OffsetVec, os.path.join(SaveFig, '%s-3NMOGth-%s.png'%(index, opt.EpName)))
            PlotGth(NMOGthGT, tVec, OffsetVec, os.path.join(SaveFig, '%s-4NMOGthGT-%s.png'%(index, opt.EpName)))
            PlotNearPWR(DataDict['Near']['spectrum'], DataDict['Near']['index'], os.path.join(SaveFig, '%s-5NearPwr-%s.png'%(index, opt.EpName)))
            RefVelPlot(NearPoints[:, :2], NearPoints[:, 2], NearVel, opt.EnBwNear, DataDict['Current']['label'], t0Vec, vVec, os.path.join(SaveFig, '%s-6NearPwrRef-%s.png'%(index, opt.EpName)))


        if PlotCount < opt.VisualNum:
            PlotModule()

    
    # write the auto picking result to csv
    PickDF = pd.DataFrame(PickDict)
    PickDF.to_csv(os.path.join(SaveLog, 'AutoPicking.csv'), index=False)
    ColName = ['Line', 'CDP', 'VMAE', 'VMRE', 'PR', 'MD', 'PickNum']
    ResultDF = pd.DataFrame(ResultList, columns=ColName)
    ResultDF.to_csv(os.path.join(SaveLog, 'TestResults.csv'), index=False)
    