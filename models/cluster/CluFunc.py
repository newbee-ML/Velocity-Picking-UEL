import argparse
import os

import matplotlib
import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter

matplotlib.use('Agg')
from data.config import SaveRoot
from data.LoadData import LoadData
from models.cluster.ClusterMethod import MyDBSCAN, MyKMeans
from models.Postprocessing import interpolation
from models.Preprocess import Img2Point, LocalScale
from utils.CheckPath import CheckFolder
from utils.LogPara import LogParameters
from utils.LogWriter import Logger
from utils.metrics import VMAE, VMRE, MeanDeviation, PickRate
from utils.PlotTools import PlotSpec


def GetPara():
    parser = argparse.ArgumentParser()
    parser.add_argument('--SetName', type=str, default='A', help='Dataset Name')
    parser.add_argument('--SaveFoldName', type=str, default='UEL', help='Dataset Name')
    parser.add_argument('--EpName', type=str, default='Ep-1', help='Dataset Root Path')
    parser.add_argument('--TestNum', type=int, default=200)
    parser.add_argument('--Resave', type=int, default=1)
    parser.add_argument('--wh', type=int, default=30, help="the height of the local mean window")
    parser.add_argument('--visual', type=int, default=1)
    parser.add_argument('--VisualNum', type=int, default=16)


    #####################
    # ablation setting
    #####################
    # cluster method
    parser.add_argument('--ClusM', type=str, default=1)  
    # use gain method
    parser.add_argument('--UseGain', type=int, default=1)  

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
    writer = SummaryWriter(os.path.join(SaveLog))
    # save the hyperparameters
    LogParameters(opt, os.path.join(SaveLog, 'Parameters.csv'))
    # get dataloader
    DataLoader = LoadData(opt.SetName)
    
    ###################################
    # main flow
    ###################################
    IndexList = DataLoader.TestIndexList[:opt.TestNum]
    PlotCount = 1

    for ind, index in enumerate(IndexList):
        # get datadict
        DataDict = DataLoader.GetDataDict(index)
        SpecData = DataDict['Current']
        LMSpec = LocalScale(SpecData['spectrum'], width=20)
        t0Vec, vVec = SpecData['scale']['t0'], SpecData['scale']['v']
        OriPoints = Img2Point(LMSpec, t0Vec, vVec, False, threshold=0.3)
        try:
            # picking processing
            if opt.ClusM == 'KM':
                CluVelPick = MyKMeans(OriPoints[:, :2], 15)
            elif opt.ClusM == 'DBSCAN':
                CluVelPick = MyDBSCAN(OriPoints[:, :2], 50, 3)
            else:
                raise ValueError
            AP = interpolation(CluVelPick, t0Vec, vVec)
        except:
            continue
        # compute metrics and save results
        line, cdp = index.split('-')
        MP = interpolation(SpecData['label'], t0Vec, vVec)
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
        log.logger.info('Line %s\tCDP %s\tVMAE %.3f\tVMER %.3f\tPR %.3f\tMD %.3f\tCenter Num %d' % (line, cdp, vmae, vmre, PR, MD, CluVelPick.shape[0]))
        ResultList.append([line, cdp, vmae, vmre, PR, MD, CluVelPick.shape[0]])
        # log2: write in the DF: line | cdp | time | velocity
        PickDict['line'] += CluVelPick.shape[0] * [int(line)]
        PickDict['trace'] += CluVelPick.shape[0] * [int(cdp)]
        PickDict['time'] += CluVelPick[:, 0].astype(np.int32).tolist()
        PickDict['velocity'] += CluVelPick[:, 1].astype(np.int32).tolist()
        # log3: output the tensorboardX
        writer.add_scalar('VMAE', vmae, global_step=ind)
        writer.add_scalar('VMRE', vmre, global_step=ind)
        writer.add_scalar('CenterNum', CluVelPick.shape[0], global_step=ind)
        if PlotCount < opt.VisualNum:
            PlotSpec(LMSpec, t0Vec, vVec, title=None, VelCurve=[AP, MP], VelPick=CluVelPick, save_path=os.path.join(SaveFig, '%s-1PWRwMPick-%s.pdf'%(index, opt.EpName)))

    # write the auto picking result to csv
    PickDF = pd.DataFrame(PickDict)
    PickDF.to_csv(os.path.join(SaveLog, 'AutoPicking.csv'), index=False)
    ColName = ['Line', 'CDP', 'VMAE', 'VMRE', 'PR', 'MD', 'PickNum']
    ResultDF = pd.DataFrame(ResultList, columns=ColName)
    ResultDF.to_csv(os.path.join(SaveLog, 'TestResults.csv'), index=False)
    