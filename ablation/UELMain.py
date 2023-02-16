from models.UEL.UELFunc import GetPara, MainFunc
from utils.tuning import ListPara, ParaStr2Dict, UpdateOpt
import os

#################################
# experiment setting
#################################

PerformanceTest = {
    'SetName': ['str', ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'A']],
    'SaveFoldName': ['str', ['UEL']], 
    'minNum': ['int', [20]],
    'TestNum': ['int', [200]],
    'VisualNum': ['int', [64]],
    'visual': ['int', [1]],
}

TestSet = 'S5'
start = 11
AblationTest = {
    # w/o Spectrum Gain
    0: {'SetName': TestSet, 
        'SaveFoldName': 'Ablation',
        'UseGain': 0,},
    # w/o Near Velocity
    1: {'SetName': TestSet, 
        'SaveFoldName': 'Ablation',
        'EnsAblM': 2,},
    # w/o Seed Velocity
    2: {'SetName': TestSet, 
        'SaveFoldName': 'Ablation',
        'EnsAblM': 1,},
    # w/o Velocity Constraints
    3: {'SetName': TestSet, 
        'SaveFoldName': 'Ablation',
        'UseGain': 1,
        'EnsAblM': 0,
        'UseVC': 0},
    # ours
    4: {'SetName': TestSet, 
        'SaveFoldName': 'Ablation',
        'minNum': 30},

}

"""

"""
if __name__ == '__main__':
    TestMode = 1
    #######################################
    # performance test
    #######################################
    if TestMode == 1:
        Start = 1
        RepeatTime = 1
        EpList = ListPara(PerformanceTest)*RepeatTime
        print('Planning Tasks: Ep-%d ~ Ep-%d' % (Start, Start+len(EpList)-1))
        # get default training parameters
        OptDefault = GetPara()
        for ind, EpName in enumerate(EpList):
            if ind != 4:
                continue
            # try:
            start = Start
            BaseName = 'Ep-%d' % (ind+start)
            print('Test %s' % BaseName)
            EpDict = ParaStr2Dict(EpName, PerformanceTest)
            EpDict.setdefault('EpName', BaseName)
            # update the para
            EpOpt = UpdateOpt(EpDict, OptDefault)
            # start this experiment
            MainFunc(EpOpt)
    elif TestMode == 2:
        for ind, EpDict in AblationTest.items():
            if ind != 3: continue
            OptDefault = GetPara()
            BaseName = 'Ep-%d' % (ind+start)
            EpDict.setdefault('EpName', BaseName)
            EpOpt = UpdateOpt(EpDict, OptDefault)
            MainFunc(EpOpt)
    
