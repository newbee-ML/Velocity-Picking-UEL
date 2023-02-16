from models.cluster.CluFunc import GetPara, MainFunc
from utils.tuning import ListPara, ParaStr2Dict, UpdateOpt
import os

#################################
# experiment setting
#################################

PerformanceTest = {
    'ClusM': ['str', ['DBSCAN', 'KM']],
    'SetName': ['str', ['A']],
    'SaveFoldName': ['str', ['Clsuter']], 
    'TestNum': ['int', [200]],
    'VisualNum': ['int', [8]],
    'visual': ['int', [1]],
}

if __name__ == '__main__':
    #######################################
    # performance test
    #######################################
    Start = 1
    EpList = ListPara(PerformanceTest)
    print('Planning Tasks: Ep-%d ~ Ep-%d' % (Start, Start+len(EpList)-1))
    # get default training parameters
    OptDefault = GetPara()
    for ind, EpName in enumerate(EpList):
        # try:
        start = Start
        BaseName = 'Ep-%d' % (ind+start)
        EpDict = ParaStr2Dict(EpName, PerformanceTest)
        EpDict.setdefault('EpName', BaseName)
        # update the para
        EpOpt = UpdateOpt(EpDict, OptDefault)
        # start this experiment
        MainFunc(EpOpt)
    