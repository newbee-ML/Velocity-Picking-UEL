import numpy as np
from scipy import interpolate
from torchvision import transforms
from data.config import SetPath
import os
import h5py
import segyio
from models.UEL.GetRefVel import *

"""
Loading Sample Data from segy, h5, npy file
"""


# make ground truth curve
def interpolation(label_point, t_interval, v_interval=None):
    # sort the label points
    label_point = np.array(sorted(label_point, key=lambda t_v: t_v[0]))

    # ensure the input is int
    t0_vec = np.array(t_interval).astype(int)

    # get the ground truth curve using interpolation
    peaks_selected = np.array(label_point)
    func = interpolate.interp1d(peaks_selected[:, 0], peaks_selected[:, 1], kind='linear', fill_value="extrapolate")
    y = func(t0_vec)
    if v_interval is not None:
        v_vec = np.array(v_interval).astype(int) 
        y = np.clip(y, v_vec[0], v_vec[-1])

    return np.hstack((t0_vec.reshape((-1, 1)), y.reshape((-1, 1))))


################################################################
# Load Basic File
################################################################
def SynLoadFile(RootPath):
    """
    Load 1. dataset info dict; 2. Gth npy path dict; 3. Pwr npy path dict
    return:
        ModelInfoDict, GthList, PwrList
    """
    # dataset info dict
    ModelInfoDict = np.load(os.path.join(RootPath, 'ModelInfo.npy'), allow_pickle=True).item()

    # Gth npy path Dict
    GthList = os.listdir(os.path.join(RootPath, 'gth'))
    GthDict = {file.strip('.npy').strip('gth-'): os.path.join(RootPath, 'gth', file) for file in GthList}
    # PWR npy path list
    PwrList = os.listdir(os.path.join(RootPath, 'pwr'))
    PwrDict = {file.strip('.npy').strip('pwr-'): os.path.join(RootPath, 'pwr', file) for file in PwrList}

    # label indexs
    LabelIndex = []
    for line in list(ModelInfoDict['VelModel'].keys()):
        LabelIndex += ['%s-%s' % (line, cdp) for cdp in list(ModelInfoDict['VelModel'][line].keys())]
    # Gth indexs
    GthIndex = list(GthDict.keys())
    # Pwr indexs
    PwrIndex = list(PwrDict.keys())
    # find the common index
    IndexList = sorted(list(set(LabelIndex) & set(GthIndex) & set(PwrIndex)))

    return ModelInfoDict, GthDict, PwrDict, IndexList


################################################################
# load data from ModelInfoDict, GthList, PwrList
################################################################
class SynLoadData:
    def __init__(self, ModelInfoDict, GthDict, PwrDict):
        self.ModelInfo = ModelInfoDict
        self.GthList = GthDict
        self.PwrList = PwrDict
    
    ################# load data from segy, h5 and label.npy ####################
    def SingleDataDict(self, index):
        # data dict
        DataDict = {}
        line, cdp = index.split('-')

        # Load Spectrum, stk, gather 
        DataDict.setdefault('spectrum', np.load(self.PwrList[index]))
        DataDict.setdefault('gth', np.load(self.GthList[index]))

        # Load scale information
        ScaleInfo = {'t0': self.ModelInfo['t0Vec'],
                     'v': self.ModelInfo['VelModel'][line][cdp]['vVec'],
                     't-gth': self.ModelInfo['tVec'],
                     'offset': self.ModelInfo['VelModel'][line][cdp]['offVec']}

        DataDict.setdefault('scale', ScaleInfo)

        try:
            DataDict.setdefault('label', np.array(self.ModelInfo['VelModel'][line][cdp]['RMSVel']))
        except:
            DataDict.setdefault('label', None)

        return DataDict
    
    def SingleSpec(self, index):
        line, cdp = index.split('-')
        return {'spectrum': np.array(np.load(self.PwrList[index])), 
                'scale': {'v': self.ModelInfo['VelModel'][line][cdp]['vVec']}}

    def SingleLabel(self, index):
        line, cdp = index.split('-')
        try:
            return np.array(self.ModelInfo['VelModel'][line][cdp]['RMSVel'])
        except KeyError:
            return np.array(self.ModelInfo['VelModel'][int(line)][int(cdp)]['RMSVel'])


################################################################
# Field: Load Basic File
################################################################
def FieldLoadFile(RootPath):
    """
    Load Segy, h5, label file from the root directory
    return:
        SegyDict, H5Dict, LabelDict
        ---------------------------
        SegyDict: include pwr, stk, gth three segy information
        H5Dict: include pwr, stk, gth three index information
        LabelDict: include all labels of Spectra 
    """
    # load segy data
    SegyName = {'pwr': 'vel.pwr.sgy',
                'stk': 'vel.stk.sgy',
                'gth': 'vel.gth.sgy'}
    SegyDict = {}
    for name, path in SegyName.items():
        SegyDict.setdefault(name, segyio.open(os.path.join(RootPath, 'segy', path), "r", strict=False))
    # load h5 file
    H5Name = {'pwr': 'SpecInfo.h5',
              'stk': 'StkInfo.h5',
              'gth': 'GatherInfo.h5'}
    H5Dict = {}
    for name, path in H5Name.items():
        H5Dict.setdefault(name, h5py.File(os.path.join(RootPath, 'h5File', path), 'r'))

    # load label.npy
    LabelDict = np.load(os.path.join(RootPath, 't_v_labels.npy'), allow_pickle=True).item()

    return SegyDict, H5Dict, LabelDict


################################################################
# Field: Find the Index of the Useful Sample
################################################################
def FieldFindIndex(H5Dict, LabelDict):
    """
    Get the index of the useful samples from the H5Dict and LabelDict
    return;
        AllIndex, LabeledIndex
        -----------------------------
        AllIndex: the samples have gth, stk, spec at the same time
        LabeledIndex: the samples in AllIndex have spectrum labels
    """
    # Get index of pwr, gth, stk and labels
    PwrIndex = set(H5Dict['pwr'].keys())
    StkIndex = set(H5Dict['stk'].keys())
    GthIndex = set(H5Dict['gth'].keys())
    HaveLabelIndex = []
    for lineN in LabelDict.keys():
        for cdpN in LabelDict[lineN].keys():
            HaveLabelIndex.append('%s_%s' % (lineN, cdpN))

    # Find the same part of these sets and sort them
    AllIndex = sorted(list((PwrIndex & StkIndex) & GthIndex))
    LabeledIndex = sorted(list((PwrIndex & StkIndex) & (GthIndex & set(HaveLabelIndex))))
    
    AllIndex = [index.replace('_', '-') for index in AllIndex]
    LabeledIndex = [index.replace('_', '-') for index in LabeledIndex]

    return AllIndex, LabeledIndex


################################################################
# Field: load data from segy, h5 and label.npy 
################################################################
class FieldLoadData:
    def __init__(self, SegyDict, H5Dict, LabelDict):
        self.SegyDict = SegyDict
        self.H5Dict = H5Dict
        self.LabelDict = LabelDict
    
    ################# load data from segy, h5 and label.npy ####################
    def SingleDataDict(self, index):
        SegyDict, H5Dict, LabelDict = self.SegyDict, self.H5Dict, self.LabelDict
        # data dict
        DataDict = {}
        index = index.replace('-', '_')
        PwrIndex = np.array(H5Dict['pwr'][index]['SpecIndex'])
        GthIndex = np.array(H5Dict['gth'][index]['GatherIndex'])
        line, cdp = index.split('_')

        # Load Spectrum, stk, gather 
        DataDict.setdefault('spectrum', np.array(SegyDict['pwr'].trace.raw[PwrIndex[0]: PwrIndex[1]].T))

        DataDict.setdefault('gth', np.array(SegyDict['gth'].trace.raw[GthIndex[0]: GthIndex[1]].T))

        # Load scale information
        ScaleInfo = {'t0': np.array(SegyDict['pwr'].samples),
                    'v': np.array(SegyDict['pwr'].attributes(segyio.TraceField.offset)[PwrIndex[0]: PwrIndex[1]]),
                    't-stk': np.array(SegyDict['stk'].samples),
                    't-gth': np.array(SegyDict['gth'].samples),
                    'offset': np.array(SegyDict['gth'].attributes(segyio.TraceField.offset)[GthIndex[0]: GthIndex[1]])}

        DataDict.setdefault('scale', ScaleInfo)

        try:
            try:
                DataDict.setdefault('label', np.array(LabelDict[int(line)][int(cdp)]))
            except KeyError:
                DataDict.setdefault('label', np.array(LabelDict[str(line)][str(cdp)]))
        except:
            DataDict.setdefault('label', None)

        return DataDict
    
    ####################### get single spectrum data #########################
    def SingleSpec(self, index):
        index = index.replace('-', '_')
        PwrIndex = np.array(self.H5Dict['pwr'][index]['SpecIndex'])
        vVec = np.array(self.SegyDict['pwr'].attributes(segyio.TraceField.offset)[PwrIndex[0]: PwrIndex[1]])
        spec = np.array(self.SegyDict['pwr'].trace.raw[PwrIndex[0]: PwrIndex[1]].T)
        return {'spectrum': spec, 'scale': {'v': vVec}}
    
    ####################### get single velocity label ########################
    def SingleLabel(self, index):
        index = index.replace('-', '_')
        line, cdp = index.split('_')
        try:
            return np.array(self.LabelDict[line][cdp])
        except KeyError:
            return np.array(self.LabelDict[int(line)][int(cdp)])


class LoadData:
    def __init__(self, SetName):
        self.DataLoader = self.GetLoader(SetName)
    
    def GetLoader(self, SetName):
        SetRootPath = SetPath[SetName]
        if SetName in ['A']:
            # Load segy, h5 and labels file
            SegyDict, H5Dict, LabelDict = FieldLoadFile(SetRootPath)
            # Find useful samples
            self.IndexList, LabeledIndex = FieldFindIndex(H5Dict, LabelDict)
            self.SeedIndexList = MakeSeedIndex(LabeledIndex, rate=0.1)
            self.TestIndexList = sorted(list(set(LabeledIndex) - set(self.SeedIndexList)))
            self.IndexScale = GetScale(self.IndexList)
            print("All %d samples,  %d are labeled, %d are seeds" % (len(self.IndexList), len(LabeledIndex), len(self.SeedIndexList)))
            # Load the dataset
            return FieldLoadData(SegyDict, H5Dict, LabelDict)
        else:
            ModelInfoDict, GthDict, PwrDict, self.IndexList = SynLoadFile(SetPath[SetName])
            self.SeedIndexList = MakeSeedIndex(self.IndexList, rate=0.1)
            self.TestIndexList = sorted(list(set(self.IndexList)-set(self.SeedIndexList)))
            print("All %d samples,  %d are seeds" % (len(self.IndexList), len(self.SeedIndexList)))
            self.IndexScale = GetScale(self.IndexList)
            return SynLoadData(ModelInfoDict, GthDict, PwrDict)
    
    def GetDataDict(self, index): 
        DataDict = {}
        # get current datadict
        DataDict.setdefault('Current', self.DataLoader.SingleDataDict(index))
        
        # get seed information
        NearSeedIndex = GetSeedIndex(index, self.SeedIndexList, scale=self.IndexScale)
        t0Vec = DataDict['Current']['scale']['t0']
        vVec = DataDict['Current']['scale']['v']
        SeedRefVel = interpolation(self.DataLoader.SingleLabel(NearSeedIndex), t0Vec, vVec)
        DataDict.setdefault('Seed', SeedRefVel)

        # get near information
        NearIndex = GetNearIndex(index, self.IndexList, scale=self.IndexScale, k=3)
        NearSpecList = [self.DataLoader.SingleSpec(ind) for ind in NearIndex]
        NearSpecList = KeepSize(DataDict['Current'], NearSpecList)
        DataDict.setdefault('Near', {'spectrum': NearSpecList, 'index': NearIndex})

        return DataDict
    
