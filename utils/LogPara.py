"""
A function to save the hyperparameters

Author: Hongtao Wang
"""

import copy
import pandas as pd

def LogParameters(opt, SavePath):
    ParaDict = copy.deepcopy(opt.__dict__)
    for key, val in ParaDict.items():
        ParaDict[key] = [ParaDict[key]]
    ParaDF = pd.DataFrame(ParaDict)
    ParaDF.to_csv(SavePath)
