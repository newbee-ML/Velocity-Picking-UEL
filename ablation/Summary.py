import os
import pandas as pd
from data.config import SaveRoot
import numpy as np
import matplotlib.pyplot as plt

def Summary():
    if not os.path.exists('summary'):
        os.makedirs('summary')
    EpResultRoot = os.path.join(SaveRoot, 'UEL')
    EpList = os.listdir(EpResultRoot)
    ColName = ['EpName', 'SetName', 'VMAE', 'VMRE', 'PR', 'MD']
    TestList = []
    for Ep in EpList:
        ParaPath = os.path.join(EpResultRoot, Ep, 'logs', 'Parameters.csv')
        Resutlpath = os.path.join(EpResultRoot, Ep, 'logs', 'TestResults.csv')
        if not os.path.exists(Resutlpath): continue
        ParaDict = pd.read_csv(ParaPath).to_dict()
        TestDict = pd.read_csv(Resutlpath).to_dict()
        SetName = ParaDict['SetName'][0]
        VMAE = np.nanmean(list(TestDict['VMAE'].values()))
        VMRE = np.nanmean(list(TestDict['VMRE'].values()))
        try:
            PR = np.nanmean(list(TestDict['PR'].values()))
            MD = np.nanmean(list(TestDict['MD'].values()))
        except:
            continue
        TestList.append([Ep, SetName, VMAE, VMRE, PR, MD])
    TestDF = pd.DataFrame(TestList, columns=ColName)
    TestDF.to_csv(os.path.join('summary', 'UEL_TEST.csv'), index=False)
    

def SummarySyn():
    TabPath = os.path.join('summary', 'SynTestResult.xlsx')
    ResultDict = pd.read_excel(TabPath).to_dict()
    ResultDict = {key: np.array(list(NL.values())) for key, NL in ResultDict.items()}
    # plot VMAE and MD
    fig = plt.figure(figsize=(10, 6), dpi=300)
    Mark = ['o', 's', 'd', '*']
    loc = [+5, -5, -5, -5]
    color = ['b', 'orange', 'green', 'red']
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
    for ind, name in enumerate(['VMAE', 'VMRE', 'PR', 'MD']):
        y = ResultDict[name]
        ax.plot(ResultDict['SNR'], y, marker=Mark[ind], label=name)
        for ind2, val in enumerate(y):
            ax.text(ResultDict['SNR'][ind2], val+loc[ind], '%.2f'%val, ha='center', va='bottom', fontsize=10, c=color[ind])
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=13)
    plt.xlabel('S/N Ratio', fontsize=13)
    plt.ylabel('Value', fontsize=13)
    plt.savefig(os.path.join('summary', 'SNRTest.pdf'), dpi=200, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    # Summary()
    SummarySyn()