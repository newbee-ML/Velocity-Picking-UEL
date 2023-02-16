import matplotlib.pyplot as plt
import matplotlib.animation as ani
import matplotlib as mpl 
from torchvision import transforms
import numpy as np
import copy
import os
import pandas as pd
import seaborn as sns
import cv2


"""
A few functions to Plot results
"""


# Basic plot setting
def plot_basic_set():
    plt.rcParams['font.sans-serif'] = 'Times New Roman'


# change axis set
def set_axis(x_dim, y_dim):
    x_dim, y_dim = x_dim.astype(int), y_dim.astype(int)
    InterX, InterY = int(len(x_dim)/4), int(len(y_dim)/5)
    show_x = range(0, len(x_dim), InterX)
    show_y = range(0, len(y_dim), InterY)
    x_d, y_d = x_dim[1] - x_dim[0], y_dim[1] - y_dim[0]
    ori_x = [x_dim[0]+i*x_d for i in show_x]
    ori_y = [y_dim[0]+i*y_d for i in show_y]
    plt.xticks(show_x, ori_x)
    plt.yticks(show_y, ori_y)


# Plot CMP gather and NMO CMP gather
def W_Plot(traces, xVec, yVec, xlab='Trace Index', 
           ylab='Time (ms)', color='black', norm='All', SavePath=None):

    ################################################################
    # Basic Index and Interval Setting
    ################################################################    
    if xVec is None:
        xVec = np.range(traces.shape[1])
    if yVec is None:
        yVec = np.range(traces.shape[0])

    xIndex = np.linspace(np.min(xVec), np.max(xVec), traces.shape[1])
    xInt = (xIndex[1] - xIndex[0]) * 0.8
    TracesCp = copy.deepcopy(traces)

    ################################################################
    # Scale each traces
    ################################################################

    # Split the positive and negative part of the traces
    TracesCpPos = np.zeros_like(TracesCp)
    TracesCpPos[TracesCp > 0] = TracesCp[TracesCp > 0]
    TracesCpNeg = np.zeros_like(TracesCp)
    TracesCpNeg[TracesCp < 0] = TracesCp[TracesCp < 0]

    # Scale the positive and negative parts
    if norm == 'All':
        MaxValue, MinValue = TracesCpPos.max(), TracesCpPos.min()
        Scale = np.max((MaxValue, -MinValue))
        TracesCpPos /= (Scale / xInt)
        TracesCpNeg /= (Scale / xInt)
    else:
        MaxValue, MinValue = np.max(TracesCpPos, axis=0), np.min(TracesCpNeg, axis=0)
        Scale = np.max(np.array([MaxValue, -MinValue]), axis=0)
        TracesCpPos /= (Scale / xInt)
        TracesCpNeg /= (Scale / xInt)
    
    ################################################################
    # Plot the wiggle figure of the traces
    ################################################################

    _, ax = plt.subplots(figsize=(2, 10), dpi=90)  # int(2*traces.shape[1]/50) + 1
    for i in range(TracesCp.shape[1]):
        ax.fill_betweenx(yVec, xIndex[i], xIndex[i] + TracesCpPos[:, i], facecolor=color, interpolate=True, alpha=1)
        # ax.plot(xIndex[i] + TracesCpNeg[:, i], yVec, c=color, linewidth=0.1)
        # ax.fill_betweenx(yVec, xIndex[i] + TracesCpNeg[:, i], xIndex[i], facecolor='white', interpolate=True)

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_xlim(min(xVec)-xInt, max(xVec)+xInt)
    ax.set_ylim(min(yVec), max(yVec))
    ax.invert_yaxis()

    if SavePath is None:
        plt.show()
    else:
        plt.savefig(SavePath, dpi=100, bbox_inches='tight')
    plt.clf()
    plt.close()

# GrayGather
def GrayGather(gather, xVec, yVec, SavePath=None):
    origin_pwr = 255 - (gather - np.min(gather)) / (np.max(gather) - np.min(gather)) * 255
    data_plot = origin_pwr.astype(np.uint8).squeeze()
    # data_plot_hot = cv2.applyColorMap(data_plot, cv2.COLORMAP_JET)  # COLORMAP_JET COLORMAP_HOT
    set_axis(xVec, yVec)
    plt.figure(figsize=(3, 10), dpi=300)
    plt.imshow(data_plot, cmap='gray_r', aspect='auto')
    plt.savefig(SavePath, dpi=150, bbox_inches='tight')
    plt.close()

# Plot spectrum with AP and MP
def PlotSpec(spectrum, t0_vec, v_vec, title=None, VelCurve=None, RefCurve=None, VelPick=None, CluPick=None, save_path=None):
    if len(t0_vec) != spectrum.shape[0]:
        t0_vec = np.linspace(t0_vec[0], t0_vec[-1], spectrum.shape[0])
    if len(v_vec) != spectrum.shape[1]:
        v_vec = np.linspace(v_vec[0], v_vec[-1], spectrum.shape[1])
    # origin_pwr = (spectrum - np.min(spectrum)) / (np.max(spectrum) - np.min(spectrum))
    # origin_pwr = spectrum
    data_plot = spectrum
    # data_plot = origin_pwr.astype(np.uint8).squeeze()
    plt.figure(figsize=(3, 10), dpi=300)
    label = ['Auto Velocity', 'Manual Velocity']
    col = ['r', 'g']
    if VelCurve is not None:
        for ind, VelC in enumerate(VelCurve):
            VCCP = copy.deepcopy(VelC)
            VCCP[:, 0] = (VCCP[:, 0]-t0_vec[0]) / (t0_vec[1]-t0_vec[0])
            VCCP[:, 1] = (VCCP[:, 1]-v_vec[0]) / (v_vec[1]-v_vec[0])
            plot_curve(VCCP, col[ind], label[ind])   
        plt.legend(loc=1)
    label = ['NearRef', 'SeedRef']
    col = ['y', 'g']
    if RefCurve is not None:
        for ind, VelC in enumerate(RefCurve):
            VCCP = copy.deepcopy(VelC)
            VCCP[:, 0] = (VCCP[:, 0]-t0_vec[0]) / (t0_vec[1]-t0_vec[0])
            VCCP[:, 1] = (VCCP[:, 1]-v_vec[0]) / (v_vec[1]-v_vec[0])
            plot_curve(VCCP, col[ind], label[ind])   
        plt.legend(loc=1)
    if VelPick is not None:
        VCCP = np.array(copy.deepcopy(VelPick))
        VCCP[:, 0] = (VCCP[:, 0]-t0_vec[0]) / (t0_vec[1]-t0_vec[0])
        VCCP[:, 1] = (VCCP[:, 1]-v_vec[0]) / (v_vec[1]-v_vec[0])
        plt.scatter(x=VCCP[:, 1], y=VCCP[:, 0], c='red',  
                    marker='x', label='AutoPick', s=20, linewidth=2)
        plt.legend(loc=1)
    if CluPick is not None:
        VCCP = np.array(copy.deepcopy(CluPick))
        VCCP[:, 0] = (VCCP[:, 0]-t0_vec[0]) / (t0_vec[1]-t0_vec[0])
        VCCP[:, 1] = (VCCP[:, 1]-v_vec[0]) / (v_vec[1]-v_vec[0])
        plt.scatter(x=VCCP[:, 1], y=VCCP[:, 0], c='green',  
                    marker='.', label='ClusterPick', s=15, linewidth=2)
        plt.legend(loc=1)
    im = plt.imshow(data_plot, cmap='seismic', aspect='auto')
    plt.colorbar(im, fraction=0.05, pad=0.04)
    plt.xlabel('Velocity (m/s)')
    set_axis(v_vec, t0_vec)
    plt.ylabel('Time (ms)')
    if title is not None:
        plt.title(title)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.clf()
    plt.close()


def PlotGth(gth, tVec, OffsetVec, SavePath=None):
    if len(tVec) != gth.shape[0]:
        tVec = np.linspace(tVec[0], tVec[-1], gth.shape[0])
    if len(OffsetVec) != gth.shape[1]:
        OffsetVec = np.linspace(OffsetVec[0], OffsetVec[-1], gth.shape[1])
    plt.figure(figsize=(3, 10), dpi=300)
    if np.max(gth) > 1000:
        gth /= 10000
    im = plt.imshow(gth, cmap='seismic', aspect='auto')
    plt.colorbar(im, fraction=0.05, pad=0.04)
    set_axis(OffsetVec, tVec)
    plt.xlabel('Offset (m)')
    plt.ylabel('Time (ms)')
    if SavePath is None:
        plt.show()
    else:
        plt.savefig(SavePath, dpi=200, bbox_inches='tight')
    plt.clf()
    plt.close()

# PointsPlot
def PlotPoints(points, values, t0Vec, vVec, title=None, save_path=None):
    plt.figure(figsize=(2, 10), dpi=300)
    col = plt.cm.jet(values)
    fig = plt.scatter(x=points[:, 1], y=points[:, 0], c=col, s=0.3)
    plt.colorbar(fig)
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('Time (ms)')
    plt.ylim((t0Vec[0], t0Vec[-1]))
    plt.xlim((vVec[0], vVec[-1]))
    plt.gca().invert_yaxis()
    if title is not None:
        plt.title(title)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    plt.clf()
    plt.close()
    

# plot original spectrum
def OriSpec(spectrum, save_path='xxx'):
    origin_pwr = 255 - (spectrum - np.min(spectrum)) / (np.max(spectrum) - np.min(spectrum)) * 255
    data_plot = origin_pwr.astype(np.uint8).squeeze()
    data_plot_hot = cv2.applyColorMap(data_plot, cv2.COLORMAP_JET)  # COLORMAP_JET COLORMAP_HOT
    plt.figure(figsize=(2, 10), dpi=300)
    plt.imshow(data_plot_hot, aspect='auto')
    plt.axis('off')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# plot the energy peaks
def plot_peaks(peaks, label, if_show=1, if_legend=1, col_index=0,
               marker='x', if_save=0, save_path='xxx'):
    col_list = ['blue', 'black']
    plt.scatter(
        x=peaks[:, 1],
        y=peaks[:, 0],
        c=col_list[col_index],
        marker=marker,
        label=label,
        edgecolors='white'
    )
    if if_legend:
        plt.legend()
    if if_show:
        plt.show()
    if if_save:
        plt.savefig(save_path)
    plt.close()


# plot the line curve
def plot_curve(curve, c, label='interpolate curve'):
    plt.plot(curve[:, 1], curve[:, 0],
             c=c,
             label=label, linewidth=2)


# plot the velocity curve of auto picking and manual picking
def plot_vel_curve(auto_curve, label_curve, t0_ind, v_ind, auto_peaks=None, save_path='xxx'):
    plt.figure(figsize=(2, 10), dpi=300)
    ax = plt.gca()
    plt.plot(auto_curve[:, 1], auto_curve[:, 0], c='#ef233c',
             label='Auto Curve', linewidth=1)
    plt.plot(label_curve[:, 1], label_curve[:, 0], c='#8d99ae',
             label='Manual Curve', linewidth=1)
    if auto_peaks is not None:
        plt.scatter(auto_peaks[:, 1], auto_peaks[:, 0], c='blue',
                    s=2, label='Auto Peaks')
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('Time (ms)')
    plt.ylim((t0_ind[0], t0_ind[-1]))
    plt.xlim((v_ind[0], v_ind[-1]))
    ax.invert_yaxis()
    plt.legend(loc=0)

    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close('all')


def plot_feature_map(feature_map_tensor, save_path='xxx'):
    Para = {
        'ws': [5, 5, 5, 10, 10, 10, 15, 15, 15],
        'st': [1, 2, 3, 1, 2, 3, 1, 2, 3],
        'eec': [1, 1.5, 2, 1, 1.5, 2, 1, 1.5, 2],
        'ln': [5, 8, 12, 5, 8, 12, 5, 8, 12]
    }

    def trim_axs(axs, N):
        axs = axs.flat
        for ax in axs[N:]:
            ax.remove()
        return axs[:N]

    cases = ['ws=%d st=%d eec=%d ln=%d' % 
            (Para['ws'][i], Para['st'][i], Para['eec'][i], Para['ln'][i]) for i in range(9)]
    axs = plt.figure(figsize=(15, 15), constrained_layout=True).subplots(3, 3)
    axs = trim_axs(axs, len(cases))
    count = 0
    spectrum_array = feature_map_tensor[1:]
    font1 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 23,
    }
    for ax, case in zip(axs, cases):
        spectrum = spectrum_array[count]
        origin_pwr = 255 - (spectrum - np.min(spectrum)) / (np.max(spectrum) - np.min(spectrum)) * 255
        data_plot = origin_pwr.astype(np.uint8)
        data_plot_hot = cv2.applyColorMap(data_plot, cv2.COLORMAP_JET)
        ax.imshow(data_plot_hot, aspect='auto')
        ax.set_title(case, font1)
        count += 1

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close('all')


# resize the spectrum
def resize_spectrum(spectrum, resize):
    spectrum = transforms.ToPILImage()(spectrum)
    resize_spec = np.array(transforms.Resize(size=resize)(spectrum))
    return resize_spec


# plot stk velocity curve
def plot_stk_vel(vel_array, save_path):
    plt.figure(figsize=(2, 10), dpi=300)
    ax = plt.gca()
    for k in range(vel_array.shape[0]):
        VC = vel_array[k, vel_array[k, :, 0]> 0, :]
        plt.plot(VC[:, 0], VC[:, 1], label=str(k), linewidth=1)
        plt.scatter(VC[:, 0], VC[:, 1], c='blue', s=3)
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('Time (ms)')
    ax.invert_yaxis()
    plt.legend(loc=1)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close('all')


# plot the data sample distributions
def PlotSampleDistributions(SampleDict, save_path='xxx'):
    plt.figure(figsize=(8, 4), dpi=300)
    CList = ['red', 'blue', 'm']
    SSize = [8, 8, 8]
    for i, (part, Sample) in enumerate(SampleDict.items()):
        SampleS = [elm.split('_') for elm in Sample]
        SampleA = np.array(SampleS, dtype=np.int).reshape((-1, 2))
        plt.scatter(SampleA[:, 1], SampleA[:, 0], s=SSize[i], c=CList[i], label=part)
    plt.ylabel('Line')
    plt.xlabel('CDP')
    plt.legend()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close('all')


# plot the reference velocity distributions
def RefVelDistributions(AllIndex, SPIndex, save_path=None):
    USPIndex = set(AllIndex) - set(SPIndex)
    plt.figure(figsize=(8, 4), dpi=100)
    USPScatter = np.array([elm.split('_') for elm in list(USPIndex)], dtype=np.int32).reshape((-1, 2))
    SPScatter = np.array([elm.split('_') for elm in list(SPIndex)], dtype=np.int32).reshape((-1, 2))
    plt.scatter(USPScatter[:, 1], USPScatter[:, 0], s=1, c='black')
    plt.scatter(SPScatter[:, 1], SPScatter[:, 0], s=2, c='red', label='Reference\nVelocity')
    plt.ylabel('Line')
    plt.xlabel('CDP')
    plt.legend()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close('all')


# Plot Velocity Field
def PlotVelField(VelField, cdpList, tInd, vInd, LineName, SavePath):
    plt.figure(figsize=(20, 10), dpi=300)
    tshow = [''] * len(tInd)
    tIndex = np.linspace(0, len(tInd)-1, num=20).astype(np.int)
    for i in tIndex:
        tshow[i] = tInd[i]
    # heatmap
    h = sns.heatmap(data=VelField, cmap='jet', linewidths=0, annot=False, cbar=False,
                    vmax=vInd[-1], vmin=vInd[0], xticklabels=cdpList, yticklabels=tshow,
                    cbar_kws={'label': 'Velocity (m/s)'})

    # color bar
    cb = h.figure.colorbar(h.collections[0])
    cb.ax.tick_params(labelsize=15)

    plt.title('Line %s Velocity Field' % LineName, fontsize=25)
    plt.xlabel('CDP', fontsize=20)
    plt.ylabel('t0', fontsize=20)
    if SavePath is None:
        plt.show()
    else:
        plt.savefig(SavePath, dpi=150, bbox_inches='tight')
    plt.clf()
    plt.close()


# PastProcessing Visual Work
def PPVisual(SegMat, YEnergy, PeakInd):
    plt.figure(figsize=(5, 10), dpi=100)
    ax1 = plt.subplot2grid((10, 5), (0, 0), colspan=4, rowspan=10)
    ax2 = plt.subplot2grid((10, 5), (0, 4), colspan=1, rowspan=10)

    # plot SegMat
    origin_pwr = (SegMat - np.min(SegMat)) / (np.max(SegMat) - np.min(SegMat)) * 255
    data_plot = origin_pwr.astype(np.uint8).squeeze()
    data_plot_hot = cv2.applyColorMap(data_plot, cv2.COLORMAP_HOT)  # COLORMAP_JET COLORMAP_HOT
    ax1.imshow(data_plot_hot, aspect='auto')
    ax1.set_xlabel('Velocity')
    ax1.set_ylabel('Time')
    ax1.set_xticks([])
    ax1.set_yticks([])
    # plot Energy Stack 
    tInd = np.arange(SegMat.shape[0])
    ax2.plot(YEnergy, tInd, c='red', linewidth=0.5)
    ax2.scatter(YEnergy[PeakInd], PeakInd, c='blue', s=5)
    # ax2.set_xticks([])
    ax2.set_ylim(min(tInd), max(tInd))
    ax2.set_yticks([])
    ax2.invert_yaxis()

    # save the fig
    if not os.path.exists('result/process'):
        os.mkdir('result/process')
    plt.savefig('result/process/PPVisual.png', dpi=100, bbox_inches='tight')
    print('Save to result/process/PPVisual.png')
    plt.close('all')


# hist for energy amp
def EnergyHist(energy):
    plt.figure(figsize=(5, 5), dpi=100)
    plt.hist(energy)
    if not os.path.exists('result/process'):
        os.mkdir('result/process')
    plt.savefig('result/process/EnergyHist.png', dpi=100, bbox_inches='tight')
    plt.close('all')


# visual enhanced process
def EnhancedProcess(ProcessDict, SavePath):
    for name, spectrum in ProcessDict.items():
        plt.figure(figsize=(2, 10), dpi=300)
        origin_pwr = 255 - (spectrum - np.min(spectrum)) / (np.max(spectrum) - np.min(spectrum)) * 255
        data_plot = origin_pwr.astype(np.uint8).squeeze()
        data_plot_hot = cv2.applyColorMap(data_plot, cv2.COLORMAP_JET)  # COLORMAP_JET COLORMAP_HOT
        plt.imshow(data_plot_hot, aspect='auto')
        plt.axis('off')
        plt.savefig(SavePath.replace('.png', '_S%d.png' % int(name)), dpi=300, bbox_inches='tight')
        plt.close('all')


# point set of spectrum
def PointSet(points, t0Vec, vVec, SavePath=None, Label=None, NearPick=None):
    plt.figure(figsize=(2, 10), dpi=300)
    plt.scatter(x=points[:, 1], y=points[:, 0], c=points[:, 2], s=0.1)
    plt.ylim((t0Vec[0], t0Vec[-1]))
    plt.xlim((vVec[0], vVec[-1]))
    if Label is not None:
        plt.plot(Label[:, 1], Label[:, 0], label='Manual Picking', linewidth=2, c='red')
    if Label is not None:
        plt.plot(NearPick[:, 1], NearPick[:, 0], label='Near Reference Velocity', linewidth=2, c='blue')

    plt.xlabel('Velocity (m/s)')
    plt.ylabel('Time (ms)')
    plt.gca().invert_yaxis()
    if Label is not None or NearPick is not None:
        plt.legend()
    if SavePath is None:
        plt.show()
    else:
        plt.savefig(SavePath, dpi=150, bbox_inches='tight')

# point set of spectrum
def CluPlot(points, centers, t0Vec, vVec, SavePath=None):
    plt.figure(figsize=(2, 10), dpi=300)
    plt.scatter(x=points[:, 1], y=points[:, 0], c=points[:, 2], s=0.1)
    plt.scatter(x=centers[:, 1], y=centers[:, 0], c='r', marker='x', label='Cluster Centers')
    plt.ylim((t0Vec[0], t0Vec[-1]))
    plt.xlim((vVec[0], vVec[-1]))
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('Time (ms)')
    plt.gca().invert_yaxis()
    plt.legend()
    if SavePath is None:
        plt.show()
    else:
        plt.savefig(SavePath, dpi=150, bbox_inches='tight')
    plt.close('all')


# Iteration processing
def IterPlot(IterList, points, t0Vec, vVec, SavePath=None):
    fig = plt.figure(figsize=(4, 10), dpi=300)
    ax = fig.add_subplot(autoscale_on=False, xlim=(vVec[0], vVec[-1]), ylim=(t0Vec[0], t0Vec[-1]))
    plt.scatter(points[:, 1], points[:, 0], c=points[:, 2], s=1)
    center, = ax.plot([], [], 'x', c='r', markersize=5)
    ax.invert_yaxis()
    ax.set_xlabel('Velocity (m/s)')
    ax.set_ylabel('Time (ms)')

    def FramPlot(i=int):
        sig, centers = IterList[i]
        center.set_data(centers[:, 1], centers[:, 0])
        ax.set_title(label='Iteration Time=%d Sigma=%.2f\nCenter Number=%d' % (i, sig, centers.shape[0]))
        return centers, ax

    animator = ani.FuncAnimation(fig, FramPlot, len(IterList), interval = 1000)
    if SavePath is None:
        plt.show()
    else:
        animator.save(SavePath)
        # plt.savefig(SavePath, dpi=150, bbox_inches='tight')
    plt.clf()
    plt.close()


# visual ensemble result
def PlotEnsemble(CluCenters, FinalCenters, NearRef, SeedRef, bw, AP, MP, t0Vec, vVec, SavePath=None):
    _, ax = plt.subplots(figsize=(2, 10), dpi=90)
    # Near ref vel
    ax.plot(NearRef[:, 1], NearRef[:, 0], c='#2d6a4f', linewidth=1, linestyle="-.", label='NearRefer')
    # Seed ref vel
    ax.plot(SeedRef[:, 1], SeedRef[:, 0], c='#370617', linewidth=1, linestyle="--", label='SeedRefer')
    # Range
    ax.fill_betweenx(NearRef[:, 0], NearRef[:, 1] - bw, NearRef[:, 1] + bw, facecolor='#4cc9f0', interpolate=False, alpha=0.2)
    ax.fill_betweenx(SeedRef[:, 0], SeedRef[:, 1] - bw, SeedRef[:, 1] + bw, facecolor='#eae2b7', interpolate=False, alpha=0.2)
    
    # Cluster Centers
    ax.scatter(x=CluCenters[:, 1], y=CluCenters[:, 0], s=5, c='blue', label='Cluster Centers')
    # Final Picking
    ax.scatter(x=FinalCenters[:, 1], y=FinalCenters[:, 0], linewidth=0.3, s=8, c='red', marker='X', label='Ensemble Picking')
    
    # Ensemble vel curve
    ax.plot(AP[:, 1], AP[:, 0], c='#ef233c', linewidth=1, label='Ensemble Velocity')
    # Manual vel curve
    ax.plot(MP[:, 1], MP[:, 0], c='#22223b', linewidth=1, label='Manual Velocity')
    # other setting
    ax.set_xlim(min(vVec), max(vVec))
    ax.set_ylim(min(t0Vec), max(t0Vec))
    ax.set_xlabel('Velocity (m/s)')
    ax.set_ylabel('Time (ms)')
    ax.invert_yaxis()
    plt.legend(loc=1, fontsize=6)
    if SavePath is None:
        plt.show()
    else:
        plt.savefig(SavePath, dpi=150, bbox_inches='tight')
    plt.close('all')


# Plot single spectrum
def PlotSingleSpec(spectrum, save_path=None):
    origin_pwr = 255 - (spectrum - np.min(spectrum)) / (np.max(spectrum) - np.min(spectrum)) * 255
    data_plot = origin_pwr.astype(np.uint8).squeeze()
    data_plot_hot = cv2.applyColorMap(data_plot, cv2.COLORMAP_JET)  # COLORMAP_JET COLORMAP_HOT
    plt.figure(figsize=(2, 10), dpi=300)
    plt.imshow(data_plot_hot, aspect='auto')
    plt.axis('off')
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.clf()
    plt.close()



def PlotNearPWR(NearPWR, IndexList, SavePath='xxx'):

    def trim_axs(axs, N):
        axs = axs.flat
        for ax in axs[N:]:
            ax.remove()
        return axs[:N]

    cases = ['Line %s CDP %s' % tuple(elm.split('-')) for elm in IndexList]
    axs = plt.figure(figsize=(5, 8), constrained_layout=True).subplots(2, 4)
    axs = trim_axs(axs, len(cases))
    count = 0
    spectrum_array = NearPWR
    font1 = {'weight': 'normal','size': 7}
    for ax, case in zip(axs, cases):
        spectrum = spectrum_array[count]
        data_plot = spectrum
        im = ax.imshow(data_plot, cmap='seismic', aspect='auto')
        ax.set_title(case, font1)
        ax.axis('off')
        count += 1
    plt.colorbar(im, ax=axs.ravel().tolist(), fraction=0.05, pad=0.04)
    plt.savefig(SavePath, dpi=200, bbox_inches='tight')
    plt.close('all')


# Refer Velocity 
def RefVelPlot(points, values, ref, bw, lab, t0Vec, vVec, save_path=None):
    plt.figure(figsize=(2, 10), dpi=150)
    col = plt.cm.jet(values)
    fig = plt.scatter(x=points[:, 1], y=points[:, 0], c=col, s=0.3)
    plt.colorbar(fig, fraction=0.05, pad=0.04)
    plt.plot(ref[:, 1], ref[:, 0], '#e36414', label='Near Velocity Prior')
    plt.plot(ref[:, 1]-bw, ref[:, 0], '--', c='black')
    plt.plot(ref[:, 1]+bw, ref[:, 0], '--', c='black')
    plt.scatter(x=lab[:, 1], y=lab[:, 0], c='#9a031e', marker='*', label='True RMS Velocity')
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('Time (ms)')
    plt.ylim((t0Vec[0], t0Vec[-1]))
    plt.xlim((vVec[0], vVec[-1]))
    # plt.yticks([])
    # plt.axis('off')
    plt.gca().invert_yaxis()
    plt.legend(loc=1, fontsize=7.3)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    
    plt.clf()
    plt.close()


# EP1 visual
def HistPlot(ResultDict, t0Vec, SavePath=None):
    fig = plt.figure(figsize=(5, 5))
    ax1 = fig.add_subplot(111)
    plt.xlabel('Time (ms)')         
    ax1.set_ylabel('Mean Deviation')     
    ax1.set_xlim((t0Vec[0], t0Vec[-1]))             
    
    for name in list(ResultDict.keys()):
        data = np.array(ResultDict[name])
        if name[-1] == '0':
            continue
        else:
            name = name.strip('-G1')
        if name[0] == 'A':
            ax1.plot(data[:, 0], data[:, 1], '*-', alpha=0.4, linewidth=3, label='MD-%s' % name)
        else:
            ax1.plot(data[:, 0], data[:, 1], '*-', alpha=0.4, label='MD-%s' % name)    
    plt.legend(loc=5, fontsize=7, bbox_to_anchor=(1.01, 0.38)) 
    ax2 = ax1.twinx()
    ax2.set_ylabel('Picking Rate')    
    for name in list(ResultDict.keys()):
        data = np.array(ResultDict[name])
        if name[-1] == '0':
            continue
        else:
            name = name.strip('-G1')
        if name[0] == 'A':
            ax2.plot(data[:, 0], data[:, 2], '^--', alpha=0.4, linewidth=3, label='PR-%s' % name)  
        else:
            ax2.plot(data[:, 0], data[:, 2], '^--',alpha=0.4, label='PR-%s' % name)  
    plt.legend(loc=0, fontsize=7)    
    plt.savefig(SavePath, dpi=400, bbox_inches='tight')     



def SingleClu(spec, PickDict, SavePath='xxx'):

    def trim_axs(axs, N):
        axs = axs.flat
        for ax in axs[N:]:
            ax.remove()
        return axs[:N]

    cases = list(PickDict.keys())
    axs = plt.figure(figsize=(6, 6), constrained_layout=True).subplots(2, 3)
    axs = trim_axs(axs, len(cases))
    font1 = {'weight': 'normal','size': 7}
    origin_pwr = 255 - (spec - np.min(spec)) / (np.max(spec) - np.min(spec)) * 255
    data_plot = origin_pwr.astype(np.uint8)
    data_plot_hot = cv2.applyColorMap(data_plot, cv2.COLORMAP_JET)
    count = 0
    for ax, case in zip(axs, cases):
        pick = PickDict[case]
        ax.imshow(data_plot_hot, aspect='auto')
        ax.scatter(pick[:, 1], pick[:, 0], marker='X', s=10, c='red', linewidth=0.2)
        ax.set_title(case, font1)
        ax.axis('off')
        count += 1
    plt.savefig(SavePath, dpi=150, bbox_inches='tight')
    plt.close('all')



def PwrFinal(spectrum, t0_vec, v_vec, VelCurve=None, save_path=None):
    if len(t0_vec) != spectrum.shape[0]:
        t0_vec = np.linspace(t0_vec[0], t0_vec[-1], spectrum.shape[0])
    if len(v_vec) != spectrum.shape[1]:
        v_vec = np.linspace(v_vec[0], v_vec[-1], spectrum.shape[1])
    origin_pwr = 255 - (spectrum - np.min(spectrum)) / (np.max(spectrum) - np.min(spectrum)) * 255
    data_plot = origin_pwr.astype(np.uint8).squeeze()
    data_plot_hot = cv2.applyColorMap(data_plot, cv2.COLORMAP_JET)  # COLORMAP_JET COLORMAP_HOT
    plt.figure(figsize=(2, 10), dpi=300)
    label = ['AP', 'MP']
    col = ['r', 'darkorange']
    for ind, VelC in enumerate(VelCurve):
        VCCP = copy.deepcopy(VelC)
        VCCP[:, 0] = (VCCP[:, 0]-t0_vec[0]) / (t0_vec[1]-t0_vec[0])
        VCCP[:, 1] = (VCCP[:, 1]-v_vec[0]) / (v_vec[1]-v_vec[0])
        plot_curve(VCCP, col[ind], label[ind])   
    
    plt.legend(loc=1, fontsize=13)   
    plt.imshow(data_plot_hot, aspect='auto')
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('Time (ms)')
    set_axis(v_vec, t0_vec)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.clf()
    plt.close()