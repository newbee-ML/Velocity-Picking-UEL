"""
Pre-processing for Velocity Spectra

Author: Hongtao Wang 
---
Tools:
- A. Local Scale on time domain
- B. Transform to low resolution
"""

import cv2
import copy
import numpy as np
from scipy import interpolate


################################################################
# Tool A: Local Scale on time domain by a window
################################################################
def LocalScale(spectrum, width):
    SpecCp = copy.deepcopy(spectrum)
    MeanX = np.max(spectrum, axis=1)
    for i in range(spectrum.shape[0]):
        low = i - width if i > width else 0
        up = i + width if i + width < spectrum.shape[0] else spectrum.shape[0] 
        ScaleP = np.max(MeanX[low: up]) if up + 1 <= spectrum.shape[0] else np.max(MeanX[low:])
        ScaleP = ScaleP if ScaleP > 0 else 1
        SpecCp[i, :] = SpecCp[i, :]/ScaleP

    return SpecCp


################################################################
# Tool B: Trans img to point set
################################################################
# Transfrom scale
def ChangeScale(data, OriScale, NewScale):
    datacp = copy.deepcopy(data)
    MedData = (datacp - OriScale[0]) / (OriScale[1] - OriScale[0])
    NewData = NewScale[0] + MedData * (NewScale[1] - NewScale[0])
    return NewData


# Main function for trans
def Img2Point(img, t0Vec, vVec, DownSample=False, threshold=0.01):
    if DownSample:
        # down sample
        Dimg = cv2.pyrDown(img)
    else:
        Dimg = img
    # copy and scale
    imgcp = copy.deepcopy(Dimg)
    imgcp = (imgcp - np.min(imgcp)) / np.ptp(imgcp)
    # get useful points
    Points = np.where(imgcp > threshold)
    Points = np.array([Points[0], Points[1]]).T
    Values = imgcp[Points[:, 0], Points[:, 1]].reshape(-1, 1)
    # Transform to a new scale
    Newt0Vec = np.linspace(t0Vec[0], t0Vec[-1], Dimg.shape[0])
    NewvVec = np.linspace(vVec[0], vVec[-1], Dimg.shape[1])
    Points[:, 0] = ChangeScale(Points[:, 0], np.arange(Dimg.shape[0]), Newt0Vec)
    Points[:, 1] = ChangeScale(Points[:, 1], np.arange(Dimg.shape[1]), NewvVec)
    Points = np.hstack((Points, Values))
    return Points


################################################################
# Tool C: Trans img to point set
################################################################
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
