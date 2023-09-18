import numpy as np
import cv2
import pickle
from cornerdata import MultiCamCheckerboardCorners
import calibflags
#import scipy
from scipy.cluster.vq import kmeans,vq
#import random
import sys
import calibflags
import os
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use("Agg")
import utilities


def plotSampled(cap, outname, sampledFrames, offset):
    cap.set(cv2.CAP_PROP_POS_FRAMES, 47571)
    _, frame = cap.read()

    # adjust the corners
    frame = frame.copy()
    #frame = frame[:, offset:offset+512]
    # need to flip the corners
    points = sampledFrames.copy()

    points = points.squeeze()
    #color_id = np.arange(points.shape[0])
    
    colormap = matplotlib.colormaps["viridis"]
    colors = colormap(np.linspace(0, 1, points.shape[0]))
    np.random.shuffle(colors)
    plt.imshow(frame)
    for i in range(points.shape[0]):
        plt.scatter(points[i, :, 0]+offset, points[i, :, 1], 1, color=colors[0, :], marker='.', linewidths=1)
    plt.savefig(outname)
    plt.close()



def plotPairCorners(cap, outname, corners, width, offsets):
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    _, frame = cap.read()

    # adjust the corners
    # plotFrames = []
    # for i in range(len(offsets)):
    #     plotFrames.append(frame[:, offsets[i]:offsets[i]+width])

    points = corners.squeeze()
    colormap = matplotlib.colormaps["viridis"]
    colors = colormap(np.linspace(0, 1, points.shape[1]))

    plt.imshow(frame)
    for i in range(points.shape[1]):
        plt.scatter(points[0, i, :, 0]+offsets[0], points[0, i, :, 1], 12, color=colors[0, :], cmap='cool', marker='.', linewidths=1)
        plt.scatter(points[1, i, :, 0]+offsets[1], points[1, i, :, 1], 12, color=colors[0, :], cmap='cool', marker='.', linewidths=1)
    plt.savefig(outname)
    plt.close()


def main():
    baseDir = '/workspace/calibration/20230830_calibrationvideos/detections'
    names = ['detect_0.pkl', 'detect_1.pkl', 'detect_2.pkl', 'detect_02.pkl', 'detect_12.pkl']

    detectedCorners = []
    for i in range(len(names)):
        with open(os.path.join(baseDir, names[i]), "rb") as fid:
            cornerData = pickle.load(fid)
            detectedCorners.append(MultiCamCheckerboardCorners.fromDict(cornerData))

    offsets = [i*detectedCorners[0].frameSize[0] for i in range(3)]
    cap = cv2.VideoCapture('/workspace/calibration/20230830_calibrationvideos/day3_avgc52_2023_08_25_10_52_33-001.mp4')
    outdir = '/workspace/calibration/20230830_calibrationvideos/'
    outname = os.path.join(outdir, 'detected_0.png')

    utilities.plotSampled(cap, outname, detectedCorners[0].corners2[0, :, :, :], offsets[0])



if __name__ == "__main__":
    main()
