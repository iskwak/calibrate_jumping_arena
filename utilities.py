import numpy as np
import cv2
from typing import Tuple, Type
import os
from matplotlib import pyplot as plt
import matplotlib
# import matplotlib
# from matplotlib import pyplot as plt
matplotlib.use("Agg")


def plotSampled(cap, outname, checkerboards, squares_xy, offset):
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    _, frame = cap.read()

    # adjust the corners
    frame = frame.copy()
    frame = frame[:, offset:offset+512]
    # need to flip the corners
    points = checkerboards.copy()
    meansEdges = cornerDistancesStats(points, squares_xy)
    maxMeanEdge = meansEdges.max()
    minMeanEdge = meansEdges.min()

    points = points.squeeze()
    
    colormap = matplotlib.colormaps["viridis"]
    colors = colormap(np.linspace(0, 1, 100))
    #np.random.shuffle(colors)

    j=0.25
    k=3
    plt.imshow(frame)
    for i in range(points.shape[0]):
        colorIdx = int((meansEdges[i]-minMeanEdge)/maxMeanEdge * 99)  #int(np.floor(float((meansEdges[i]-minMeanEdge))/maxMeanEdge))
        plt.scatter(points[i, :, 0], points[i, :, 1], k, color=colors[colorIdx, :], marker='.', linewidths=1, alpha=j)
        #plt.scatter(points[i, :, 0], points[i, :, 1], k , color='r', marker='.', linewidths=1, alpha=j)
    outname = '/workspace/calibration/20230830_calibrationvideos/test_colors_{:.2f}_{:.2f}.jpg'.format(k,j)
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
        plt.scatter(points[0, i, :, 0]+offsets[0], points[0, i, :, 1], 5, color=colors[0, :], cmap='cool', marker='.', linewidths=1, alpha=0.25)
        plt.scatter(points[1, i, :, 0]+offsets[1], points[1, i, :, 1], 5, color=colors[0, :], cmap='cool', marker='.', linewidths=1, alpha=0.25)
    plt.savefig(outname)
    plt.close()


def cornerDistancesStats(checkerboards, squares_xy):

    checkerboards = checkerboards.copy().squeeze()
    numFrames, numCorners, dims = checkerboards.shape

    # corner distances stats
    #maxEdgesPerBoard = []
    #minEdgesPerBoard = []
    meanEdgesPerBoard = np.zeros(numFrames)
    for i in range(numFrames):
        distances = getCornerDistances(checkerboards[i], squares_xy)
        meanEdgesPerBoard[i] = distances.mean()
    return meanEdgesPerBoard


def getCornerDistances(checkerCorners, squares_xy):
    numCorners = checkerCorners.shape[0]
    distances = np.zeros(numCorners * 2 - squares_xy[0] - squares_xy[1])
    idx = 0
    for i in range(numCorners-1):
        # next row should be i + 1
        # next col should be i + #cols
        if i % squares_xy[0] != 3:
            diff = checkerCorners[i,:]-checkerCorners[i+1,:]
            distances[idx] = np.sqrt((diff*diff).sum())
            idx+=1
        if np.floor(i / squares_xy[0]) != squares_xy[1]-1:
            diff = checkerCorners[i,:]-checkerCorners[i+squares_xy[0],:]
            distances[idx] = np.sqrt((diff*diff).sum())
            idx+=1

    return distances
