import numpy as np
import cv2
# import glob
from matplotlib import pyplot as plt
from absl import app
from absl import flags
import pickle
from calibrationdata import CalibrationFrames
import time
import scipy
import scipy.io
from scipy.cluster.vq import kmeans,vq,whiten
import random
import os


def draw_corner_numbers(image, corners, offset):
    num_corners = corners.shape[0]
    corners = corners.squeeze()
    color_step = 209 / num_corners

    for i in range(num_corners):
        cv2.putText(image, "{}".format(i),
            (int(corners[i, 0] + offset), int(corners[i, 1])),
            cv2.FONT_HERSHEY_SIMPLEX, 0.25, (i * color_step, 80, 0, 255), 1)


def draw_corners_with_gradient(image, corners, color, markerSize, offset=0):
    num_corners = corners.shape[0]
    color_step = color[0] / num_corners
    for i in range(len(corners)):
        #current_color = color
        current_color = (color_step * i, color[1], color[2])
        cv2.drawMarker(image, (int(corners[i, 0] + offset), int(corners[i, 1])), current_color,
            markerType=cv2.MARKER_CROSS, markerSize=markerSize)


def mean_std_corner_dists(corners):
    # 0 -> 6 are on one line.
    # then 7 -> 13 and so on. total of 5 rows.
    # Get the distance between (i, i+1) and (i,i+7) unless the last column (6n) or last column (i>=35)
    corners = corners.squeeze()
    # corners should be num corners x dimensions (42x2). can't really handle the above stuff unless the dimensions
    # of the target is provided. will need to make that a parameter at some point
    num_corners = corners.shape[0]
    
    # dont want to think about indexing. going to append and then convert to np array.
    edge_dists = []
    for i in range(num_corners - 1):
        if i % 7 != 6:
            # not the last col, check for next column
            dist = np.sqrt(np.sum(np.square(corners[i] - corners[i + 1])))
            edge_dists.append(dist)
        if i < 35:
            # not the last row, look for the next row
            dist = np.sqrt(np.sum(np.square(corners[i] - corners[i + 7])))
            edge_dists.append(dist)
    edge_dists = np.asarray(edge_dists)
    # import pdb; pdb.set_trace()
    return edge_dists.mean(), edge_dists.std(), edge_dists


def index_list(main_list, index_list):
    temp = []
    for i in index_list:
        temp.append(main_list[i])
    return temp