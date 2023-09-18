import numpy as np
import cv2
from typing import Tuple, Type
import os
from matplotlib import pyplot as plt
import matplotlib


def plotSampled(cap, outname, sampledFrames, offset):
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    _, frame = cap.read()

    # adjust the corners
    frame = frame.copy()
    frame = frame[:, offset:offset+512]
    # need to flip the corners
    points = sampledFrames.copy()

    points = points.squeeze()
    #color_id = np.arange(points.shape[0])
    
    colormap = matplotlib.colormaps["viridis"]
    colors = colormap(np.linspace(0, 1, points.shape[0]))
    np.random.shuffle(colors)
    plt.imshow(frame)
    for i in range(points.shape[0]):
        plt.scatter(points[i, :, 0], points[i, :, 1], 1, color=colors[0, :], marker='.', linewidths=1)
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


# # def readMultiViewVideo(filename: str, numViews: int) -> Type["cv2.VideoCapture"]:
# #     cap = cv2.VideoCapture(filename)
# #     fullWidth  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# #     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# #     width = int(fullWidth / numViews)
# #     fps = cap.get(cv2.CAP_PROP_FPS)

# #     return cap, fullWidth, height, width, fps

# def getCapInfo(cap: Type["cv2.VideoCapture"], numViews: int) -> Tuple[int, int, int, float]:
#     fullWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     width = int(fullWidth / numViews)
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)

#     return fullWidth, width, height, fps
#     # fullWidth  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     # width = int(fullWidth / numViews)
#     # fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
#     # fps = cap.get(cv2.CAP_PROP_FPS)

# import numpy as np
# import cv2
# # import glob
# from matplotlib import pyplot as plt
# from absl import app
# from absl import flags
# import pickle
# from calibrationdata import CalibrationFrames
# import time
# import scipy
# import scipy.io
# from scipy.cluster.vq import kmeans,vq,whiten
# import random
# import os


# def draw_corner_numbers(image, corners, offset):
#     num_corners = corners.shape[0]
#     corners = corners.squeeze()
#     color_step = 209 / num_corners

#     for i in range(num_corners):
#         cv2.putText(image, "{}".format(i),
#             (int(corners[i, 0] + offset), int(corners[i, 1])),
#             cv2.FONT_HERSHEY_SIMPLEX, 0.25, (i * color_step, 80, 0, 255), 1)


# def draw_corners_with_gradient(image, corners, color, markerSize, offset=0):
#     num_corners = corners.shape[0]
#     color_step = color[0] / num_corners
#     for i in range(len(corners)):
#         #current_color = color
#         current_color = (color_step * i, color[1], color[2])
#         cv2.drawMarker(image, (int(corners[i, 0] + offset), int(corners[i, 1])), current_color,
#             markerType=cv2.MARKER_CROSS, markerSize=markerSize)


# def mean_std_corner_dists(corners):
#     # 0 -> 6 are on one line.
#     # then 7 -> 13 and so on. total of 5 rows.
#     # Get the distance between (i, i+1) and (i,i+7) unless the last column (6n) or last column (i>=35)
#     corners = corners.squeeze()
#     # corners should be num corners x dimensions (42x2). can't really handle the above stuff unless the dimensions
#     # of the target is provided. will need to make that a parameter at some point
#     num_corners = corners.shape[0]
    
#     # dont want to think about indexing. going to append and then convert to np array.
#     edge_dists = []
#     for i in range(num_corners - 1):
#         if i % 7 != 6:
#             # not the last col, check for next column
#             dist = np.sqrt(np.sum(np.square(corners[i] - corners[i + 1])))
#             edge_dists.append(dist)
#         if i < 35:
#             # not the last row, look for the next row
#             dist = np.sqrt(np.sum(np.square(corners[i] - corners[i + 7])))
#             edge_dists.append(dist)
#     edge_dists = np.asarray(edge_dists)
#     # import pdb; pdb.set_trace()
#     return edge_dists.mean(), edge_dists.std(), edge_dists


# def index_list(main_list, index_list):
#     temp = []
#     for i in index_list:
#         temp.append(main_list[i])
#     return temp


# def plot_write_cropped_corners(frame, outname, corners, offset=0, figure=None):
#     mins = corners.min(axis=0).squeeze().astype('int')
#     maxs = corners.max(axis=0).squeeze().astype('int')

#     mins = mins - 60
#     if mins[0] < 0:
#         mins[0] = 0
#     if mins[1] < 0:
#         mins[1] = 0
#     maxs = mins + 180

#     frame = frame.copy()
#     frame = frame[mins[1]:maxs[1], mins[0]+offset:maxs[0]+offset]

#     corners = corners.copy()
#     corners[:, 0, 0] = corners[:, 0, 0] - mins[0]
#     corners[:, 0, 1] = corners[:, 0, 1] - mins[1]

#     color_id = np.arange(corners.shape[0])
#     plt.imshow(frame)
#     plt.scatter(corners[:, 0], corners[:, 1], 12, c=color_id, cmap='cool', marker='x', linewidths=1)
#     plt.scatter(corners2[:, 0], corners2[:, 1], 12, c=color_id, cmap='plasma', marker='+', linewidths=1)
#     plt.savefig(outname)
#     #plt.show()
#     plt.close()


# # def plot_write_corners(frame, outname, corners, offset=0):
# #     mins1 = corners1.min(axis=0).squeeze().astype('int')
# #     maxs1 = corners1.max(axis=0).squeeze().astype('int')

# #     mins = mins - 60
# #     if mins[0] < 0:
# #         mins[0] = 0
# #     if mins[1] < 0:
# #         mins[1] = 0
# #     maxs = mins + 180

# #     # adjust the corners
# #     frame = frame.copy()
# #     frame = frame[mins[1]:maxs[1], mins[0]+offset:maxs[0]+offset]
# #     # need to flip the corners
# #     imgpoints = imgpoints.copy()
# #     imgpoints2 = imgpoints2.copy()

# #     imgpoints[:, 0, 0] = imgpoints[:, 0, 0] - mins[0]
# #     imgpoints[:, 0, 1] = imgpoints[:, 0, 1] - mins[1]
# #     imgpoints2[:, 0, 0] = imgpoints2[:, 0, 0] - mins[0]
# #     imgpoints2[:, 0, 1] = imgpoints2[:, 0, 1] - mins[1]
# #     # cv2.drawChessboardCorners(frame, squares_xy, imgpoints, True)
# #     # draw_corner_numbers(frame_flipped, reordered)
# #     # # mark the frame number on a flipped example
# #     # cv2.putText(frame_flipped, "{}: {}".format(i, frame_num),
# #     #     (20, 20),
# #     #     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (209, 80, 0, 255), 1)
# #     corners = imgpoints.squeeze()
# #     #draw_corners(frame, corners, (255, 0, 255), 5)
# #     corners2 = imgpoints2.squeeze()
# #     #draw_corners(frame, corners, (0, 255, 255), 5)
# #     color_id = np.arange(corners.shape[0])

# #     plt.imshow(frame)
# #     plt.scatter(corners[:, 0], corners[:, 1], 12, c=color_id, cmap='cool', marker='x', linewidths=1)
# #     plt.scatter(corners2[:, 0], corners2[:, 1], 12, c=color_id, cmap='plasma', marker='+', linewidths=1)
# #     plt.savefig(outname)
# #     #plt.show()
# #     plt.close()

# #     # cv2.imshow("frame", frame)
# #     # cv2.waitKey()
# #     # cv2.destroyAllWindows()
# #     #cv2.imwrite("reprojections/{}.png".format(frame_idx), frame)