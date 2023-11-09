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
matplotlib.use("TkAgg")
import utilities


def plotDetectionCorners(detectedCorners):

    cap, width, fullWidth, height, fourcc, fps = utilities.loadVideo(detectedCorners.videoName, detectedCorners.numViews)
    cameraIds =detectedCorners.cameraIds
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    fullOutDir = os.path.join(params["base_dir"], params["output_dir"])
    outname = os.path.join(fullOutDir, "detection_movie.avi")
    if not os.path.isdir(fullOutDir):
        os.makedirs(fullOutDir)
    writer = cv2.VideoWriter(outname, fourcc, fps, (fullWidth, height))

    # create a plot with all the detections on one image.
    
 
    frameNumbers = detectedCorners.frameNumbers.copy()
    frameNum = 0
    frameIdx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            if frameIdx < len(frameNumbers) and frameNum == frameNumbers[frameIdx]:
                for i in range(len(cameraIds)):
                    if detectedCorners.cornerCameraFlag[i, frameIdx] == True:
                        offset = 512*i
                        tempCorners = detectedCorners.corners[i, frameIdx, :, :]
                        tempCorners[:, :, 0] += offset
                        cv2.drawChessboardCorners(frame, detectedCorners.squares_xy, detectedCorners.corners[i, frameIdx, :, :], ret)
                frameIdx += 1

            writer.write(frame)
            frameNum += 1
        else:
            break

    writer.release()

    return


if __name__ == "__main__":
    params = calibflags.parseArgs(sys.argv[1:])
    with open(os.path.join(params["base_dir"], params["detected_corners"]), "rb") as fid:
        cornerData = pickle.load(fid)
        detectedCorners = MultiCamCheckerboardCorners.fromDict(cornerData)

    plotDetectionCorners(detectedCorners)
