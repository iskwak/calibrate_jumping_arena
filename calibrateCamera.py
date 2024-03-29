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
from tqdm import tqdm

def plotReprojection(cap, frameNum, offset, outname, points, reprojected):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frameNum)
    _, frame = cap.read()

    mins = points.min(axis=0).squeeze().astype('int')
    maxs = points.max(axis=0).squeeze().astype('int')

    mins = mins - 60
    if mins[0] < 0:
        mins[0] = 0
    if mins[1] < 0:
        mins[1] = 0
    maxs = mins + 180

    # adjust the corners
    frame = frame.copy()
    frame = frame[mins[1]:maxs[1], mins[0]+offset:maxs[0]+offset]
    # need to flip the corners
    points = points.copy()
    reprojected = reprojected.copy()

    points[:, 0, 0] = points[:, 0, 0] - mins[0]
    points[:, 0, 1] = points[:, 0, 1] - mins[1]
    reprojected[:, 0, 0] = reprojected[:, 0, 0] - mins[0]
    reprojected[:, 0, 1] = reprojected[:, 0, 1] - mins[1]

    points = points.squeeze()
    reprojected = reprojected.squeeze()
    color_id = np.arange(points.shape[0])

    plt.imshow(frame) 
    plt.scatter(points[:, 0], points[:, 1], 12, c=color_id, cmap='cool', marker='+', facecolors='none', linewidths=1)
    plt.scatter(reprojected[:, 0], reprojected[:, 1], 12, c=color_id, cmap='plasma', marker='x', linewidths=1)
    plt.savefig(outname)
    #plt.show()
    plt.close()


def sampleCorners(rng, corners, numClusters=100, seed=123):
    centroids, _ = kmeans(corners, numClusters, seed=seed)
    clx, _ = vq(corners, centroids)

    # if the number of centroids is less than the requested, return less samples.
    sampledIdx = np.zeros((centroids.shape[0],))
    for i in range(centroids.shape[0]):
        clusteredIdx = np.where(clx == i)

        # np.where returns a tuple for me, i don't understand, because it looks like the
        # documentation says it outputs an array
        clusteredIdx = clusteredIdx[0]
        rng.shuffle(clusteredIdx)

        sampledIdx[i] = clusteredIdx[0]

    return sampledIdx.astype('int')


def main(params):
    with open(os.path.join(params["base_dir"], params["detected_corners"]), "rb") as fid:
        cornerData = pickle.load(fid)
        detectedCorners = MultiCamCheckerboardCorners.fromDict(cornerData)

    cameraIds = params["views"]
    numViews = params["num_views"]
    squares_xy = params["squares_xy"]

    rng = np.random.RandomState(123)
    seed = 123
    numClusters = 100

    cap = None
    if params["debug_image"] and params["calib_video"] and params["out_video_dir"]:
        cap = cv2.VideoCapture(os.path.join(params["base_dir"], params["calib_video"]))
        fullWidth  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(fullWidth / numViews)
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cameraIds = params["views"]
    offsets = [i*detectedCorners.frameSize[0] for i in range(numViews)]

    utilities.setCalibrationFileNames(params)

    for i in range(len(cameraIds)):
        print(f'Intrinsic calibration for view {cameraIds[i]} ({i+1}/{len(cameraIds)})')
        objpoints = detectedCorners.setupObjPoints()

        corners = detectedCorners.corners[:,detectedCorners.cornerCameraFlag[cameraIds[i], :], :, :, :]
        corners = corners[cameraIds[i], :, :, :]
        frameNumbers = detectedCorners.frameNumbers[detectedCorners.cornerCameraFlag[cameraIds[i], :]]

        sampledIdx = sampleCorners(rng, corners.squeeze()[:, 0, :], numClusters, seed)
        sampled = corners[sampledIdx, :, :]
        sampledFrames = np.asarray(frameNumbers)[sampledIdx]
        if numClusters != sampledIdx.shape[0]:
            print("Warning: unable to sample {} checkerboard frames, only sampled {} checkerboard frames.".format(numClusters, sampledIdx.shape[0]))
        numClusters = sampledIdx.shape[0]

        _, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints[:numClusters], sampled, (512, 512), None, None)
        calibrationData = {
            "mtx": mtx,
            "dist": dist,
            "rvecs": rvecs,
            "tvecs": tvecs,
            "sampledIdx": sampledIdx,
            "cornerFile": os.path.join(params["base_dir"], params["detected_corners"])
        }

        fullOutDir = os.path.split(os.path.join(params["base_dir"], params["calibration"][i]))[0]
        if not os.path.isdir(fullOutDir):
            os.makedirs(fullOutDir)
        fullOutVideoDir = os.path.join(params["base_dir"],params["out_video_dir"])
        if not os.path.isdir(fullOutVideoDir):
            os.makedirs(fullOutVideoDir)
        print(f'Outputting calibration to {params["calibration"][i]}...')
        with open(os.path.join(params["base_dir"], params["calibration"][i]), "wb") as fid:
            pickle.dump(calibrationData, fid)

        meanPixelError = 0
        worstError = 0
        worstErrorIdx = 0
        allReprojections = []
        print(f'Computing, plotting reprojection error')
        for j in tqdm(range(numClusters)):
            imgpoints2, _ = cv2.projectPoints(objpoints[j], rvecs[j], tvecs[j], mtx, dist)
            allReprojections.append(imgpoints2)
            errorPixel = np.sqrt(np.sum(np.square(sampled[j].squeeze() - imgpoints2.squeeze()), axis=1))

            meanPixelError += errorPixel.sum()
            if max(errorPixel) > worstError:
                worstError = max(errorPixel)
                worstErrorIdx = j
            if cap:
                # write the frames to disk
                outname = os.path.join(
                    params["base_dir"],
                    params["out_video_dir"],
                    "cam_{}_error_{}_reproj_{}.png".format(cameraIds[i], np.mean(errorPixel), sampledFrames[j]))
                plotReprojection(cap, sampledFrames[j], offsets[cameraIds[i]], outname, sampled[j], imgpoints2)
        if cap is not None:
            outname = os.path.join(
                params["base_dir"],
                params["out_video_dir"],
                "cam_{}_sampled.svg".format(cameraIds[i]))

            utilities.plotSampled(cap, outname, sampled, squares_xy, offsets[cameraIds[i]])
        print("total pixel mean error: {}".format(meanPixelError/(squares_xy[0] * squares_xy[1] * numClusters)))


if __name__ == "__main__":
    params = calibflags.parseArgs(sys.argv[1:],paramtype="intrinsic")
    main(params)
