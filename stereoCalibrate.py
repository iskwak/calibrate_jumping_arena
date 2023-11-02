import numpy as np
import cv2
# import glob
from cornerdata import MultiCamCheckerboardCorners
from matplotlib import pyplot as plt
import sys
import calibflags
import pickle
import time
import scipy
import scipy.io
# from scipy.cluster.vq import kmeans,vq,whiten
# import random
import calibrateCamera
import os
import utilities
import matplotlib
matplotlib.use("Agg")


def writeStereoReprojection(cap, baseDir, corners, reprojections, frameNum, offsets, cameraIds):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frameNum)
    _, orgframe = cap.read()
    outDir = "{}/reprojections/{}{}/".format(baseDir, cameraIds[0], cameraIds[1])
    os.makedirs(outDir, exist_ok=True)

    mins = corners[0, :, :, :].min(axis=0).squeeze().astype('int')
    maxs = corners[0, :, :, :].max(axis=0).squeeze().astype('int')
    mins = mins - 60
    if mins[0] < 0:
        mins[0] = 0
    if mins[1] < 0:
        mins[1] = 0
    maxs = mins + 180

    # adjust the corners
    frame = orgframe.copy()
    frame = frame[mins[1]:maxs[1], mins[0]+offsets[0]:maxs[0]+offsets[0]]

    reprojections = reprojections.squeeze()
    points = corners.squeeze()
    colormap = matplotlib.colormaps["viridis"]
    colors = colormap(np.linspace(0, 1, points.shape[1]))

    plt.imshow(frame)
    #for i in range(points.shape[0]):
    plt.scatter(points[0, :, 0]-mins[0], points[0, :, 1]-mins[1], 40, color=colors, marker='o', facecolors='none', linewidths=1)
    plt.scatter(reprojections[0, :, 0]-mins[0], reprojections[0, :, 1]-mins[1], 40, color=colors, marker='x', linewidths=1)

    outname = os.path.join(outDir, str(frameNum)+"_0.jpg")
    plt.savefig(outname)
    plt.close()

    mins = corners[1, :, :, :].min(axis=0).squeeze().astype('int')
    maxs = corners[1, :, :, :].max(axis=0).squeeze().astype('int')
    mins = mins - 60
    if mins[0] < 0:
        mins[0] = 0
    if mins[1] < 0:
        mins[1] = 0
    maxs = mins + 180

    # adjust the corners
    frame = orgframe.copy()
    frame = frame[mins[1]:maxs[1], mins[0]+offsets[1]:maxs[0]+offsets[1]]

    reprojections = reprojections.squeeze()
    points = corners.squeeze()
    colormap = matplotlib.colormaps["viridis"]
    colors = colormap(np.linspace(0, 1, points.shape[1]))

    plt.imshow(frame)
    #for i in range(points.shape[0]):
    plt.scatter(points[1, :, 0]-mins[0], points[1, :, 1]-mins[1], 40, color=colors, marker='o', facecolors='none', linewidths=1)
    plt.scatter(reprojections[1, :, 0]-mins[0], reprojections[1, :, 1]-mins[1], 40, color=colors, marker='x', linewidths=1)

    outname = os.path.join(outDir, str(frameNum)+"_1.jpg")
    plt.savefig(outname)
    plt.close()


def DLT(P1, P2, point1, point2):
    A = [point1[1]*P1[2,:] - P1[1,:],
         P1[0,:] - point1[0]*P1[2,:],
         point2[1]*P2[2,:] - P2[1,:],
         P2[0,:] - point2[0]*P2[2,:]
        ]
    A = np.array(A).reshape((4,4))
    #print('A: ')
    #print(A)
 
    B = A.transpose() @ A
    from scipy import linalg
    U, s, Vh = linalg.svd(B, full_matrices = False)
 
    # print('Triangulated point: ')
    # print(Vh[3,0:3]/Vh[3,3])
    return Vh[3,0:3]/Vh[3,3]


def stereoCalibrate(params, detectedCorners, cameraCalibrations):
    cap = cv2.VideoCapture(os.path.join(params["base_dir"], params["calib_video"]))
    full_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(full_width / 3)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fps = cap.get(cv2.CAP_PROP_FPS)

    numViews = params["num_views"]
    offsets = [i*detectedCorners.frameSize[0] for i in range(numViews)]
    #offsets = [0, width, 2 * width]
    ret, frame = cap.read()

    #os.makedirs("{}/paired".format(FLAGS.out_dir), exist_ok=True)
    os.makedirs("{}/stereo_reprojections".format(params["out_video_dir"]), exist_ok=True)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.0001)
    #for overlapping_frames in all_overlapping_frames:

    rng = np.random.RandomState(100)
    idx = list(range(len(detectedCorners.frameNumbers)))
    rng.shuffle(idx)
    numSamples = 200
    cameraIds = params["views"]

    mtx1 = cameraCalibrations[0]["mtx"]
    dist1 = cameraCalibrations[0]["dist"]
    mtx2 = cameraCalibrations[1]["mtx"]
    dist2 = cameraCalibrations[1]["dist"]

    objpoints = detectedCorners.setupObjPoints()
    objpoints = objpoints[:numSamples]

    # imgpoints1 = detectedCorners.corners2[cameraIds[0], idx[:numSamples], :, :].astype('float32')
    # imgpoints2 = detectedCorners.corners2[cameraIds[1], idx[:numSamples], :, :].astype('float32')
    sampledCorners = detectedCorners.corners2[:, idx[:numSamples], :, :]
    sampledFrameNumbers = np.asarray(detectedCorners.frameNumbers)[idx[:numSamples]]
    corners = detectedCorners.corners2[cameraIds, :, :, :].astype('float32')
    corners = corners[:, idx[:numSamples], :, :]

    # outname = os.path.join(
    #     params["base_dir"], params["out_video_dir"], "test_"+
    #     str(cameraIds[0])+str(cameraIds[1])+"_0.svg")
    # utilities.plotSampled(cap, outname, corners[0, :, :, :], offsets[0])
    # outname = os.path.join(
    #     params["base_dir"], params["out_video_dir"], "test_"+
    #     str(cameraIds[0])+str(cameraIds[1])+"_1.svg")
    # utilities.plotSampled(cap, outname, corners[1, :, :, :], offsets[1])


    outname = os.path.join(
        params["base_dir"], params["out_video_dir"], "stereo_corners_"+
        str(cameraIds[0])+str(cameraIds[1])+".png")
    utilities.plotPairCorners(cap, outname, corners, 512, np.asarray(offsets)[cameraIds])

    start_time = time.time()
    ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        objpoints, corners[0, :, :, :], corners[1, :, :, :], mtx1, dist1, mtx2, dist2, (512, 512), criteria=criteria,
        flags=cv2.CALIB_FIX_INTRINSIC)
    print("error: {}".format(ret))
    print("Time taken: {}".format(time.time() - start_time))


    # reproject and check errors
    RT1 = np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1)
    RT2 = np.concatenate([R, T], axis = -1)
    projMat1 = RT1
    projMat2 = RT2

    mean_error = 0
    allTriangulated = np.zeros((numSamples, corners.shape[2], 3))
    allReprojected = np.zeros((numSamples, corners.shape[2], 2, 2))
    for i in range(corners.shape[1]):
        points1u = cv2.undistortPoints(corners[0, i, :, :, :], mtx1, dist1, R=None, P=None)
        points2u = cv2.undistortPoints(corners[1, i, :, :, :], mtx2, dist2, R=None, P=None)
        #triangulated = cv2.triangulatePoints(proj_mat1, proj_mat2, imgpoints1[i], imgpoints2[i])
        triangulated = cv2.triangulatePoints(projMat1, projMat2, points1u, points2u)
        cam1RefPoints = triangulated/triangulated[3, :]
        cam1RefPoints = cam1RefPoints[:3, :].T

        imgpointsReproj, _ = cv2.projectPoints(cam1RefPoints, np.eye(3), np.zeros((3,1)), mtx1, dist1)
        #error = cv2.norm(imgpoints1[i], imgpoints_reproj, cv2.NORM_L2)/len(imgpoints_reproj)
        error = np.sqrt(np.sum(np.square(corners[0, i, :, :, :].squeeze() - imgpointsReproj.squeeze()), axis=1)).sum() / imgpointsReproj.shape[0]
        mean_error += error


        imgpointsReproj2, _ = cv2.projectPoints(cam1RefPoints, R, T, mtx2, dist2)
        error = np.sqrt(np.sum(np.square(corners[1, i, :, :, :].squeeze() - imgpointsReproj2.squeeze()), axis=1)).sum() / imgpointsReproj2.shape[0]
        mean_error += error

        allTriangulated[i, :, :] = cam1RefPoints
        allReprojected[i, :, :, 0] = imgpointsReproj.squeeze()
        allReprojected[i, :, :, 1] = imgpointsReproj2.squeeze()
        writeStereoReprojection(
            cap, params["base_dir"], corners[:, i, :, :, :],
            np.asarray([imgpointsReproj, imgpointsReproj2]),
            detectedCorners.frameNumbers[idx[i]], np.asarray(offsets)[cameraIds],
            cameraIds)


    print( "total error: {}".format(mean_error/(2*numSamples)) )
    #print( "total error: {}".format(mean_error/(len(objpoints))) )

    # write the data to a mat file.
    # need, R, T, square size, num squares, fc, cc and skew.
    # {'om' 'T' 'R' 'active_images_left' 'recompute_intrinsic_right'}
    om = cv2.Rodrigues(R)
    om = om[0]
    out_dict = {
        "calib_name_left": "cam_{}".format(cameraIds[0]),
        "calib_name_right": "cam_{}".format(cameraIds[1]),
        "cam0_id": cameraIds[0],
        "cam1_id": cameraIds[1],
        "dX": 5,
        "nx": 512,
        "ny": 512,
        "fc_left": [mtx1[0, 0], mtx1[1, 1]],
        "cc_left": [mtx1[0, 2], mtx1[1, 2]],
        "alpha_c_left": 0.0, # opencv doesnt use the skew parameter
        "kc_left": dist1,
        "fc_right": [mtx2[0, 0], mtx2[1, 1]],
        "cc_right": [mtx2[0, 2], mtx2[1, 2]],
        "alpha_c_right": 0.0, # opencv doesnt use the skew parameter
        "kc_right": dist2,
        "om": om,
        "R": R,
        "T": T,
        "F": F,
        "active_images_left": [],
        "cc_left_error": 0,
        "cc_right_error": 0,
        "recompute_intrinsic_right": 1,
        "recompute_intrinsic_left": 1
    }
    scipy.io.savemat("{}/cam_{}{}_opencv.mat".format(
        params["base_dir"], cameraIds[0], cameraIds[1]), out_dict)
    # save sampled points for testing in matlab.
    out_dict2 = {
        "sampledCorners": sampledCorners,
        "allTriangulated": allTriangulated,
        "allReprojected": allReprojected,
        "frameNumbers": sampledFrameNumbers
    }
    print("saving...")
    scipy.io.savemat("{}/sampled_{}{}.mat".format(params["base_dir"], cameraIds[0], cameraIds[1]), out_dict2)

def main(params):

    cameraIds = params["views"]
    cameraCalibrations = []
    for i in range(len(cameraIds)):
        with open(os.path.join(params["base_dir"], params["calibration"][i]), "rb") as fid:
            cameraCalibrations.append(pickle.load(fid))

    with open(os.path.join(params["base_dir"], params["detected_corners"]), "rb") as fid:
        detectedCorners = MultiCamCheckerboardCorners.fromDict(pickle.load(fid))

    # calibrate each pair of cameras
    stereoCalibrate(params, detectedCorners, cameraCalibrations)


if __name__ == "__main__":
    params = calibflags.parseArgs(sys.argv[1:])
    main(params)
