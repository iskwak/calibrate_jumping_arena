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


def main(params):
    cameraIds = params["views"]
    cameraCalibrations = []
    for i in range(len(cameraIds)):
        with open(os.path.join(params["base_dir"], params["calibration"][i]), "rb") as fid:
            cameraCalibrations.append(pickle.load(fid))

    calib02 = scipy.io.loadmat(os.path.join(params["base_dir"], params["stereos"][0]))
    #calib12 = scipy.io.loadmat(os.path.join(params["base_dir"], params["stereos"][1]))
    calib21 = scipy.io.loadmat(os.path.join(params["base_dir"], params["stereos"][1]))
    calib12 = scipy.io.loadmat(os.path.join(params["base_dir"], params["stereos"][2]))

    # R21 = calib12["R"].T
    # T21 = -R21 * calib12["T"]

    R01 = np.matmul(calib21["R"], calib02["R"])
    T01 = np.matmul(calib21["R"], calib02["T"]) + calib21["T"]

    mtx1 = cameraCalibrations[0]["mtx"]
    dist1 = cameraCalibrations[0]["dist"]
    mtx2 = cameraCalibrations[1]["mtx"]
    dist2 = cameraCalibrations[1]["dist"]

    om = cv2.Rodrigues(R01)
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
        "R": R01,
        "T": T01,
        "F": np.zeros((3,3)),
        "active_images_left": [],
        "cc_left_error": 0,
        "cc_right_error": 0,
        "recompute_intrinsic_right": 1,
        "recompute_intrinsic_left": 1
    }
    scipy.io.savemat("{}/cam_{}{}_opencv.mat".format(
        params["base_dir"], cameraIds[0], cameraIds[1]), out_dict)


if __name__ == "__main__":
    params = calibflags.parseArgs(sys.argv[1:])
    main(params)
