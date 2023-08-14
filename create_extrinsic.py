import os
import numpy as np
# import glob
from matplotlib import pyplot as plt
from absl import app
from absl import flags
import scipy
import scipy.io
import torch


FLAGS = flags.FLAGS
flags.DEFINE_string("calib02", "../calibration/20221011/cam_02_opencv.mat", "02 calib file")
flags.DEFINE_string("calib12", "../calibration/20221011/cam_12_opencv.mat", "12 calib file")
flags.DEFINE_string("calib01", "../calibration/20221011/cam_01_opencv.mat", "01 calib file")
flags.DEFINE_string("outputfile", "extrinsic_mouse.pth", "Combined Calibration Information.")


def main(argv):
    del argv

    calib01 = scipy.io.loadmat(FLAGS.calib01)
    calib02 = scipy.io.loadmat(FLAGS.calib02)
    calib12 = scipy.io.loadmat(FLAGS.calib12)

    # construct an output pth file with the camera parameters. Have everything from camera 0's frame of reference.
    # May want to change this later.

    # calib0 goes form camera 0, camera facing the mouse's right side, to camera 1, camera facing mouse's left side.
    # calib1 goes from camera 0, camera facing the mouse's right side, to camera 2, camera facing mouse's face.
    # calib2 goes from camera 1, camera facing the mouse's left side, to camera 2, camera facing mouse's face.
    # going to create transformations that go from camera 0 to camera 1, by going through camera 2.
    # camera 0 -> camera 2 -> camera 1
    # R02 = calib02["R"]
    # T02 = calib02["T"]
    # R21 = calib12["R"].T
    # T21 = np.dot(-R21, calib12["T"])

    # whoops, the world is better through camera 2's lenses. ie, the camera that points at the mouse.
    # converting to all be through cam2's frame of reference.

    # structure of the yaml file is a dictionary of cameras (ie, cam0, cam1, cam2, ...). Each dictionary field/camera has
    # * "T_cam_imu": the rotation+translation matrix homogeneous coordinates
    # * "intrinsics": intrinsic params, focal length and image center.
    # * "distortion_coeffs": the distortion coefficients
    calibs = {}
    R20 = calib02["R"].T
    T20 = np.dot(-R20, calib02["T"])
    RT = np.concatenate([R20, T20], axis=1)
    RT = np.concatenate([RT, np.zeros((1, RT.shape[1]))], axis=0)
    RT[-1, -1] = 1
    calibs['cam0'] = {}
    calibs['cam0']['T_cam_imu'] = RT
    calibs['cam0']['intrinsics'] = [calib02['fc_left'][0, 0], calib02['fc_left'][0, 1], calib02['cc_left'][0, 0], calib02['cc_left'][0, 1]]
    calibs['cam0']['distortion_coeffs'] = calib02['kc_left'][0, :]

    # camera 1
    R21 = calib12["R"].T
    T21 = np.dot(-R21, calib12["T"])
    RT = np.concatenate([R21, T21], axis=1)
    RT = np.concatenate([RT, np.zeros((1, RT.shape[1]))], axis=0)
    RT[-1, -1] = 1
    calibs['cam1'] = {}
    calibs['cam1']['T_cam_imu'] = RT
    calibs['cam1']['intrinsics'] = [calib01['fc_right'][0, 0], calib01['fc_right'][0, 1], calib01['cc_right'][0, 0], calib01['cc_right'][0, 1]]
    calibs['cam1']['distortion_coeffs'] = calib01['kc_right'][0, :]

    # camera 2, frame of reference
    calibs['cam2'] = {}
    calibs['cam2']['T_cam_imu'] = np.eye(4) 
    calibs['cam2']['intrinsics'] = [calib02['fc_right'][0, 0], calib02['fc_right'][0, 1], calib02['cc_right'][0, 0], calib02['cc_right'][0, 1]]
    calibs['cam2']['distortion_coeffs'] = calib02['kc_right'][0, :]

    torch.save(calibs, FLAGS.outputfile)


if __name__ == "__main__":
    app.run(main)
