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
import calibrate_cameras
import os

FLAGS = flags.FLAGS
# flags.DEFINE_string("calib_frames", None, "Calibration frames data.")
# flags.DEFINE_string("calibrated_name", None, "Calibrated Camera Output File Name.")




def main(argv):
    del argv

    # get the overlapped frame samples
    overlapping_filename = FLAGS.out_dir + "/overlapped_sampled.pkl"
    with open(overlapping_filename, "rb") as fid:
        overlapping_sampled = pickle.load(fid)

    # get the original corner data
    calib_frames_filename = FLAGS.out_dir + "/filtered_frames.pkl"
    with open(calib_frames_filename, "rb") as fid:
        calib_frames = pickle.load(fid)


    import pdb; pdb.set_trace()
    # with open(FLAGS.calib_frames, "rb") as fid:
    #     calib_frames = pickle.load(fid)

    # with open(FLAGS.calibrated_name, "rb") as fid:
    #     calibration_data = pickle.load(fid)
    # camera_calibs = calibration_data["calibrated"]

    # # first collect overlapping frames from each pair of cameras    
    # all_overlapping = get_all_overlapping_frames(calib_frames) 

    # # calibrate each pair of cameras
    # calibrate_all_camera_pairs(calib_frames, all_overlapping, camera_calibs)


if __name__ == "__main__":
    app.run(main)
