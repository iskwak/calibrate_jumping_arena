# check the quality of the calibrations

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
import random
import os
import utilities

FLAGS = flags.FLAGS
flags.DEFINE_string("frame_info", "../calibration/20220726_bigvideo_test/flipped_frames.pkl", "Calibration frames data.")
flags.DEFINE_string("calib_video", "/workspace/calibration/calibration_videos/merged/calibration.avi", "Calibrated Camera Output File Name.")
flags.DEFINE_string("camera_calibs", "../calibration/20220726_bigvideo_test/calibrated_cameras.pkl", "Calibrated camera data")
flags.DEFINE_string("out_dir", "/workspace/outputs/calib_check", "Output directory")
flags.DEFINE_string("stereo_calib", "/workspace/calibration/20220726_bigvideo_test/cam_01_opencv.mat", "Stereo calibration matfile")
flags.DEFINE_string("overlapping_samples", "/workspace/calibration/20220726_bigvideo_test/overlapped_sampled.pkl", "overlapping frames and sampled frames info")

def main(argv):
    del argv

    with open(FLAGS.frame_info, "rb") as fid:
        calib_frames = pickle.load(fid)

    with open(FLAGS.camera_calibs, "rb") as fid:
        camera_calibs = pickle.load(fid)

    with open(FLAGS.overlapping_samples, "rb") as fid:
        all_overlapped_sampled = pickle.load(fid)

    os.makedirs(FLAGS.out_dir, exist_ok=True)
    stereo_calib = scipy.io.loadmat(FLAGS.stereo_calib)

    # load up the calibration movie, and then use it to plot some reprojections.
    cap = cv2.VideoCapture(FLAGS.calib_video)
    if cap.isOpened() == False:
        exit()
    full_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(full_width / 3)
    offsets = [0, width, 2 * width]

    cam0_id = stereo_calib["cam0_id"][0, 0]
    cam1_id = stereo_calib["cam1_id"][0, 0]

    offset0 = offsets[cam0_id]
    offset1 = offsets[cam1_id]

    # need to figure out the right overlapped pair information to use.
    all_overlapped = all_overlapped_sampled["overlapped"]
    for i in range(len(all_overlapped)):
        if all_overlapped[i]["view1"] == cam0_id and all_overlapped[i]["view2"] == cam1_id:
            overlapped_idx = i
            break

    overlapped = all_overlapped[overlapped_idx]
    sampled_idx = all_overlapped_sampled["sampled_idx"][overlapped_idx]

    # filter out the 
    corners_cam0 = utilities.index_list(calib_frames[cam0_id].corners2, overlapped["overlapping1"])
    corners_cam1 = utilities.index_list(calib_frames[cam1_id].corners2, overlapped["overlapping2"])

    frame_idx = utilities.index_list(calib_frames[cam0_id].frame_numbers, overlapped["overlapping1"])
    frame_idx = utilities.index_list(frame_idx, sampled_idx)

    corners_cam0 = utilities.index_list(corners_cam0, sampled_idx)
    corners_cam1 = utilities.index_list(corners_cam1, sampled_idx)

    #objpoints = calib_frames[cam0_id].setup_obj_points()
    #objpoints = objpoints[:len(corners_cam0)]

    # construct the stereo data.
    R = stereo_calib["R"]
    T = stereo_calib["T"]

    mtx1 = np.zeros((3, 3))
    mtx1[0, 0] = stereo_calib["fc_right"][0, 0]
    mtx1[1, 1] = stereo_calib["fc_right"][0, 1]
    mtx1[0, 2] = stereo_calib["cc_right"][0, 0]
    mtx1[1, 2] = stereo_calib["cc_right"][0, 1]
    mtx1[2, 2] = 1

    mtx0 = np.zeros((3, 3))
    mtx0[0, 0] = stereo_calib["fc_left"][0, 0]
    mtx0[1, 1] = stereo_calib["fc_left"][0, 1]
    mtx0[0, 2] = stereo_calib["cc_left"][0, 0]
    mtx0[1, 2] = stereo_calib["cc_left"][0, 1]
    mtx0[2, 2] = 1

    distortion1 = stereo_calib["kc_right"]
    distortion0 = stereo_calib["kc_left"]

    # create the projection matrix
    RT_eye = np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1)
    RT = np.concatenate([R, T], axis = -1)
    #RT01 = np.concatenate([R02, T02], axis = -1)

    proj_mat0 = mtx0 @ RT_eye
    proj_mat1 = mtx1 @ RT

    target_out_dir = FLAGS.out_dir + "/targets"
    os.makedirs(target_out_dir, exist_ok=True)


    errors = np.zeros((len(frame_idx), 2))
    for i in range(len(frame_idx)):
        # for each frame triangulate and save the error
        triangulated = cv2.triangulatePoints(proj_mat0, proj_mat1, corners_cam0[i], corners_cam1[i])
        cam0_ref_points = triangulated/triangulated[3, :]
        cam0_ref_points = cam0_ref_points[:3, :].T

        imgpoints_reproj, _ = cv2.projectPoints(cam0_ref_points, np.eye(3), np.zeros((3,1)), mtx0, distortion0)

        #error = cv2.norm(corners_cam0[i], imgpoints_reproj, cv2.NORM_L2)/len(imgpoints_reproj)
        error = np.sqrt(np.sum(np.square(corners_cam0[i].squeeze() - imgpoints_reproj.squeeze()), axis=1)).sum() / len(imgpoints_reproj)
        errors[i, 0] = error

        imgpoints_reproj1, _ = cv2.projectPoints(cam0_ref_points, R, T, mtx1, distortion1)
        error = np.sqrt(np.sum(np.square(corners_cam1[i].squeeze() - imgpoints_reproj1.squeeze()), axis=1)).sum() / len(imgpoints_reproj)
        errors[i, 1] = error

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx[i])
        ret, frame = cap.read()
        plt.figure(figsize=(30, 10), dpi=100)
        plt.imshow(frame)

        corners0 = corners_cam0[i].squeeze()
        imgpoints0 = imgpoints_reproj.squeeze()
        plt.plot(corners0[:, 0]  + offset0, corners0[:, 1], 'bx')
        plt.plot(imgpoints0[:, 0] + offset0, imgpoints0[:, 1], 'rx')

        corners1 = corners_cam1[i].squeeze()
        imgpoints1 = imgpoints_reproj1.squeeze()
        plt.plot(corners1[:, 0] + offset1, corners1[:, 1], 'bx')
        plt.plot(imgpoints1[:, 0] + offset1, imgpoints1[:, 1], 'rx')
        #plt.show()
        plt.savefig(target_out_dir + "/{}_frame_{}.png".format(i, frame_idx[i]))
        plt.close()

    mean_error = error.mean()

    # save the results to compare to matlab. save the overlapped stuff to mat file so matlab can use the information to
    # compare to.
    out_dict = {
        "stereo_calib_name": os.path.basename(FLAGS.stereo_calib),
        "mean_error": mean_error,
        "errors": errors,
        "corners0": corners_cam0,
        "corners1": corners_cam1
    }

    out_basename = os.path.splitext(os.path.basename(FLAGS.stereo_calib))[0]
    scipy.io.savemat(FLAGS.out_dir + "/stereo_errors.mat", out_dict)
    cap.release()

if __name__ == "__main__":
    app.run(main)