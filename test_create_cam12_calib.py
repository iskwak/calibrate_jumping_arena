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
import stereo_calibration
import calibrate_cameras


FLAGS = flags.FLAGS
flags.DEFINE_string("calib02", "../calibration/20220726_bigvideo_test/cam_02_opencv.mat", "First calib file")
flags.DEFINE_string("calib12", "../calibration/20220726_bigvideo_test/cam_12_opencv.mat", "Second calib file")
flags.DEFINE_string("frame_info", "../calibration/20220726_bigvideo_test/flipped_frames.pkl", "Calibration frames data.")
flags.DEFINE_string("calib_video", "/workspace/calibration/calibration_videos/merged/calibration.avi", "Calibrated Camera Output File Name.")
flags.DEFINE_string("camera_calibs", "../calibration/20220726_bigvideo_test/calibrated_cameras.pkl", "Calibrated camera data")


def draw_stereo_reprojection(frame, imgpoints, imgpoints2, offset, cam1_id, cam2_id):
    mins = imgpoints.min(axis=0).squeeze().astype('int')
    maxs = imgpoints.max(axis=0).squeeze().astype('int')

    mins = mins - 60
    if mins[0] < 0:
        mins[0] = 0
    if mins[1] < 0:
        mins[1] = 0
    maxs = mins + 180

    # imgpoints[:, 0, 0] = imgpoints[:, 0, 0] + offset
    # imgpoints2[:, 0, 0] = imgpoints2[:, 0, 0] + offset

    corners = imgpoints.squeeze()
    calibrate_cameras.draw_corners(frame, corners, (255, 0, 255), 5)
    # corners = imgpoints2.squeeze()
    #calibrate_cameras.draw_corners(frame, corners, (0, 255, 255), 5)
    # adjust the corners
    # mins[0] = mins[0] + offset
    # maxs[0] = maxs[0] + offset
    # frame = frame[mins[1]:maxs[1], mins[0]:maxs[0]]

    return frame


def main(argv):
    del argv
    with open(FLAGS.frame_info, "rb") as fid:
        calib_frames = pickle.load(fid)
    with open(FLAGS.camera_calibs, "rb") as fid:
        camera_calibs = pickle.load(fid)

    calib02 = scipy.io.loadmat(FLAGS.calib02)
    calib12 = scipy.io.loadmat(FLAGS.calib12)
    calib01 = scipy.io.loadmat("../calibration/20220726_bigvideo_test/cam_01_opencv.mat")
    # calib1 goes from camera 0, camera facing the mouse's right side, to camera 2, camera facing mouse's face.
    # calib2 goes from camera 1, camera facing the mouse's right side, to camera 2, camera facing mouse's face.
    # going to create transformations that go from camera 0 to camera 1, by going through camera 2.
    # camera 0 -> camera 2 -> camera 1
    R02 = calib02["R"]
    T02 = calib02["T"]
    R21 = calib12["R"].T
    T21 = np.dot(-R21, calib12["T"])

    # the transformation goes from the "right" camera to the "left" camera. So camera 0, or the first camera, is the
    # right camera. A bit confusing... probably need to update the APT stuff to be a bit more friendly at some point.
    mtx0 = np.zeros((3, 3))
    mtx0[0, 0] = calib02["fc_right"][0, 0]
    mtx0[1, 1] = calib02["fc_right"][0, 1]
    mtx0[0, 2] = calib02["cc_right"][0, 0]
    mtx0[1, 2] = calib02["cc_right"][0, 1]
    mtx0[2, 2] = 1

    mtx1 = np.zeros((3, 3))
    mtx1[0, 0] = calib12["fc_right"][0, 0]
    mtx1[1, 1] = calib12["fc_right"][0, 1]
    mtx1[0, 2] = calib12["cc_right"][0, 0]
    mtx1[1, 2] = calib12["cc_right"][0, 1]
    # mtx1[0, 0] = calib02["fc_left"][0, 0]
    # mtx1[1, 1] = calib02["fc_left"][0, 1]
    # mtx1[0, 2] = calib02["cc_left"][0, 0]
    # mtx1[1, 2] = calib02["cc_left"][0, 1]
    mtx1[2, 2] = 1

    R01 = np.dot(R21, R02)
    T01 = np.dot(R21, T02) + T21
    # R01 = calib01["R"]
    # T01 = calib01["T"]

    # create the projection matrix
    RT_eye = np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1)
    RT01 = np.concatenate([R01, T01], axis = -1)
    #RT01 = np.concatenate([R02, T02], axis = -1)

    proj_mat0 = mtx0 @ RT_eye
    proj_mat1 = mtx1 @ RT01

    # load up the calibration movie, and then use it to plot some reprojections.
    cap = cv2.VideoCapture(FLAGS.calib_video)
    if cap.isOpened() == False:
        exit()
    full_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(full_width / 3)
    offsets = [0, width, 2 * width]

    # i guess we need to find the overlapping frames... wonder if the means i should be saving this information
    # somewhere. i don't think it is a super fast process
    all_overlapping = stereo_calibration.get_all_overlapping_frames(calib_frames) 

    # for each overlapping pair between cam0 and cam1, calculate the reprojection error
    overlapped_points = all_overlapping[0]
    num_overlapping = len(overlapped_points["overlapping1"])
    cam0_id = overlapped_points["view1"]
    cam1_id = overlapped_points["view2"]
    calib0 = calib_frames[cam0_id]
    calib1 = calib_frames[cam1_id]

    mean_error = 0
    for i in range(num_overlapping):
        view0_idx = overlapped_points["overlapping1"][i]
        view1_idx = overlapped_points["overlapping2"][i]

        frame_num0 = calib0.frame_numbers[view0_idx]
        frame_num1 = calib1.frame_numbers[view1_idx]
        assert(frame_num0 == frame_num1)
        imgpoints0 = calib0.corners2[view0_idx]
        imgpoints1 = calib1.corners2[view1_idx]

        # jump to the frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num0)
        ret, frame = cap.read()
        cv2.waitKey()
        if ret == True:
            triangulated = cv2.triangulatePoints(proj_mat0, proj_mat1, imgpoints0, imgpoints1)
            cam0_ref_points = triangulated/triangulated[3, :]
            cam0_ref_points = cam0_ref_points[:3, :].T
            distortion0 = camera_calibs["calibrated"][cam0_id]["dist"]
            distortion1 = camera_calibs["calibrated"][cam1_id]["dist"]

            imgpoints_reproj, _ = cv2.projectPoints(cam0_ref_points, np.eye(3), np.zeros((3,1)), mtx0, distortion0)

            error0 = cv2.norm(imgpoints0, imgpoints_reproj, cv2.NORM_L2)/len(imgpoints_reproj)
            mean_error += error0

            imgpoints_reproj1, _ = cv2.projectPoints(cam0_ref_points, R01, T01, mtx1, distortion1)
            error1 = cv2.norm(imgpoints1, imgpoints_reproj1, cv2.NORM_L2)/len(imgpoints_reproj)
            mean_error += error1

            if error0 > 0:
                print("error: {}".format(error0))
                new_frame =  draw_stereo_reprojection(frame, imgpoints0, imgpoints_reproj, offsets[cam0_id], cam0_id, cam1_id)
                cv2.imshow("moo", new_frame)
                cv2.waitKey()
                #import pdb; pdb.set_trace()
    print("total error: {}".format(mean_error/num_overlapping))
    # print("total error: {}".format(mean_error)

    # triangulated = cv2.triangulatePoints(proj_mat1, proj_mat2, imgpoints1[i], imgpoints2[i])
    # cam1_ref_points = triangulated/triangulated[3, :]
    # cam1_ref_points = cam1_ref_points[:3, :].T

    # #imgpoints_reproj, _ = cv2.projectPoints(cam1_ref_points, R, T, mtx1, dist1)
    # imgpoints_reproj, _ = cv2.projectPoints(cam1_ref_points, np.eye(3), np.zeros((3,1)), mtx1, dist1)
    # error = cv2.norm(imgpoints1[i], imgpoints_reproj, cv2.NORM_L2)/len(imgpoints_reproj)
    # mean_error += error
    # write_stereo_reprojection(cap, imgpoints1[i], imgpoints_reproj, frame_idx[i], offsets[cam1_id], cam1_id, cam2_id)

    # imgpoints_reproj2, _ = cv2.projectPoints(cam1_ref_points, R, T, mtx2, dist2)

    # # try to see if there are any frames where the calibration target is visible in all 3 frames.
    # # ... don't need to do a loop over all the all_overlapping's twice
    # # The all_overlapping list has 3 dictionaries. Overlapping frames betweeen camera's (0, 1), (0, 2), (1, 2), in that
    # # order. Want frames that appear in (0, 1) and (1, 2). Having trouble thinking this one out.
    # camera_idx1 = all_overlapping[0]["view1"]
    # for j in range(len(all_overlapping)):
    #     camera_idx2 = all_overlapping[j]["view1"]

    #     if 0 == j:
    #         continue
    #     # there is a smarter way to do this, but whatever, make it quick and dirty
    #     for i_over1 in range(len(all_overlapping[i]["overlapping1"])):
    #         frame_numbers_idx1 = all_overlapping[i]["overlapping1"][i_over1]
    #         frame1 = camera_calibs[camera_idx1].frame_numbers[frame_numbers_idx1]

    #         for j_over2 in range(len(all_overlapping[j]["overlapping1"])):                    
    #             frame_numbers_idx2 = all_overlapping[j]["overlapping1"][i_over2]                     
    #             frame2 = camera_calibs[camera_idx2].frame_numbers[frame_numbers_idx2]


if __name__ == "__main__":
    app.run(main)
