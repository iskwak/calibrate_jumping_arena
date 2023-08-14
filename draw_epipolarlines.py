import numpy as np
import cv2
# import glob
from matplotlib import pyplot as plt
from absl import app
from absl import flags
import pickle
from calibrationdata import CalibrationFrames
import scipy
import scipy.io
import random
import calibrate_cameras
import os
import utilities


FLAGS = flags.FLAGS
flags.DEFINE_string("frame_info", "../calibration/20220913_stereo_test/filtered_frames.pkl", "Calibration frames data.")
flags.DEFINE_string("calib_video", "/workspace/calibration/calibration_videos/merged/calibration.avi", "Calibrated Camera Output File Name.")
flags.DEFINE_string("camera_calibs", "../calibration/20220913_stereo_test/calibrated_cameras.pkl", "Calibrated camera data")
#flags.DEFINE_string("out_dir", "/workspace/outputs/calib_check", "Output directory")
flags.DEFINE_string("stereo_calib", "/workspace/calibration/20220913_stereo_test/cam_02_opencv.mat", "Stereo calibration matfile")
flags.DEFINE_string("overlapping_samples", "/workspace/calibration/20220913_stereo_test/overlapped_sampled.pkl", "overlapping frames and sampled frames info")


def drawlines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    fig, ax = plt.subplots(1, 2)
    r,c,_ = img1.shape
    #img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    #img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    ax[0].imshow(img1)
    ax[1].imshow(img2)

    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1] ])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        
        ax[0].plot([x0, x1-1], [y0, y1-1])
        ax[0].scatter(pt1[0], pt1[1], marker='x')        

        ax[1].scatter(pt2[0], pt2[1], marker='x')
        # img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        # img1 = cv2.circle(img1,(int(pt1[0]), int(pt1[1])),5,color,-1)
        # img2 = cv2.circle(img2,(int(pt1[0]), int(pt1[1])),5,color,-1)
    plt.show()
    return fig


def main(argv):
    del argv

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.0001)
    with open(FLAGS.frame_info, "rb") as fid:
        checkerboard_frames = pickle.load(fid)

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

    R = stereo_calib["R"]
    T = stereo_calib["T"]
    F = stereo_calib["F"]

    
    cam1_id = stereo_calib["cam0_id"][0, 0]
    cam2_id = stereo_calib["cam1_id"][0, 0]

    offset1 = offsets[cam1_id]
    offset2 = offsets[cam2_id]

    # need to figure out the right overlapped pair information to use.
    all_overlapped = all_overlapped_sampled["overlapped"]
    for i in range(len(all_overlapped)):
        if all_overlapped[i]["view1"] == cam1_id and all_overlapped[i]["view2"] == cam2_id:
            overlapped_idx = i
            break

    overlapped = all_overlapped[overlapped_idx]
    sampled_idx = all_overlapped_sampled["sampled_idx"][overlapped_idx]

    # filter out the 
    corners_cam1 = utilities.index_list(checkerboard_frames[cam1_id].corners2, overlapped["overlapping1"])
    corners_cam2 = utilities.index_list(checkerboard_frames[cam2_id].corners2, overlapped["overlapping2"])

    frame_idx = utilities.index_list(checkerboard_frames[cam1_id].frame_numbers, overlapped["overlapping1"])
    frame_idx = utilities.index_list(frame_idx, sampled_idx)

    corners_cam1 = utilities.index_list(corners_cam1, sampled_idx)
    corners_cam2 = utilities.index_list(corners_cam2, sampled_idx)

    #objpoints = checkerboard_frames[cam0_id].setup_obj_points()
    #objpoints = objpoints[:len(corners_cam0)]

    mtx2 = np.zeros((3, 3))
    mtx2[0, 0] = stereo_calib["fc_right"][0, 0]
    mtx2[1, 1] = stereo_calib["fc_right"][0, 1]
    mtx2[0, 2] = stereo_calib["cc_right"][0, 0]
    mtx2[1, 2] = stereo_calib["cc_right"][0, 1]
    mtx2[2, 2] = 1

    mtx1 = np.zeros((3, 3))
    mtx1[0, 0] = stereo_calib["fc_left"][0, 0]
    mtx1[1, 1] = stereo_calib["fc_left"][0, 1]
    mtx1[0, 2] = stereo_calib["cc_left"][0, 0]
    mtx1[1, 2] = stereo_calib["cc_left"][0, 1]
    mtx1[2, 2] = 1

    distortion2 = stereo_calib["kc_right"]
    distortion1 = stereo_calib["kc_left"]

    # # create the projection matrix
    # RT_eye = np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1)
    # RT = np.concatenate([R, T], axis = -1)
    # #RT01 = np.concatenate([R02, T02], axis = -1)

    # proj_mat0 = mtx0 @ RT_eye
    # proj_mat1 = mtx1 @ RT

    # target_out_dir = FLAGS.out_dir + "/targets"
    # os.makedirs(target_out_dir, exist_ok=True)


    errors = np.zeros((len(frame_idx), 2))
    for i in range(len(frame_idx)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx[i])
        ret, frame = cap.read()
        
        frame1 = frame[:, offset1:offset1 + 512, :]
        frame2 = frame[:, offset2:offset2 + 512, :]
        # plt.figure()
        # plt.imshow(frame1)
        # plt.figure()
        # plt.imshow(frame2)
        # plt.figure()
        # plt.imshow(frame)
        # plt.show()
        # plt.close()

        h,  w = frame1.shape[:2]
        newcameramtx1, roi1 = cv2.getOptimalNewCameraMatrix(mtx1, distortion1, (w,h), 1, (w,h))
        # undistort
        undist1 = cv2.undistort(frame1, mtx1, distortion1, None, None)
        map1_x, map1_y = cv2.initUndistortRectifyMap(mtx1, distortion1, None, newcameramtx1,
            (w,h), cv2.CV_32FC1, (w,h))
        undist1_map = cv2.remap(frame1, map1_x, map1_y, cv2.INTER_CUBIC);

        # crop the image
        x1, y1, w1, h1 = roi1
        undist1 = undist1[y1:y1+h1, x1:x1+w1]
        
        newcameramtx2, roi2 = cv2.getOptimalNewCameraMatrix(mtx2, distortion2, (w,h), 1, (w,h))
        # undistort
        undist2 = cv2.undistort(frame2, mtx2, distortion2, None, newcameramtx2)
        map2_x, map2_y = cv2.initUndistortRectifyMap(mtx2, distortion2, None, newcameramtx2,
            (w,h), cv2.CV_32FC1, (w,h))
        undist2_map = cv2.remap(frame2, map2_x, map2_y, cv2.INTER_CUBIC);

        # crop the image
        x2, y2, w2, h2 = roi2
        undist2 = undist2[y2:y2+h2, x2:x2+w2]
        #undist2_map = undist2_map[y2:y2+h2, x2:x2+w2]

        corner1 = corners_cam1[i][[0, 6, 41], :, :].squeeze()
        corner2 = corners_cam2[i][[0, 6, 41], :, :].squeeze()
        # undistort this point and then plot
        undist_corner1 = cv2.undistortPointsIter(corner1, mtx1, distortion1, None, newcameramtx1, criteria=criteria)
        undist_corner2 = cv2.undistortPointsIter(corner2, mtx2, distortion2, None, newcameramtx2, criteria=criteria)
        
        undist_corner1_2 = [
            map1_x[int(corner1[0,0]), int(corner1[0,1])],
            map1_y[int(corner1[0,0]), int(corner1[0,1])]]

        undist_corner2_2 = [
            map2_x[int(corner2[1, 0]), int(corner2[1, 1])],
            map2_y[int(corner2[1, 0]), int(corner2[1, 1])]]


        # _, ax = plt.subplots(1, 2)
        # ax[0].imshow(undist1)
        # ax[1].imshow(frame1)
        # #ax[0].scatter(undist_corner1[:, :, 0], undist_corner1[:, :, 1], marker='x')
        # ax[0].scatter(undist_corner1_2[0], undist_corner1_2[1], marker='x')
        # ax[1].scatter(corner1[0, 0], corner1[0, 1], marker='x')

        # _, ax = plt.subplots(1, 2)
        # ax[0].imshow(undist2_map)
        # ax[1].imshow(frame2)
        # #ax[0].scatter(undist_corner2[:, :, 0], undist_corner2[:, :, 1], marker='x')
        # #ax[0].scatter(undist_corner2_2[:, 0], undist_corner2_2[:, 1], marker='x')
        # ax[1].scatter(corner2[:, 0], corner2[:, 1], marker='x')
        # plt.show()

        test_corner = np.asarray([[314.49, 265.92]]).astype('float32')
        lines_img2 = cv2.computeCorrespondEpilines(test_corner, 1, F)
        lines_img2 = lines_img2.reshape(-1,3)
        fig = drawlines(undist2_map, undist1, lines_img2, np.asarray([[0,0]]).astype('float32'), test_corner)


# # Find epilines corresponding to points in right image (second image) and
# # drawing its lines on left image
# lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
# lines1 = lines1.reshape(-1,3)
# img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)
# # Find epilines corresponding to points in left image (first image) and
# # drawing its lines on right image
# lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
# lines2 = lines2.reshape(-1,3)
# img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)
# plt.subplot(121),plt.imshow(img5)
# plt.subplot(122),plt.imshow(img3)
# plt.show()


if __name__ == "__main__":
    app.run(main)
