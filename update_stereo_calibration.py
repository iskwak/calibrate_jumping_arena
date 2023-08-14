import numpy as np
import cv2
from absl import app
from absl import flags
import pickle
from calibrationdata import CalibrationFrames
import time
import scipy
import scipy.io
from scipy.cluster.vq import kmeans,vq,whiten
import calibrate_cameras
import os
import plot_all_sampled_overlapping

FLAGS = flags.FLAGS


# def calibrate_all_camera_pairs(calib_frames, all_overlapping_frames, camera_calibs):
#     cap = cv2.VideoCapture(FLAGS.input_video)
#     full_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     width = int(full_width / 3)
#     fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     offsets = [0, width, 2 * width]
#     ret, frame = cap.read()

#     # terribly named function... this is not all the corners of the targets. this is getting
#     # only the top left corner of the target, for all targets.
#     all_corners = calibrate_cameras.get_corners(frame, calib_frames, offsets, True)
#     all_overlapping_sampled_idx = []
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.0001)
#     #for overlapping_frames in all_overlapping_frames:
#     for i_overlap in range(len(all_overlapping_frames)):
#         overlapping_frames = all_overlapping_frames[i_overlap]
#         cam1_id = overlapping_frames["view1"]
#         cam2_id = overlapping_frames["view2"]

#         cam1 = camera_calibs[cam1_id]
#         cam2 = camera_calibs[cam2_id]

#         cam1_frames = calib_frames[cam1_id]
#         cam2_frames = calib_frames[cam2_id]
#         print("num overlapping: {}".format(len(overlapping_frames["overlapping1"])))

#         imgpoints1 = calibrate_cameras.index_list(cam1_frames.corners2, overlapping_frames["overlapping1"])
#         imgpoints2 = calibrate_cameras.index_list(cam2_frames.corners2, overlapping_frames["overlapping2"])
#         frame_idx = calibrate_cameras.index_list(cam1_frames.frame_numbers, overlapping_frames["overlapping1"])

#         clustering_corner_cam1 = all_corners[cam1_id][overlapping_frames["overlapping1"], :]
#         clustering_corner_cam2 = all_corners[cam2_id][overlapping_frames["overlapping2"], :]

#         # cluster the corners
#         rng = np.random.RandomState(100)
#         seed = 100
#         # num_clusters = 100

#         cluster_ids, centroids = cluster_corners(clustering_corner_cam1, FLAGS.num_frames, seed)
#         flat_sampled, flat_sampled_idx, flat_cluster_ids = sample_corners(rng, clustering_corner_cam1, cluster_ids, FLAGS.num_frames, num_samples=1)
#         all_overlapping_sampled_idx.append(flat_sampled_idx)

#         # do a sampling here...
#         imgpoints1 = calibrate_cameras.index_list(imgpoints1, flat_sampled_idx)
#         imgpoints2 = calibrate_cameras.index_list(imgpoints2, flat_sampled_idx)
#         objpoints = cam1_frames.setup_obj_points()
#         objpoints = objpoints[:len(imgpoints1)]
#         frame_idx = calibrate_cameras.index_list(frame_idx, flat_sampled_idx)
#         #import pdb; pdb.set_trace()

#         mtx1 = cam1["mtx"]
#         dist1 = cam1["dist"]

#         mtx2 = cam2["mtx"] 
#         dist2 = cam2["dist"]

#         start_time = time.time()
#         ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
#             objpoints, imgpoints1, imgpoints2, mtx1, dist1, mtx2, dist2, (512, 512), criteria=criteria,
#             flags=cv2.CALIB_FIX_INTRINSIC)
#         print("error: {}".format(ret))
#         print("Time taken: {}".format(time.time() - start_time))

#         # ret, _, _, _, _, R_test, T_test, E_test, F_test = cv2.stereoCalibrate(
#         #     objpoints, imgpoints2, imgpoints1, mtx2, dist2, mtx1, dist1, (512, 512), criteria=criteria,
#         #     flags=cv2.CALIB_FIX_INTRINSIC)

#         # reproject and check errors
#         RT1 = np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1)
#         RT2 = np.concatenate([R, T], axis = -1)
#         proj_mat1 = RT1
#         proj_mat2 = RT2

#         mean_error = 0
#         all_triangulated = []
#         all_reproj1 = []
#         all_reproj2 = []
#         all_frame_numbers = []
#         for i in range(len(imgpoints1)):
#             points1u = cv2.undistortPoints(imgpoints1[i], mtx1, dist1, R=None, P=None)
#             points2u = cv2.undistortPoints(imgpoints2[i], mtx2, dist2, R=None, P=None)
#             #triangulated = cv2.triangulatePoints(proj_mat1, proj_mat2, imgpoints1[i], imgpoints2[i])
#             triangulated = cv2.triangulatePoints(proj_mat1, proj_mat2, points1u, points2u)
#             cam1_ref_points = triangulated/triangulated[3, :]
#             cam1_ref_points = cam1_ref_points[:3, :].T
#             all_triangulated.append(cam1_ref_points)
#             all_frame_numbers.append(frame_idx[i])

#             #imgpoints_reproj, _ = cv2.projectPoints(cam1_ref_points, R, T, mtx1, dist1)
#             imgpoints_reproj, _ = cv2.projectPoints(cam1_ref_points, np.eye(3), np.zeros((3,1)), mtx1, dist1)
#             #error = cv2.norm(imgpoints1[i], imgpoints_reproj, cv2.NORM_L2)/len(imgpoints_reproj)
#             error = np.sqrt(np.sum(np.square(imgpoints1[i].squeeze() - imgpoints_reproj.squeeze()), axis=1)).sum() / len(imgpoints_reproj)
#             mean_error += error
#             all_reproj1.append(imgpoints_reproj)

#             imgpoints_reproj2, _ = cv2.projectPoints(cam1_ref_points, R, T, mtx2, dist2)
#             all_reproj2.append(imgpoints_reproj2)
#             #error = cv2.norm(imgpoints2[i], imgpoints_reproj2, cv2.NORM_L2)/len(imgpoints_reproj)
#             error = np.sqrt(np.sum(np.square(imgpoints2[i].squeeze() - imgpoints_reproj2.squeeze()), axis=1)).sum() / len(imgpoints_reproj)
#             mean_error += error
#             #import pdb; pdb.set_trace()

#             # # manual change of frame of reference to test...
#             # cam2_ref_points = (R @ cam1_ref_points.T + T).T
#             # test_points, _ = cv2.projectPoints(cam2_ref_points, np.eye(3), np.zeros((3,1)), mtx2, dist2)
#             # test_points = test_points.astype('float32')
#             # test_error = cv2.norm(test_points, imgpoints_reproj2, cv2.NORM_L2)/len(imgpoints_reproj)

#         print( "total error: {}".format(mean_error/(2*len(objpoints))) )
#         #print( "total error: {}".format(mean_error/(len(objpoints))) )

#         # write the data to a mat file.
#         # need, R, T, square size, num squares, fc, cc and skew.
#         # {'om' 'T' 'R' 'active_images_left' 'recompute_intrinsic_right'}
#         om = cv2.Rodrigues(R)
#         om = om[0]
#         out_dict = {
#             "calib_name_left": "cam_{}".format(cam1_id),
#             "calib_name_right": "cam_{}".format(cam2_id),
#             "cam0_id": cam1_id,
#             "cam1_id": cam2_id,
#             "dX": 3,
#             "nx": 512,
#             "ny": 512,
#             "fc_left": [mtx1[0, 0], mtx1[1, 1]],
#             "cc_left": [mtx1[0, 2], mtx1[1, 2]],
#             "alpha_c_left": 0.0, # opencv doesnt use the skew parameter
#             "kc_left": dist1,
#             "fc_right": [mtx2[0, 0], mtx2[1, 1]],
#             "cc_right": [mtx2[0, 2], mtx2[1, 2]],
#             "alpha_c_right": 0.0, # opencv doesnt use the skew parameter
#             "kc_right": dist2,
#             "om": om,
#             "R": R,
#             "T": T,
#             "F": F,
#             "active_images_left": [],
#             "cc_left_error": 0,
#             "cc_right_error": 0,
#             "recompute_intrinsic_right": 1,
#             "recompute_intrinsic_left": 1
#         }
#         #scipy.io.savemat("{}/cam_{}{}_opencv.mat".format(FLAGS.out_dir, cam1_id, cam2_id), out_dict)
#         # save sampled points for testing in matlab.
#         out_dict2 = {
#             "imgpoints1": imgpoints1,
#             "imgpoints2": imgpoints2,
#             "all_triangulated": all_triangulated,
#             "all_reproj1": all_reproj1,
#             "all_reproj2": all_reproj2,
#             "frame_numbers": all_frame_numbers
#         }
#         print("saving...")
#         #scipy.io.savemat("{}/sampled_{}{}.mat".format(FLAGS.out_dir, cam1_id, cam2_id), out_dict2)

#     overlapped_sampled_name = FLAGS.out_dir + "/overlapped_sampled.pkl"
#     overlapped_sampled = {
#         "overlapped": all_overlapping_frames,
#         "sampled_idx": all_overlapping_sampled_idx
#     }
#     # with open(overlapped_sampled_name, "wb") as fid:
#     #     pickle.dump(overlapped_sampled, fid)

#     cap.release()


def main(argv):
    del argv

    with open(FLAGS.calib_frames, "rb") as fid:
        calib_frames = pickle.load(fid)
    objpoints = calib_frames[0].setup_obj_points()

    with open(FLAGS.overlapping_sampled, "rb") as fid:
        overlapped_sampled = pickle.load(fid)

    with open(FLAGS.calibrated_name, "rb") as fid:
        calibration_data = pickle.load(fid)
    camera_calibs = calibration_data["calibrated"]

    # setup output directory
    base_out_dir = FLAGS.out_dir + "/update_stereo"
    os.makedirs(base_out_dir, exist_ok=True)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.0001)
    num_camera_pairs = len(overlapped_sampled['overlapped'])
    for i in range(num_camera_pairs):
        view1 = overlapped_sampled['overlapped'][i]['view1']
        view2 = overlapped_sampled['overlapped'][i]['view2']

        overlapping_idx1 = overlapped_sampled['overlapped'][i]['overlapping1']
        overlapping_idx2 = overlapped_sampled['overlapped'][i]['overlapping2']

        calib_frames1 = calib_frames[view1]
        calib_frames2 = calib_frames[view2]
        
        [overlapping_corners1, overlapping_frames1] = plot_all_sampled_overlapping.get_overlapping(calib_frames1, overlapping_idx1)
        [overlapping_corners2, overlapping_frames2] = plot_all_sampled_overlapping.get_overlapping(calib_frames2, overlapping_idx2)

        overlapping_corners1 = np.squeeze(overlapping_corners1)
        overlapping_corners2 = np.squeeze(overlapping_corners2)

        sampled_corners1 = overlapping_corners1[overlapped_sampled['sampled_idx'][i], :, :]
        sampled_corners2 = overlapping_corners2[overlapped_sampled['sampled_idx'][i], :, :]
        sampled_frames = overlapping_frames1[overlapped_sampled['sampled_idx'][i]]

        [num_frames, num_corners] = sampled_corners1.shape[0:2]
        sampled_corners1_flat = sampled_corners1.reshape(num_frames * num_corners, 2)
        sampled_corners2_flat = sampled_corners2.reshape(num_frames * num_corners, 2)

        # get single view camera calibration data
        mtx1 = camera_calibs[view1]["mtx"]
        dist1 = camera_calibs[view1]["dist"]

        mtx2 = camera_calibs[view2]["mtx"]
        dist2 = camera_calibs[view2]["dist"]

        # the "fix" in "fix intrinsic" here means hold fixed, not repair.
        # important to note because stereo calibrate can update the intrinsic estimates.
        tic = time.time()
        ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
            objpoints[0:len(sampled_corners1)], sampled_corners1, sampled_corners2,
            mtx1, dist1, mtx2, dist2, (512, 512), criteria=criteria, flags=cv2.CALIB_FIX_INTRINSIC)
        print(time.time() - tic)
        points1u = cv2.undistortPoints(sampled_corners1_flat, mtx1, dist1, R=None, P=None)
        points2u = cv2.undistortPoints(sampled_corners2_flat, mtx2, dist2, R=None, P=None)
        cv2.findEssentialMat(points1u, points2u, np.eye(3))

    # # first collect overlapping frames from each pair of cameras    
    # all_overlapping = get_all_overlapping_frames(calib_frames) 

    # # calibrate each pair of cameras
    # calibrate_all_camera_pairs(calib_frames, all_overlapping, camera_calibs)
    # print('hello')


if __name__ == "__main__":
    app.run(main)
