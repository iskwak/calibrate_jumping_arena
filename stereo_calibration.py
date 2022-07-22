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

FLAGS = flags.FLAGS
# flags.DEFINE_string("calib_frames", None, "Calibration frames data.")
# flags.DEFINE_string("calibrated_name", None, "Calibrated Camera Output File Name.")


def get_overlapping_frames(cam1, cam2):
    overlapping_cam1_idx = []
    overlapping_cam2_idx = []
    cam1_frames = cam1.frame_numbers
    cam2_frames = cam2.frame_numbers

    for i in range(len(cam1_frames)):
        for j in range(len(cam2_frames)):
            if cam1_frames[i] == cam2_frames[j]:
                overlapping_cam1_idx.append(i)
                overlapping_cam2_idx.append(j)

    return overlapping_cam1_idx, overlapping_cam2_idx


def get_all_overlapping_frames(calib_frames):
    num_cams = len(calib_frames)
    all_overlapping = []
    for i in range(num_cams):
        for j in range(num_cams):
            if i >= j:
                continue
            
            overlap1, overlap2 = get_overlapping_frames(calib_frames[i], calib_frames[j])
            all_overlapping.append({
                "view1": i,
                "view2": j,
                "overlapping1": overlap1,
                "overlapping2": overlap2
            })
    return all_overlapping


def write_stereo_reprojection(cap, imgpoints, imgpoints2, frame_idx, offset, cam1_id, cam2_id):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
    ret, frame = cap.read()

    mins = imgpoints.min(axis=0).squeeze().astype('int')
    maxs = imgpoints.max(axis=0).squeeze().astype('int')

    mins = mins - 60
    if mins[0] < 0:
        mins[0] = 0
    if mins[1] < 0:
        mins[1] = 0
    maxs = mins + 180

    # adjust the corners
    #frame = frame[mins[1]:maxs[1], mins[0]:maxs[0]]

    # need to flip the corners
    frame_reproj = frame.copy()

    imgpoints[:, 0, 0] = imgpoints[:, 0, 0] + offset
    imgpoints2[:, 0, 0] = imgpoints2[:, 0, 0] + offset
    # imgpoints[:, 0, 0] = imgpoints[:, 0, 0] - mins[0]
    # imgpoints[:, 0, 1] = imgpoints[:, 0, 1] - mins[1]
    # imgpoints2[:, 0, 0] = imgpoints2[:, 0, 0] - mins[0]
    # imgpoints2[:, 0, 1] = imgpoints2[:, 0, 1] - mins[1]
    # cv2.drawChessboardCorners(frame, squares_xy, imgpoints, True)
    # draw_corner_numbers(frame_flipped, reordered)
    # # mark the frame number on a flipped example
    # cv2.putText(frame_flipped, "{}: {}".format(i, frame_num),
    #     (20, 20),
    #     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (209, 80, 0, 255), 1)
    corners = imgpoints.squeeze()
    calibrate_cameras.draw_corners(frame, corners, (255, 0, 255), 5)
    corners = imgpoints2.squeeze()
    calibrate_cameras.draw_corners(frame, corners, (0, 255, 255), 5)

    # cv2.imshow("frame", frame)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    cv2.imwrite("stereo_reprojections/{}{}_{}.png".format(cam1_id, cam2_id, frame_idx), frame)


def write_stereo_points(cap, cam1points, cam2points, frame_idx, offset1, offset2, cam1_id, cam2_id):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
    ret, frame = cap.read()

    # adjust the corners
    #frame = frame[mins[1]:maxs[1], mins[0]:maxs[0]]

    # need to flip the corners
    frame_reproj = frame.copy()

    cam1points[:, 0, 0] = cam1points[:, 0, 0]
    cam2points[:, 0, 0] = cam2points[:, 0, 0]
    # imgpoints[:, 0, 0] = imgpoints[:, 0, 0] - mins[0]
    # imgpoints[:, 0, 1] = imgpoints[:, 0, 1] - mins[1]
    # imgpoints2[:, 0, 0] = imgpoints2[:, 0, 0] - mins[0]
    # imgpoints2[:, 0, 1] = imgpoints2[:, 0, 1] - mins[1]
    # cv2.drawChessboardCorners(frame, squares_xy, imgpoints, True)
    # draw_corner_numbers(frame_flipped, reordered)
    # # mark the frame number on a flipped example
    # cv2.putText(frame_flipped, "{}: {}".format(i, frame_num),
    #     (20, 20),
    #     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (209, 80, 0, 255), 1)
    corners = cam1points.squeeze()
    draw_corners_with_offset(frame, corners, (255, 0, 255), 5, offset1)

    corners = cam2points.squeeze()
    draw_corners_with_offset(frame, corners, (0, 255, 255), 5, offset2)

    # cv2.imshow("frame", frame)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    cv2.imwrite("paired/{}{}_{}.png".format(cam1_id, cam2_id, frame_idx), frame)


def draw_corners_with_offset(image, corners, color, markerSize, offset):
    for i in range(len(corners)):
        cv2.drawMarker(image, (int(corners[i, 0])+offset, int(corners[i, 1])), color,
            markerType=cv2.MARKER_CROSS, markerSize=markerSize)



def write_all_stereo_points(cap, imgpoints1, imgpoints2, frame_idx, offsets, cam1_id, cam2_id):
    for i in range(len(imgpoints1)):
        cam1points = imgpoints1[i]
        cam2points = imgpoints2[i]
        write_stereo_points(cap, cam1points, cam2points, frame_idx[i], offsets[cam1_id], offsets[cam2_id], cam1_id, cam2_id)        


def sample_corners(rng, current_corners, cluster_ids, num_clusters, num_samples=10):
    # store the sampled points here, then concatentate
    sampled_corners =  []
    sampled_idx = [] # indexing into the corners
    sampled_cluster_idx = [] # looks dumb, but useful for the drawing function

    for j in range(num_clusters):
        clustered_indices = np.where(cluster_ids == j)
        clustered_indices = clustered_indices[0] # where returns a tuple for me
        rng.shuffle(clustered_indices) # in place operation
        sampled_idx.append(clustered_indices[:num_samples])
        sampled = calibrate_cameras.index_list(current_corners, clustered_indices[:num_samples])
        sampled_corners.append(sampled)
        sampled_cluster_idx.append([j] * num_samples)

    flat_sampled = np.concatenate(sampled_corners, axis=0)
    flat_cluster_ids = [x for xs in sampled_cluster_idx for x in xs]
    flat_sampled_idx = [x for xs in sampled_idx for x in xs]

    return flat_sampled, flat_sampled_idx, flat_cluster_ids


def cluster_corners(corners, num_clusters, seed):
    centroids, _ = kmeans(corners, num_clusters, seed=seed)
    clx, _ = vq(corners, centroids)

    return clx


def calibrate_all_camera_pairs(calib_frames, all_overlapping_frames, camera_calibs):
    cap = cv2.VideoCapture(FLAGS.input_video)
    full_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(full_width / 3)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fps = cap.get(cv2.CAP_PROP_FPS)
    offsets = [0, width, 2 * width]
    ret, frame = cap.read()
    num_clusters = 10 
    all_corners = calibrate_cameras.get_corners(frame, calib_frames, offsets, True)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    all_calibs = []
    for overlapping_frames in all_overlapping_frames:
        cam1_id = overlapping_frames["view1"]
        cam2_id = overlapping_frames["view2"]

        cam1 = camera_calibs[cam1_id]
        cam2 = camera_calibs[cam2_id]

        cam1_frames = calib_frames[cam1_id]
        cam2_frames = calib_frames[cam2_id]
        print("num overlapping: {}".format(len(overlapping_frames["overlapping1"])))

        imgpoints1 = calibrate_cameras.index_list(cam1_frames.corners2, overlapping_frames["overlapping1"])
        imgpoints2 = calibrate_cameras.index_list(cam2_frames.corners2, overlapping_frames["overlapping2"])
        frame_idx = calibrate_cameras.index_list(cam1_frames.frame_numbers, overlapping_frames["overlapping1"])
        
        clustering_corner_cam1 = all_corners[cam1_id][overlapping_frames["overlapping1"], :]
        clustering_corner_cam2 = all_corners[cam2_id][overlapping_frames["overlapping2"], :]
        
        # cluster the corners
        rng = np.random.RandomState(123)
        seed = 123
        num_clusters = 10

        cluster_ids = cluster_corners(clustering_corner_cam1, num_clusters, seed)
        flat_sampled, flat_sampled_idx, flat_cluster_ids = sample_corners(rng, imgpoints1, cluster_ids, num_clusters, num_samples=10)
        #import pdb; pdb.set_trace()

        # do a sampling here...
        imgpoints1 = calibrate_cameras.index_list(imgpoints1, flat_sampled_idx)
        imgpoints2 = calibrate_cameras.index_list(imgpoints2, flat_sampled_idx)
        objpoints = cam1_frames.setup_obj_points()
        objpoints = objpoints[:100]
        frame_idx = calibrate_cameras.index_list(frame_idx, flat_sampled_idx)
        write_all_stereo_points(cap, imgpoints1, imgpoints2, frame_idx, offsets, cam1_id, cam2_id)

        mtx1 = cam1["mtx"]
        dist1 = cam1["dist"]

        mtx2 = cam2["mtx"] 
        dist2 = cam2["dist"]

        # matlab has the order flipped... so we gotta do that here
        # ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        #     objpoints, imgpoints2, imgpoints1, mtx2, dist2, mtx1, dist1, (512, 512), criteria=criteria,
        #     flags=cv2.CALIB_FIX_INTRINSIC)
        ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
            objpoints, imgpoints1, imgpoints2, mtx1, dist1, mtx2, dist2, (512, 512), criteria=criteria,
            flags=cv2.CALIB_FIX_INTRINSIC)
        print("error: {}".format(ret))

        ret, _, _, _, _, R_test, T_test, E_test, F_test = cv2.stereoCalibrate(
            objpoints, imgpoints2, imgpoints1, mtx2, dist2, mtx1, dist1, (512, 512), criteria=criteria,
            flags=cv2.CALIB_FIX_INTRINSIC)

        # reproject and check errors
        RT1 = np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1)
        RT2 = np.concatenate([R, T], axis = -1)
        # RT1 = np.concatenate([R, T], axis = -1)
        # RT2 = np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1)


        proj_mat1 = mtx1 @ RT1
        proj_mat2 = mtx2 @ RT2

        mean_error = 0
        for i in range(len(imgpoints1)):
            triangulated = cv2.triangulatePoints(proj_mat1, proj_mat2, imgpoints1[i], imgpoints2[i])
            cam1_ref_points = triangulated/triangulated[3, :]
            cam1_ref_points = cam1_ref_points[:3, :].T

            #imgpoints_reproj, _ = cv2.projectPoints(cam1_ref_points, R, T, mtx1, dist1)
            imgpoints_reproj, _ = cv2.projectPoints(cam1_ref_points, np.eye(3), np.zeros((3,1)), mtx1, dist1)
            error = cv2.norm(imgpoints1[i], imgpoints_reproj, cv2.NORM_L2)/len(imgpoints_reproj)
            mean_error += error
            write_stereo_reprojection(cap, imgpoints1[i], imgpoints_reproj, frame_idx[i], offsets[cam1_id], cam1_id, cam2_id)

            #imgpoints_reproj2, _ = cv2.projectPoints(cam1_ref_points, np.eye(3), np.zeros((3,1)), mtx2, dist2)
            imgpoints_reproj2, _ = cv2.projectPoints(cam1_ref_points, R, T, mtx2, dist2)

            # manual change of frame of reference to test...
            cam2_ref_points = (R @ cam1_ref_points.T + T).T
            test_points, _ = cv2.projectPoints(cam2_ref_points, np.eye(3), np.zeros((3,1)), mtx2, dist2)
            test_points = test_points.astype('float32')
            test_error = cv2.norm(test_points, imgpoints_reproj2, cv2.NORM_L2)/len(imgpoints_reproj)

            # write_stereo_reprojection(cap, imgpoints2[i], imgpoints_reproj2, frame_idx[i], offsets[cam2_id], cam2_id, cam1_id)

        print( "total error: {}".format(mean_error/len(objpoints)) )

        # write the data to a mat file.
        # need, R, T, square size, num squares, fc, cc and skew.
        # {'om' 'T' 'R' 'active_images_left' 'recompute_intrinsic_right'}
        om = cv2.Rodrigues(R)
        om = om[0]
        om_test = cv2.Rodrigues(R_test)
        om_test = om_test[0]
        #import pdb; pdb.set_trace()
        out_dict = {
            "calib_name_left": "cam_{}".format(cam2_id),
            "calib_name_right": "cam_{}".format(cam1_id),
            "dX": 3,
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
            "active_images_left": [],
            "cc_left_error": 0,
            "cc_right_error": 0,
            "recompute_intrinsic_right": 1,
            "recompute_intrinsic_left": 1
        }

        scipy.io.savemat("cam_{}{}_opencv.mat".format(cam1_id, cam2_id), out_dict)

    cap.release()





def main(argv):
    del argv

    with open(FLAGS.calib_frames, "rb") as fid:
        calib_frames = pickle.load(fid)

    with open(FLAGS.calibrated_name, "rb") as fid:
        calibration_data = pickle.load(fid)
    camera_calibs = calibration_data["calibrated"]

    # first collect overlapping frames from each pair of cameras    
    all_overlapping = get_all_overlapping_frames(calib_frames) 

    # calibrate each pair of cameras
    calibrate_all_camera_pairs(calib_frames, all_overlapping, camera_calibs)


if __name__ == "__main__":
    app.run(main)
