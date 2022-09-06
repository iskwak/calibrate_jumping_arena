# the quality of the corner detector seems to get worse as the target is further
# from the camera. Probably due to blurring. Check this by using the tvecs output
# of the camera calibration. This will tell us how far the target was from the
# camera.

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
#import calibrate_cameras
import os
import utilities

FLAGS = flags.FLAGS
flags.DEFINE_string("frame_info", "../calibration/20220726_bigvideo_test/flipped_frames.pkl", "Calibration frames data.")
flags.DEFINE_string("calib_video", "/workspace/calibration/calibration_videos/merged/calibration.avi", "Calibrated Camera Output File Name.")
flags.DEFINE_string("camera_calibs", "../calibration/20220726_bigvideo_test/calibrated_cameras.pkl", "Calibrated camera data")
flags.DEFINE_string("out_dir", "/workspace/outputs/checkerboard_edge_check", "Output directory")


def create_save_targets(out_dir, cap, distances, means, stds, sampled_frame_nums, sampled_corners, offset):
    mean_edge_out_dir = out_dir + "/mean_edge_length_binned"
    os.makedirs(mean_edge_out_dir, exist_ok=True)
    counts, bins = np.histogram(means)
    bin_idxs = np.digitize(means, bins[1:], right=True)
    for i in range(len(bins) - 1):
        os.makedirs(mean_edge_out_dir + "/{}".format(i), exist_ok=True)

    for i in range(len(bin_idxs)):
        current_out_dir = mean_edge_out_dir + "/{}".format(bin_idxs[i])
        # sampled_frame_num = frame_nums[sampled_frame_idx[i]]
        # corners = calib_corners[sampled_frame_idx[i]]
        corners = sampled_corners[i]
        frame_num = sampled_frame_nums[i]
        mean_edge = means[i]
        std_edge = stds[i]
        dist_to_target = distances[i]

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret == True:
            corners = corners.squeeze()
            # draw_corner_numbers(frame, corners, offsets[1])
            utilities.draw_corners_with_gradient(frame, corners, (255, 0, 255), 5, offset)
            # cv2.imshow("moo", frame)
            # cv2.waitKey()
            cv2.imwrite(current_out_dir + "/frame_{}_dist_{}_mean_{}_std_{}.png".format(
                frame_num, dist_to_target, mean_edge, std_edge), frame)

    stds_edge_out_dir = out_dir + "/stds_edge_length_binned"
    os.makedirs(stds_edge_out_dir, exist_ok=True)
    counts, bins = np.histogram(stds)
    bin_idxs = np.digitize(stds, bins[1:], right=True)
    for i in range(len(bins) - 1):
        os.makedirs(stds_edge_out_dir + "/{}".format(i), exist_ok=True)

    for i in range(len(bin_idxs)):
        current_out_dir = stds_edge_out_dir + "/{}".format(bin_idxs[i])
        # sampled_frame_num = frame_nums[sampled_frame_idx[i]]
        # corners = calib_corners[sampled_frame_idx[i]]
        corners = sampled_corners[i]
        frame_num = sampled_frame_nums[i]
        mean_edge = means[i]
        std_edge = stds[i]
        dist_to_target = distances[i]

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret == True:
            corners = corners.squeeze()
            # draw_corner_numbers(frame, corners, offsets[1])
            utilities.draw_corners_with_gradient(frame, corners, (255, 0, 255), 5, offset)
            # cv2.imshow("moo", frame)
            # cv2.waitKey()
            cv2.imwrite(current_out_dir + "/frame_{}_dist_{}_mean_{}_std_{}.png".format(
                frame_num, dist_to_target, mean_edge, std_edge), frame)   


# def draw_corner_numbers(image, corners, offset):
#     num_corners = corners.shape[0]
#     corners = corners.squeeze()
#     color_step = 209 / num_corners

#     for i in range(num_corners):
#         cv2.putText(image, "{}".format(i),
#             (int(corners[i, 0] + offset), int(corners[i, 1])),
#             cv2.FONT_HERSHEY_SIMPLEX, 0.25, (i * color_step, 80, 0, 255), 1)


# def draw_corners_with_gradient(image, corners, color, markerSize, offset=0):
#     num_corners = corners.shape[0]
#     color_step = color[0] / num_corners
#     for i in range(len(corners)):
#         #current_color = color
#         current_color = (color_step * i, color[1], color[2])
#         cv2.drawMarker(image, (int(corners[i, 0] + offset), int(corners[i, 1])), current_color,
#             markerType=cv2.MARKER_CROSS, markerSize=markerSize)


def main(argv):
    with open(FLAGS.frame_info, "rb") as fid:
        calib_frames = pickle.load(fid)
    with open(FLAGS.camera_calibs, "rb") as fid:
        camera_calibs = pickle.load(fid)

    os.makedirs(FLAGS.out_dir, exist_ok=True)

    cap = cv2.VideoCapture(FLAGS.calib_video)
    if cap.isOpened() == False:
        exit()
    full_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(full_width / 3)
    offsets = [0, width, 2 * width] 


    for i_cam in range(len(calib_frames)):
        out_dir = FLAGS.out_dir + "/cam_{}".format(i_cam)
        os.makedirs(out_dir, exist_ok=True)
        # first test, create scatter plots of distance to camera and mean edge length of detected corners and the variance
        # of the corners in the target.
        # going to represent each of targets as a mean edge distance, and variance of the edge distance.
        frame_nums = calib_frames[i_cam].frame_numbers
        calib_corners = calib_frames[i_cam].corners2
        tvecs = camera_calibs["calibrated"][i_cam]["tvecs"]
        sampled_frame_idx = camera_calibs["sampled_idx"][i_cam]
        distances_to_target = np.zeros(len(sampled_frame_idx))
        mean_edge_lengths = np.zeros(len(sampled_frame_idx))
        std_edge_lengths = np.zeros(len(sampled_frame_idx))
        # max_edge_lengths = np.zeros(len(sampled_frame_idx))
        # min_edge_lengths = np.zeros(len(sampled_frame_idx))
        sampled_frame_nums = np.zeros(len(sampled_frame_idx))
        sampled_corners = []

        for i in range(len(sampled_frame_idx)):
            sampled_frame_num = frame_nums[sampled_frame_idx[i]]
            corners = calib_corners[sampled_frame_idx[i]]

            distances_to_target[i] = np.sqrt(np.sum(np.square(tvecs[i])))
            mean_edge_lengths[i], std_edge_lengths[i], _ = utilities.mean_std_corner_dists(corners)
            sampled_frame_nums[i] = sampled_frame_num
            sampled_corners.append(corners)

            # if distances_to_target[i] < 120 and std_edge_lengths[i] > 2.0:
            #     cap.set(cv2.CAP_PROP_POS_FRAMES, sampled_frame_num)
            #     ret, frame = cap.read()
            #     if ret == True:
            #         corners = corners.squeeze()
            #         draw_corner_numbers(frame, corners, offsets[1])
            #         draw_corners_with_gradient(frame, corners, (255, 0, 255), 5, offsets[1])
            #         cv2.imshow("moo", frame)
            #         cv2.waitKey()
        plt.scatter(distances_to_target, std_edge_lengths, s=5)
        out_name = (out_dir + "/distance_vs_std.png")
        plt.savefig(out_name)
        #plt.show()
        
        plt.clf()
        plt.scatter(distances_to_target, mean_edge_lengths, s=5)
        out_name = (out_dir + "/distance_vs_mean.png")
        plt.savefig(out_name)
        #plt.show()

        # next, create a plot of mean corner length vs the mean variance of the edge distance.
        plt.clf()
        plt.scatter(mean_edge_lengths, std_edge_lengths, s=5)
        out_name = (out_dir + "/mean_vs_std.png")
        plt.savefig(out_name)
        plt.close()

        create_save_targets(out_dir, cap, distances_to_target, mean_edge_lengths, std_edge_lengths,
            sampled_frame_nums, sampled_corners, offsets[i_cam])
    #plt.show()

    # thought, the close stuff with high variance also seem to be targets that are skewed. The variance of these targets
    # might not be due poor corner detection.

    # next lets save the targets organized by distance and/or the variance in
    # corner locations.
    # mean_edge_out_dir = FLAGS.out_dir + "/mean_edge_length_binned"
    # os.makedirs(mean_edge_out_dir, exist_ok=True)
    # counts, bins = np.histogram(mean_edge_lengths)
    # bin_idxs = np.digitize(mean_edge_lengths, bins[1:], right=True)
    # for i in range(len(bins) - 1):
    #     os.makedirs(mean_edge_out_dir + "/{}".format(i), exist_ok=True)
        
    # for i in range(len(bin_idxs)):
    #     current_out_dir = mean_edge_out_dir + "/{}".format(bin_idxs[i])
    #     sampled_frame_num = frame_nums[sampled_frame_idx[i]]
    #     corners = calib_corners[sampled_frame_idx[i]]
    #     mean_edge = mean_edge_lengths[i]
    #     std_edge = std_edge_lengths[i]
    #     dist_to_target = distances_to_target[i]

    #     cap.set(cv2.CAP_PROP_POS_FRAMES, sampled_frame_num)
    #     ret, frame = cap.read()
    #     if ret == True:
    #         corners = corners.squeeze()
    #         # draw_corner_numbers(frame, corners, offsets[1])
    #         draw_corners_with_gradient(frame, corners, (255, 0, 255), 5, offsets[1])
    #         # cv2.imshow("moo", frame)
    #         # cv2.waitKey()
    #         cv2.imwrite(current_out_dir + "/frame_{}_dist_{}_mean_{}_std_{}.png".format(
    #             sampled_frame_num, dist_to_target, mean_edge, std_edge), frame)

    #import pdb; pdb.set_trace()


if __name__ == "__main__":
    app.run(main)