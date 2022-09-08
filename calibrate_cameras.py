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
from scipy.cluster.vq import kmeans,vq,whiten
import random
import os

FLAGS = flags.FLAGS
flags.DEFINE_string("calib_frames", None, "Calibration frames data.")
flags.DEFINE_string("calibrated_name", None, "Calibrated Camera Output File Name.")
flags.DEFINE_string("input_video", None, "Input video, used to make example outputs.")
flags.DEFINE_string("out_dir", None, "Output directory to save some debug outputs.")
flags.DEFINE_integer("num_frames", 100, "Number of frames to use for calibration.")


def draw_corner(image, corner, offset, color, markerSize):
    cv2.drawMarker(image, (int(corner[0])+offset, int(corner[1])), color,
        markerType=cv2.MARKER_CROSS, markerSize=markerSize)


def get_corners(frame, calib_frames, offsets, to_draw):
    all_corners = []

    # get each set of corner 0's 
    for i in range(len(calib_frames)):
        curr_calib = calib_frames[i]
        num_frames = len(curr_calib.frame_numbers)
        corners = np.zeros((num_frames, 2))
        for j in range(num_frames):
            corner = curr_calib.corners2[j][0].squeeze()
            corners[j, :] = corner
            if to_draw == True:
                draw_corner(frame, corner, offsets[i], (0, 255, 255, 0), 5)
        all_corners.append(corners)

    return all_corners


def cluster_corners(all_corners, num_clusters, seed):
    all_cluster_idx = []
    for i in range(len(all_corners)):
        centroids, _ = kmeans(all_corners[i], num_clusters, seed=seed)
        clx, _ = vq(all_corners[i], centroids)
        all_cluster_idx.append(clx)

    return all_cluster_idx


def draw_clusters(rng, frame, all_corners, all_cluster_idx, num_clusters, offsets):
    # create random colors
    colors = []
    for i in range(num_clusters):
        random_color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
        # random_color = (int(random_color[0]), int(random_color[1]), int(random_color[2]))
        colors.append(random_color)

    for i in range(len(all_corners)):
        corners = all_corners[i]
        cluster_idx = all_cluster_idx[i]
    
        for j in range(len(corners)):
            cluster_id = cluster_idx[j]
            draw_corner(frame, corners[j], offsets[i], colors[cluster_id], 5)


def sample_corners(rng, all_corners, all_cluster_idx, num_clusters):
    # for each set of corners, and each cluster, grab 1 points.
    num_samples = 10
    all_sampled = []
    all_sampled_idxs = []
    all_cluster_ids = []

    for i in range(len(all_corners)):
        # store the sampled points here, then concatentate
        sampled_corners =  []
        sampled_idxs = [] # indexing into the corners
        sampled_cluster_idx = [] # looks dumb, but useful for the drawing function
        current_cluster_id = all_cluster_idx[i]
        current_corners = all_corners[i]

        for j in range(num_clusters):
            # randomly choose a point in the cluster to be the sampled frame for
            # this cluster.
            clustered_indices = np.where(current_cluster_id == j)
            # np.where returns a tuple for me, i don't understand, because it looks like the
            # documentation says it outputs an array
            clustered_indices = clustered_indices[0]
            rng.shuffle(clustered_indices) # in place operation
        #     sampled_idxs.append(clustered_indices[:10]) # take the first sample

        #     # the sampled corner
        #     sampled_corners.append(current_corners[clustered_indices[0]])
        #     sampled_cluster_idx.append(j)

        # all_sampled.append(sampled_corners)
        # all_cluster_ids.append(sampled_cluster_idx)
        # all_sampled_idxs.append(sampled_idxs)
            sampled_idxs.append(clustered_indices[:num_samples])
            sampled = current_corners[clustered_indices[:num_samples]]
            sampled_corners.append(sampled)
            sampled_cluster_idx.append([j] * num_samples)

        all_sampled.append(np.concatenate(sampled_corners, axis=0))
        flat_cluster_ids = [x for xs in sampled_cluster_idx for x in xs]
        all_cluster_ids.append(flat_cluster_ids)
        flat_sampled_idx = [x for xs in sampled_idxs for x in xs]
        all_sampled_idxs.append(flat_sampled_idx)

    return all_sampled, all_sampled_idxs, all_cluster_ids


def index_list(main_list, index_list):
    temp = []
    for i in index_list:
        temp.append(main_list[i])
    return temp


def draw_corners(image, corners, color, markerSize):
    for i in range(len(corners)):
        cv2.drawMarker(image, (int(corners[i, 0]), int(corners[i, 1])), color,
            markerType=cv2.MARKER_CROSS, markerSize=markerSize)


def plot_reprojection(cap, outname, squares_xy, imgpoints, imgpoints2, frame_idx, offset=0):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
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
    frame = frame.copy()
    frame = frame[mins[1]:maxs[1], mins[0]+offset:maxs[0]+offset]
    # need to flip the corners
    imgpoints = imgpoints.copy()
    imgpoints2 = imgpoints2.copy()

    imgpoints[:, 0, 0] = imgpoints[:, 0, 0] - mins[0]
    imgpoints[:, 0, 1] = imgpoints[:, 0, 1] - mins[1]
    imgpoints2[:, 0, 0] = imgpoints2[:, 0, 0] - mins[0]
    imgpoints2[:, 0, 1] = imgpoints2[:, 0, 1] - mins[1]
    # cv2.drawChessboardCorners(frame, squares_xy, imgpoints, True)
    # draw_corner_numbers(frame_flipped, reordered)
    # # mark the frame number on a flipped example
    # cv2.putText(frame_flipped, "{}: {}".format(i, frame_num),
    #     (20, 20),
    #     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (209, 80, 0, 255), 1)
    corners = imgpoints.squeeze()
    #draw_corners(frame, corners, (255, 0, 255), 5)
    corners2 = imgpoints2.squeeze()
    #draw_corners(frame, corners, (0, 255, 255), 5)
    color_id = np.arange(corners.shape[0])

    plt.imshow(frame)
    plt.scatter(corners[:, 0], corners[:, 1], 12, c=color_id, cmap='cool', marker='x', linewidths=1)
    plt.scatter(corners2[:, 0], corners2[:, 1], 12, c=color_id, cmap='plasma', marker='+', linewidths=1)
    plt.savefig(outname)
    #plt.show()
    plt.close()

    # cv2.imshow("frame", frame)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    #cv2.imwrite("reprojections/{}.png".format(frame_idx), frame)


def main(argv):
    del argv

    with open(FLAGS.calib_frames, "rb") as fid:
        calib_frames = pickle.load(fid)

    if FLAGS.input_video is not None:
        cap = cv2.VideoCapture(FLAGS.input_video)
        full_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(full_width / 3)
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        fps = cap.get(cv2.CAP_PROP_FPS)

        ret, frame = cap.read()
        org_frame = frame.copy()

        to_draw = True
    else:
        frame = None
        to_draw = False

    if FLAGS.out_dir is not None:
        os.makedirs(FLAGS.out_dir, exist_ok=True)

    width = calib_frames[0].frame_size
    width = width[0]
    offsets = [0, width, 2 * width]
    all_corners = get_corners(frame, calib_frames, offsets, to_draw)

    if FLAGS.out_dir is not None:
        outname = FLAGS.out_dir + "/all_corners.png"
        cv2.imwrite(outname, frame)
    # cv2.imshow("moo", frame)
    # cv2.waitKey()

    # cluster the corners
    rng = np.random.RandomState(123)
    seed = 123
    # num_clusters = 10
    all_cluster_idx = cluster_corners(all_corners, 20, seed)
    
    # draw the clusters
    if FLAGS.out_dir is not None and FLAGS.input_video is not None:
        frame = org_frame.copy()
        draw_clusters(rng, frame, all_corners, all_cluster_idx, FLAGS.num_frames, offsets)
        outname = FLAGS.out_dir + "/clustered_corners.png"
        cv2.imwrite(outname, frame)
        # cv2.imshow("frame", frame)
        # cv2.waitKey()

    # for each cluster, sample 1 checker boards to use for calibration
    all_sampled, all_sampled_idxs, all_cluster_ids = sample_corners(rng, all_corners, all_cluster_idx, 20)

    # draw the sampled clusters
    if FLAGS.out_dir is not None and FLAGS.input_video is not None:
        frame = org_frame.copy()
        draw_clusters(rng, frame, all_sampled, all_cluster_ids, FLAGS.num_frames, offsets)
        outname = FLAGS.out_dir + "/clusters_sampled.png"
        cv2.imwrite(outname, frame)
        # cv2.imshow("frame", frame)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

    calibrated_cam_data = []
    # use the sampled points to create the camera calibration
    for i in range(len(calib_frames)):
        curr_frames = calib_frames[i]

        imgpoints = index_list(curr_frames.corners2, all_sampled_idxs[i])
        frame_idx = index_list(curr_frames.frame_numbers, all_sampled_idxs[i])
        objpoints = curr_frames.setup_obj_points()
        objpoints = objpoints[:len(imgpoints)]
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (512, 512), None, None)

        calibrated_cam_data.append({
            "mtx": mtx,
            "dist": dist,
            "rvecs": rvecs,
            "tvecs": tvecs
        })
        mean_error = 0
        mean_pixel_error = 0
        worst_error = 0
        worst_error_idx = 0
        all_reprojections = []
        for j in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[j], rvecs[j], tvecs[j], mtx, dist)
            all_reprojections.append(imgpoints2)
            error = cv2.norm(imgpoints[j], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            error_pixel = np.sqrt(np.sum(np.square(imgpoints[j].squeeze() - imgpoints2.squeeze()), axis=1)).sum() / len(imgpoints2)

            if error > worst_error:
                worst_error = error
                worst_error_idx = j
            mean_error += error
            mean_pixel_error += error_pixel
            if FLAGS.out_dir is not None and FLAGS.input_video is not None:
                # write the frames to disk
                outname = FLAGS.out_dir + "/cam_{}_error_{}_reproj_{}.png".format(i, error, frame_idx[j])
                plot_reprojection(cap, outname, curr_frames.squares_xy, imgpoints[j], imgpoints2, frame_idx[j], offsets[i])
 
        if FLAGS.out_dir is not None and FLAGS.input_video is not None:
            outname = FLAGS.out_dir + "/worst_cam_{}_error_{}_frame_{}.png".format(i, worst_error, frame_idx[worst_error_idx], offsets[i])
            plot_reprojection(cap, outname, curr_frames.squares_xy, imgpoints[worst_error_idx],
                all_reprojections[worst_error_idx], frame_idx[worst_error_idx], offsets[i])

        print( "total mean error: {}".format(mean_error/len(objpoints)) )
        print( "total pixel mean error: {}".format(mean_pixel_error/len(objpoints)) )

    all_calib_data = {
        "calibrated": calibrated_cam_data,
        "sampled_idx": all_sampled_idxs
    }

    with open(FLAGS.calibrated_name, "wb") as fid:
        pickle.dump(all_calib_data, fid)

    if FLAGS.input_video is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    app.run(main)
