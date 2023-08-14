import numpy as np
import cv2
from absl import app
from absl import flags
import pickle
from calibrationdata import CalibrationFrames
import calibrate_cameras
import os
from matplotlib import pyplot as plt
import time


FLAGS = flags.FLAGS
flags.DEFINE_string("overlapping_sampled", None, "Overlapping samples pickle file (used for stereo calib)")


def get_overlapping(calib_data, idx):
    # easier to make into a numpy array?
    corners = np.stack(calib_data.corners2)
    corners = corners[idx, :, :, :]

    # in order to make future slicing easier, keep this as a np array.
    #frame_numbers = np.asarray(calib_data.frame_numbers)[idx].tolist()
    frame_numbers = np.asarray(calib_data.frame_numbers)[idx]
    
    return corners, frame_numbers


def plot_corners_overlapping(cap, calib_frames, overlapping, offsets):
    out_dir = FLAGS.out_dir + "/review_frames/all_pairs/" + str(overlapping['view1']) + str(overlapping["view2"])
    os.makedirs(out_dir, exist_ok=True)
    overlapping_idx1 = overlapping['overlapping1']
    overlapping_idx2 = overlapping['overlapping2']
    calib_frames1 = calib_frames[overlapping['view1']]
    calib_frames2 = calib_frames[overlapping['view2']]
    offset1 = offsets[overlapping['view1']]
    offset2 = offsets[overlapping['view2']]

    [overlapping_corners1, overlapping_frames1] = get_overlapping(calib_frames1, overlapping_idx1)
    [overlapping_corners2, overlapping_frames2] = get_overlapping(calib_frames2, overlapping_idx2)

    # loop over the frames, and plot the corners to check the overlapping
    t = t = time.time()
    for i in range(len(overlapping_corners1)):
        print(i)
        # check the frame numbers
        assert(overlapping_frames1[i] == overlapping_frames2[i])
        cap.set(cv2.CAP_PROP_POS_FRAMES, overlapping_frames1[i])
        _, frame = cap.read()

        corners1 = np.squeeze(overlapping_corners1[i, :, :, :])
        corners2 = np.squeeze(overlapping_corners2[i, :, :, :])
        fig = plt.figure(figsize=(30, 10), dpi=100)
        plt.imshow(frame)
        plt.scatter(corners1[:, 0] + offset1, corners1[:, 1], 12, marker='x', linewidths=1)
        plt.scatter(corners2[:, 0] + offset2, corners2[:, 1], 12, marker='x', linewidths=1)

        outname = out_dir + "/" + str(overlapping_frames1[i]) + ".png"
        plt.savefig(outname)
        #plt.show()
        plt.close(fig)
        #plt.close()

    print(time.time() - t)


def main(argv):
    del argv

    with open(FLAGS.calib_frames, "rb") as fid:
        calib_frames = pickle.load(fid)

    with open(FLAGS.overlapping_sampled, "rb") as fid:
        overlapped_sampled = pickle.load(fid)

    # setup output directory
    base_out_dir = FLAGS.out_dir + "/review_frames"
    base_all_pairs = base_out_dir + "/all_pairs"
    os.makedirs(base_out_dir, exist_ok=True)

    # load the input video, grab one frame for plotting corners on top of.
    cap = cv2.VideoCapture(FLAGS.input_video)
    full_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(full_width / 3)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fps = cap.get(cv2.CAP_PROP_FPS)
    offsets = [0, width, 2 * width]
    ret, frame = cap.read()

    #cap.release()

    # plt.imshow(frame)
    # plt.show()
    # plt.close()
    # for each overlapping sampling, plot the checkerboard corners
    # will create 3 images (one for each camera pair).

    num_camera_pairs = len(overlapped_sampled['overlapped'])
    for i in range(num_camera_pairs):
        # plot corners on each frame
        # would make sense to do the filtering of frames out here and then pass into the plot
        # function. need the calib_frames to be able to filter more nicely. dont want to
        # filter and then pass around 4 things.
        #plot_corners_overlapping(cap, calib_frames, overlapped_sampled['overlapped'][i], offsets)
        offset1 = offsets[overlapped_sampled['overlapped'][i]['view1']]
        offset2 = offsets[overlapped_sampled['overlapped'][i]['view2']]
        # next get the sampled corners
        overlapping_idx1 = overlapped_sampled['overlapped'][i]['overlapping1']
        overlapping_idx2 = overlapped_sampled['overlapped'][i]['overlapping2']

        calib_frames1 = calib_frames[overlapped_sampled['overlapped'][i]['view1']]
        calib_frames2 = calib_frames[overlapped_sampled['overlapped'][i]['view2']]
        [overlapping_corners1, overlapping_frames1] = get_overlapping(calib_frames1, overlapping_idx1)
        [overlapping_corners2, overlapping_frames2] = get_overlapping(calib_frames2, overlapping_idx2)

        overlapping_corners1 = np.squeeze(overlapping_corners1)
        overlapping_corners2 = np.squeeze(overlapping_corners2)
        fig = plt.figure(figsize=(30, 10), dpi=100)
        plt.imshow(frame)
        for j in range(len(overlapping_frames1)):
            plt.scatter(overlapping_corners1[j, :, 0] + offset1, overlapping_corners1[j, :, 1], 12, marker='x', linewidths=1)
            plt.scatter(overlapping_corners2[j, :, 0] + offset2, overlapping_corners2[j, :, 1], 12, marker='x', linewidths=1)
        #plt.show()
        outname = base_out_dir + "/all_" + str(overlapped_sampled['overlapped'][i]['view1']) + str(overlapped_sampled['overlapped'][i]['view2'])
        plt.savefig(outname)
        plt.close(fig)

        sampled_corners1 = overlapping_corners1[overlapped_sampled['sampled_idx'][i], :, :]
        sampled_corners2 = overlapping_corners2[overlapped_sampled['sampled_idx'][i], :, :]
        sampled_frames = overlapping_frames1[overlapped_sampled['sampled_idx'][i]]

        # loop over the checkedboard locations in each frame and draw the corners
        # onto the movie frame. (add offsets for the corner locations)
        fig = plt.figure(figsize=(30, 10), dpi=100)
        plt.imshow(frame)
        for j in range(len(sampled_frames)):
            plt.scatter(sampled_corners1[j, :, 0] + offset1, sampled_corners1[j, :, 1], 12, marker='x', linewidths=1)
            plt.scatter(sampled_corners2[j, :, 0] + offset2, sampled_corners2[j, :, 1], 12, marker='x', linewidths=1)
        #plt.show()
        outname = base_out_dir + "/sampled_" + str(overlapped_sampled['overlapped'][i]['view1']) + str(overlapped_sampled['overlapped'][i]['view2'])
        plt.savefig(outname)
        plt.close(fig)



if __name__ == "__main__":
    app.run(main)
