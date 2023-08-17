# Remove corners that are too small, and are on the "backside" of the target.
# Seems the two targets are not aligned well, or the cameras have moved a lot
# since the calibration.
import numpy as np
import cv2
from matplotlib import pyplot as plt
from absl import app
from absl import flags
import pickle
import time
import os
import utilities
from cornerdata import CheckerboardCorners
import calibflags


FLAGS = flags.FLAGS
flags.DEFINE_string("filtered_corners", None, "name of the filtered checkerboards pickle")
flags.DEFINE_float("threshold", 8.0, "threshold for edge size")
flags.DEFINE_string("removed_corners", None, "name for removed corners")
flags.DEFINE_string("output_basename", None, "base output video name")
flags.adopt_module_key_flags(calibflags)

def write_corners(cap, frame_num, corners, offset):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()

    if ret == True:
        utilities.draw_corners_with_gradient(frame, corners, (255, 0, 255), 5, offset)

    return frame


def write_sample_examples(out_dir, cap, calib_frames, idx, offset, cam_num, num_sample=50):
    if len(idx) < num_sample:
        num_sample = len(idx)
    print(len(idx))

    corners2 = calib_frames.corners2
    frame_nums = calib_frames.frame_numbers
    for i in range(num_sample):
        frame_num = frame_nums[idx[i]]
        corners = corners2[idx[i]]

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

        if ret == True:
            plt.figure(figsize=(30, 10), dpi=100)
            plt.imshow(frame)
            corners = corners.squeeze()
            plt.plot(corners[:, 0]  + offset, corners[:, 1], 'rx')
            plt.savefig(out_dir + "/{}_frame_{}.png".format(cam_num, frame_num))
            #plt.show()
            plt.close()


def writeFilteredVideo(filteredIdx, 


def main(argv):
    del argv

    # go through all the corners, and get statistics on the square edge lengths.
    # for each camera, save screenshots of the targets, probably organized by the mean edge length.
    with open(FLAGS.detected_corners , "rb") as fid:
        cornerDict = pickle.load(fid) 
        cornerData = []
        for i in range(len(cornerDict)):
            cornerData.append(CheckerBoardCorners.fromDict(cornerDict[i]))

    # for each set of detections, prune the results. 
    numViews = len(cornerData)
    for i in range(numViews):
        # ignore flipped frames... they might be causing problems.
        corners2 = cornerData[i].corners2
        numCorners = corners2.shape[0]
        filteredIdx = np.ones((numCorners,), dtype=bool)

        for j in range(numCorners):
            # first, if the corners are flipped, then automatically filter it out.
            if corners2[j, -1, 0, 0] < corners2[j, 0, 0, 0]:
                filteredIdx[j] = False
            else:
                mean_edge, std_edge, edge_lengths = utilities.mean_std_corner_dists(corners2[j])
                if mean_edge <= FLAGS.threshold:
                    filteredIdx[j] = False

    # after filtering the detections, optionally create a video to see the
    # corners that were kept/removed.
    if FLAGS.output_basename is not None:
        writeVideo(
        
        cap = cv2.VideoCapture(FLAGS.calib_video)
        fullWidth  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(fullWidth / numViews)
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        fps = cap.get(cv2.CAP_PROP_FPS)
        #offsets = [0, width, 2 * width]
        
        keptWriter = cv2.VideoWriter(FLAGS.output_basename, fourcc, fps, (fullWidth, height))




    # create a debug visualization of all the top left corners



if __name__ == "__main__":
    app.run(main)