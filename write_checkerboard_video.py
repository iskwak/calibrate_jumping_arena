import numpy as np
import cv2
# import glob
from matplotlib import pyplot as plt
from absl import app
from absl import flags
import pickle
from calibrationdata import CalibrationFrames
import os


FLAGS = flags.FLAGS
flags.DEFINE_string("input_video", None, "Calibration Video")
flags.DEFINE_string("output_dir", None, "Output directory name")
flags.DEFINE_string("frame_pickle", None, "Base pickle name for data.")
flags.DEFINE_boolean("debug", False, "Detect corners in a subset of the frames, to speed up the process")


def plot_corners(ax, frame, corners, offset=0):

    corners = corners.copy().squeeze()
    corners[:, 0] = corners[:, 0] + offset
    color_id = np.arange(corners.shape[0])

    ax.imshow(frame)
    ax.scatter(corners[:, 0], corners[:, 1], 12, c=color_id, cmap='cool', marker='x', linewidths=1)
    # fig.show()
    # fig.close()


def main(argv):
    del argv

    with open(FLAGS.frame_pickle, "rb") as fid:
        calib_frames = pickle.load(fid)

    cap = cv2.VideoCapture(FLAGS.input_video)
    full_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(full_width / 3)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fps = cap.get(cv2.CAP_PROP_FPS)
    offsets = [0, width, 2 * width]

    writers = []
    for i in range(3):
        outname = "{}/{}.avi".format(FLAGS.output_dir, i)
        writers.append(cv2.VideoWriter(outname, fourcc, fps, (width, height)))

    # order, 4 frames from 01, 02, 12
    # start with 2 bad ones, then 2 goods ones
    hand_picked_frames = [
        755, 18847, 14077, 18838,
        24425, 5712, 19324, 10614,
        11268, 3641, 21897, 23803
    ]
    for i in range(len(hand_picked_frames)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, hand_picked_frames[i])
        ret, frame = cap.read()
        if ret == True:
            for i in range(3):
                writers[i].write(frame[:, offsets[i]:offsets[i]+512, :])


    idxs = [0, 0, 0]
    to_loop = True
    while True:
        for i in range(len(idxs)):
            if idxs[i] == len(calib_frames[i].frame_numbers):
                to_loop = False
                break
        if to_loop == False:
            break

        if idxs[0] > 10 and idxs[1] > 10 and idxs[2] > 10:
            break

        # probably a better way to do this, but need something quick.
        min_frame = min(calib_frames[0].frame_numbers[idxs[0]], calib_frames[1].frame_numbers[idxs[1]], calib_frames[2].frame_numbers[idxs[2]])

        cap.set(cv2.CAP_PROP_POS_FRAMES, min_frame)
        ret, frame = cap.read()
        if ret == True:
            # fig = plt.figure()
            # ax = fig.add_subplot(1, 1, 1)

            for i in range(len(calib_frames)):
                if calib_frames[i].frame_numbers[idxs[i]] == min_frame:
                    idxs[i] = idxs[i] + 1
            # # plt.imshow(frame)
            # plt.show()
            # plt.close()
            for i in range(3):
                writers[i].write(frame[:, offsets[i]:offsets[i]+512, :])
    for i in range(3):
        writers[i].release()


if __name__ == "__main__":
    app.run(main)
