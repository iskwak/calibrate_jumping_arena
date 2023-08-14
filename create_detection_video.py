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
flags.DEFINE_string("output_video", None, "Output video name")
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

    idxs = [0, 0, 0]
    to_loop = True
    frame_num = 0
    while True:
        for i in range(len(idxs)):
            if idxs[i] == len(calib_frames[i].frame_numbers):
                to_loop = False
                break
        if to_loop == False:
            break

        # probably a better way to do this, but need something quick.
        min_frame = min(calib_frames[0].frame_numbers[idxs[0]], calib_frames[1].frame_numbers[idxs[1]], calib_frames[2].frame_numbers[idxs[2]])

        #import pdb; pdb.set_trace()
        while frame_num < min_frame:
            _, _ = cap.read() 
            frame_num += 1
        #cap.set(cv2.CAP_PROP_POS_FRAMES, 390)
        print(min_frame)
        print(frame_num)
        ret, frame = cap.read()
        if ret == True:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)

            for i in range(len(calib_frames)):
                if calib_frames[i].frame_numbers[idxs[i]] == min_frame:
                    corners = calib_frames[i].corners2[idxs[i]]
                    plot_corners(ax, frame, corners, offset=offsets[i])
                    idxs[i] = idxs[i] + 1
            # plt.imshow(frame)
            plt.show()
            plt.close()
            import pdb; pdb.set_trace()


if __name__ == "__main__":
    app.run(main)
