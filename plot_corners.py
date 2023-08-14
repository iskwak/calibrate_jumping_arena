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
flags.DEFINE_string("calib_obj_pickle", None, "Pickle file with calibration flags.")
flags.DEFINE_string("name", None, "Name of the targets to plot")
flags.DEFINE_string("calibration_data", None, "Calibration parameters and sampling")


def main(argv):
    del argv

    with open(FLAGS.calib_obj_pickle, "rb") as fid:
        calib_frames = pickle.load(fid)

    if FLAGS.calibration_data is not None:
        with open(FLAGS.calibration_data, "rb") as fid:
            calibration_data = pickle.load(fid)
        sampled_idxs = calibration_data["sampled_idx"]
    else:
        sampled_idxs = []
        for i in range(len(calib_frames)):
            sampled_idxs.append(list(range(len(calib_frames[i].frame_numbers))))

    # setup output directory
    base_out_dir = FLAGS.out_dir + "/" + FLAGS.name
    os.makedirs(base_out_dir, exist_ok=True)

    # load the input video, grab one frame for plotting corners on top of.
    cap = cv2.VideoCapture(FLAGS.input_video)
    full_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(full_width / 3)
    #fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    #fps = cap.get(cv2.CAP_PROP_FPS)
    offsets = [0, width, 2 * width]
    _, frame = cap.read()

    for i in range(len(calib_frames)):
        frame_numbers = calib_frames[i].frame_numbers
        corners2 = np.squeeze(np.stack(calib_frames[i].corners2))
        offset = offsets[i]
        fig = plt.figure(figsize=(30, 10), dpi=100)
        plt.imshow(frame)
        for j in range(len(sampled_idxs[i])):
            sampled = sampled_idxs[i][j]
            # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_numbers[j])
            # _, frame = cap.read()
            plt.scatter(corners2[sampled, :, 0] + offset, corners2[sampled, :, 1], 12, marker='x', linewidths=1)
        outname = base_out_dir + "/detections_" + str(i) + ".png"
        plt.savefig(outname)
        #plt.show()
        plt.close(fig)


if __name__ == "__main__":
    app.run(main)
