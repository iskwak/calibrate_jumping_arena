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


def main(argv):
    del argv

    with open(FLAGS.calib_frames, "rb") as fid:
        calib_frames = pickle.load(fid)

    with open(FLAGS.overlapping_sampled, "rb") as fid:
        overlapped_sampled = pickle.load(fid)

    # setup output directory
    base_out_dir = FLAGS.out_dir + "/export_corners"
    os.makedirs(base_out_dir, exist_ok=True)

    # # load the input video, grab one frame for plotting corners on top of.
    # cap = cv2.VideoCapture(FLAGS.input_video)
    # full_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # width = int(full_width / 3)
    # fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # offsets = [0, width, 2 * width]
    # ret, frame = cap.read()

    #cap.release()
    views1 = []
    views2 = []
    corners1 = []
    corners2 = []
    frames = []
    num_camera_pairs = len(overlapped_sampled['overlapped'])
    for i in range(num_camera_pairs):
        # next get the sampled corners
        overlapping_idx1 = overlapped_sampled['overlapped'][i]['overlapping1']
        overlapping_idx2 = overlapped_sampled['overlapped'][i]['overlapping2']

        calib_frames1 = calib_frames[overlapped_sampled['overlapped'][i]['view1']]
        calib_frames2 = calib_frames[overlapped_sampled['overlapped'][i]['view2']]
        [overlapping_corners1, overlapping_frames1] = plot_all_sampled_overlapping.get_overlapping(calib_frames1, overlapping_idx1)
        [overlapping_corners2, _] = plot_all_sampled_overlapping.get_overlapping(calib_frames2, overlapping_idx2)

        overlapping_corners1 = np.squeeze(overlapping_corners1)
        overlapping_corners2 = np.squeeze(overlapping_corners2)

        sampled_corners1 = overlapping_corners1[overlapped_sampled['sampled_idx'][i], :, :]
        sampled_corners2 = overlapping_corners2[overlapped_sampled['sampled_idx'][i], :, :]
        sampled_frames = overlapping_frames1[overlapped_sampled['sampled_idx'][i]]

        views1.append(overlapped_sampled['overlapped'][i]['view1']) 
        views2.append(overlapped_sampled['overlapped'][i]['view2'])
        corners1.append(sampled_corners1)
        corners2.append(sampled_corners2)
        frames.append(sampled_frames)
        # corner_data.append({
        #     'view1': overlapped_sampled['overlapped'][i]['view1'],
        #     'view2': overlapped_sampled['overlapped'][i]['view2'],
        #     'corners1': sampled_corners1,
        #     'corners2': sampled_corners2
        # })
    corner_data = {
        "views1": views1,
        "views2": views2,
        "corners1": corners1,
        "corners2": corners2,
        "frames": frames
    }
    scipy.io.savemat("{}/sampled_stereo_corners.mat".format(base_out_dir), corner_data)
    # doesn't seem like a list can be saved to savemat.
    #scipy.io.savemat("{}/sampled_stereo_corners.mat".format(base_out_dir), {'data': corner_data})


if __name__ == "__main__":
    app.run(main)
