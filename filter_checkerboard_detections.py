import numpy as np
import cv2
from matplotlib import pyplot as plt
from absl import app
from absl import flags
import pickle
from calibrationdata import CheckerboardDetectedFrames
import os
import utilities
import shared_flags

FLAGS = flags.FLAGS
# flags.DEFINE_string("filtered_name", None, "name of the filtered checkerboards pickle")
# flags.DEFINE_string("flipped_name", None, "name of the flipped corners pickle")
# flags.DEFINE_string("calib_video", None, "name of the calibration video")
# flags.DEFINE_string("out_dir", None, "optional, output directory for target images")
flags.DEFINE_float("threshold", 8.0, "threshold for edge size")

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


def main(argv):
    # this feels dumb, argv isn't used, but needed for the FLAGS stuff to work right.
    del argv

    # go through all the corners, and get statistics on the square edge lengths.
    # for each camera, save screenshots of the targets, probably organized by the mean edge length.
    cap = cv2.VideoCapture(FLAGS.calib_video)
    full_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(full_width / 3)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fps = cap.get(cv2.CAP_PROP_FPS)
    offsets = [0, width, 2 * width]

    with open(FLAGS.flipped_frames, "rb") as fid:
        calib_data = pickle.load(fid)
        calib_frames = []
        for i in range(len(calib_data)):
            calib_frames.append(CheckerboardDetectedFrames.from_data(calib_data[i]))

    rng = np.random.RandomState(123)

    if FLAGS.output_dir is not None:
        os.makedirs(FLAGS.output_dir, exist_ok=True)
        big_square_out = FLAGS.output_dir + "/big_squares"
        small_square_out = FLAGS.output_dir + "/small_squares"
        os.makedirs(big_square_out, exist_ok=True)
        os.makedirs(small_square_out, exist_ok=True)

    # may want to different types of binning and saving, but first one, lets do two bins, greater than 10mm and less
    # than.
    filtered_frames = []
    for i in range(len(calib_frames)):
        print("camera {}".format(i))
        corners = calib_frames[i].corners
        corners2 = calib_frames[i].corners2
        frame_nums = calib_frames[i].frame_numbers

        filtered_frames.append(CheckerboardDetectedFrames("camera {}".format(i), FLAGS.calib_video, (height, width)))
        big_square_idx = []
        small_square_idx = []
        for j in range(len(corners)):
            mean_edge, std_edge, edge_lengths = utilities.mean_std_corner_dists(corners2[j])
            frame_num = frame_nums[j]

            if mean_edge > FLAGS.threshold:
                filtered_frames[i].add_data(frame_num, corners[j], corners2[j])
                big_square_idx.append(j)
            else:
                small_square_idx.append(j)


        # after going through all the corners for this camera. sample the big and small squares to get an idea of what
        # was filtered. Go for 10% or a max of 100 frames, whichever is smaller.
        if FLAGS.output_dir is not None:
            rng.shuffle(big_square_idx) # shuffle is an in place operation
            rng.shuffle(small_square_idx)

            write_sample_examples(big_square_out, cap, calib_frames[i], big_square_idx, offsets[i], i)
            write_sample_examples(small_square_out, cap, calib_frames[i], small_square_idx, offsets[i], i)
            # frame = write_corners(cap, frame_nums[j], corners[j].squeeze(), offsets[i])
            # cv2.imshow("moo", frame)
            # cv2.waitKey()
            # cv2.destroyAllWindows()

    with open(FLAGS.filtered_frames, "wb") as fid:
        calib_data = []
        for i in range(len(filtered_frames)):
            calib_data.append(filtered_frames[i].serialize_data())
        pickle.dump(calib_data, fid)


if __name__ == "__main__":
    app.run(main)