import numpy as np
import cv2
# import glob
from matplotlib import pyplot as plt
from absl import app
from absl import flags
import calibrationdata
from calibrationdata import CheckerboardDetectedFrames
#import time
#import shared_flags
import utilities

FLAGS = flags.FLAGS
flags.DEFINE_string("out_video", None, "Output video name")
flags.DEFINE_boolean("crop", False, "Crop the output video, can make it easier to follow.")
flags.mark_flag_as_required("detected_frames")
flags.mark_flag_as_required("flipped_frames")


def main(argv):
    del argv

    checkerboard_frames = calibrationdata.load_calibs(FLAGS.detected_frames)

    cap = cv2.VideoCapture(FLAGS.calib_video)
    full_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(full_width / 3)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fps = cap.get(cv2.CAP_PROP_FPS)
    offsets = [0, width, 2 * width]

    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    if FLAGS.crop == True:
        writer = cv2.VideoWriter(FLAGS.flipped_video, fourcc, fps, (180, 2 * 180))
    else:
        writer = cv2.VideoWriter(FLAGS.flipped_video, fourcc, fps, (full_width, 2 * height))

    # for each checkerboard object get each detection frame and plot it
    for i in range(len(checkerboard_frames)):
        frame_numbers = checkerboard_frames[i].frame_numbers
        corners2 = checkerboard_frames[i].corners2
        for j in range(len(frame_numbers)):
            frame_num = frame_numbers[j]
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if ret == True:
                if FLAGS.crop == True:
                    # crop the frame to make the video writing a bit less hard to parse.
                    mins = corners2.min(axis=0).squeeze().astype('int')
                    maxs = corners2.max(axis=0).squeeze().astype('int')

                    mins = mins - 60
                    if mins[0] < 0:
                        mins[0] = 0
                    if mins[1] < 0:
                        mins[1] = 0
                    maxs = mins + 180
                    # cropped = frame[mins[0]:maxs[0], mins[1]:maxs[1]]

                    # adjust the corners
                    frame = frame[mins[1]:maxs[1], mins[0]:maxs[0]]

                # need to flip the corners
                frame_flipped = frame.copy()

                if FLAGS.crop == True:
                    corners2[:, 0, 0] = corners2[:, 0, 0] - mins[0]
                    corners2[:, 0, 1] = corners2[:, 0, 1] - mins[1]
                #cv2.drawChessboardCorners(frame, cur_calib.squares_xy, corners2, True)
                #draw_corner_numbers(frame, corners2, offsets[j])
                utilities.plot_write_cropped_corners(frame, 
                full_frame = np.concatenate((frame, frame_flipped), axis=0)
                writer.write(full_frame)


if __name__ == "__main__":
    app.run(main)
