import numpy as np
import cv2
# import glob
from matplotlib import pyplot as plt
from absl import app
from absl import flags
import pickle
from calibrationdata import CalibrationFrames
import time

FLAGS = flags.FLAGS
flags.DEFINE_string("calib_frames", None, "Filaname for calibration frame pickle.")
flags.DEFINE_string("flipped_name", None, "Filename for fixed corner order.")
flags.DEFINE_string("output_video", None, "Filename for video showing flipped corner order. If blank, no video will be written")
flags.DEFINE_string("input_video", None, "Filename for original data video. If blank, no video will be written")
flags.DEFINE_boolean("crop", False, "Crop the test video, to make it easier to see the detected corners.")

flags.mark_flag_as_required("calib_frames")
flags.mark_flag_as_required("flipped_name")


def reorder_corners(corners, square_rows, square_cols):
    reordered = np.zeros(corners.shape, dtype=corners.dtype)
    num_rows = reordered.shape[0]
    # the corners are in row major order. The order of the rows need to be flipped,
    # but the corners are fine.
    for i in range(num_rows):
        # convert the index into (row,col) format and then flip it.
        row = int(i / square_cols)
        col = i % square_cols
        #new_col = square_cols - col - 1
        new_col = col
        new_row = square_rows - row - 1
        new_idx = new_row * square_cols + new_col
        reordered[new_idx, :, :] = corners[i, :, :]
    return reordered


def draw_corner_numbers(image, corners, offset):
    num_corners = corners.shape[0]
    corners = corners.squeeze()
    for i in range(num_corners):
        cv2.putText(image, "{}".format(i),
            (int(corners[i, 0] + offset), int(corners[i, 1])),
            cv2.FONT_HERSHEY_SIMPLEX, 0.25, (209, 80, 0, 255), 1)

def main(argv):
    del argv

    with open(FLAGS.calib_frames, "rb") as fid:
        calib_frames = pickle.load(fid)

    write_video = True
    if FLAGS.output_video is None or FLAGS.input_video is None:
        write_video = False

    if write_video == True:
        cap = cv2.VideoCapture(FLAGS.input_video)
        full_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(full_width / 3)
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        fps = cap.get(cv2.CAP_PROP_FPS)
        offsets = [0, width, 2 * width]

        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        if FLAGS.crop == True:
            writer = cv2.VideoWriter(FLAGS.output_video, fourcc, fps, (180, 2 * 180))
        else:
            writer = cv2.VideoWriter(FLAGS.output_video, fourcc, fps, (full_width, 2 * height))


    # width, height
    (num_cols, num_rows) = calib_frames[0].squares_xy
    # go through the corners and flip them.
    # create an image to plot on.
    # for j in range(3):
    for j in range(len(calib_frames)):
        #if j > 1 and write_video == True:
        #    break
        cur_calib = calib_frames[j]
        for i in range(len(cur_calib.frame_numbers)):

            frame_num = cur_calib.frame_numbers[i]
            corners2 = cur_calib.corners2[i]
            reorder_needed = False
            if corners2[-1, 0, 0] < corners2[0, 0, 0]:
                reorder_needed = True
                # reorder the corners
                reordered = reorder_corners(corners2, num_rows, num_cols)
                # update the calibration corners
                cur_calib.corners2[i] = reordered

            if write_video == True:
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
                    if reorder_needed == True:
                        reordered = reorder_corners(corners2, num_rows, num_cols)
                        #cv2.drawChessboardCorners(frame_flipped, cur_calib.squares_xy, reordered, True)
                        draw_corner_numbers(frame_flipped, reordered, offsets[j])
                        # mark the frame number on a flipped example
                        cv2.putText(frame_flipped, "{}: {}".format(i, frame_num),
                            (20, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (209, 80, 0, 255), 1)

                    #cv2.drawChessboardCorners(frame, cur_calib.squares_xy, corners2, True)
                    draw_corner_numbers(frame, corners2, offsets[j])
                    full_frame = np.concatenate((frame, frame_flipped), axis=0)
                    writer.write(full_frame)
    if write_video == True:
        writer.release()
        cap.release()
    
    # write the new corners to disk
    with open(FLAGS.flipped_name, 'wb') as flipped_pickle:
        pickle.dump(calib_frames, flipped_pickle)

if __name__ == "__main__":
    app.run(main)
