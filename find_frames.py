import numpy as np
import cv2
# import glob
from matplotlib import pyplot as plt
from absl import app
from absl import flags
import pickle
from calibrationdata import CalibrationFrames
import shared_flags
import utilities


FLAGS = flags.FLAGS
# flags.DEFINE_string("video", None, "Calibration Video")
flags.DEFINE_string("detection_video", None, "Video showing the detections")
# flags.DEFINE_string("outpickle", None, "Base pickle name for data.")
# flags.DEFINE_boolean("show_plot", False, "Shows plots of detections.")
# flags.DEFINE_boolean("debug", False, "Detect corners in a subset of the frames, to speed up the process")

flags.mark_flag_as_required("calib_video")
flags.mark_flag_as_required("detected_frames")



def find_corners(calib_frames, frame, gray, frame_num):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

    ret, corners = cv2.findChessboardCorners(gray, calib_frames.squares_xy, None)
    if ret == True:
        corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1,-1), criteria)
        # calib_frames.corners.append(corners)
        # calib_frames.corners2.append(corners2)
        # calib_frames.frame_number.append(frame_num)
        cv2.drawChessboardCorners(frame, calib_frames.squares_xy, corners2, ret)
        calib_frames.add_data(frame_num, corners, corners2)


def main(argv):
    del argv

    cap = cv2.VideoCapture(FLAGS.video)
    full_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(full_width / 3)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Arrays to store object points and image points from all the images.
    camera_calib_frames = [
        CalibrationFrames("left", FLAGS.video, (height, width)),
        CalibrationFrames("right", FLAGS.video, (height, width)),
        CalibrationFrames("center", FLAGS.video, (height, width))
    ]

    if FLAGS.detection_video is not None:
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        writer = cv2.VideoWriter(FLAGS.outvideo, fourcc, fps, (full_width, height))

    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if FLAGS.debug is True and (frame_num %  10) != 0:
            frame_num = frame_num + 1
            continue

        if ret == True:

            color_left = frame[:, 0:width, :]
            color_right = frame[:, width:2*width, :]
            color_center = frame[:, 2*width:, :]

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # split the image into 3 frames.
            gray_left = gray[:, 0:width]
            gray_right = gray[:, width:2*width]
            gray_center = gray[:, 2*width:]

            find_corners(camera_calib_frames[0], color_left, gray_left, frame_num)
            find_corners(camera_calib_frames[1], color_right, gray_right, frame_num)
            find_corners(camera_calib_frames[2], color_center, gray_center, frame_num)

            if FLAGS.outvideo is not None:
                full_frame = np.concatenate((color_left, color_right), axis=1)
                full_frame = np.concatenate((full_frame, color_center), axis=1)
                writer.write(full_frame)
                # if FLAGS.show_plot:
                #     cv2.imshow("Frame Left", color_left)
                #     cv2.imshow("Frame Right", color_right)
                #     cv2.imshow("Frame Center", color_center)

                #     if cv2.waitKey(1) & 0xFF == ord('q'):
                #         break
        else:
            break

        frame_num = frame_num + 1

    cap.release()
    cv2.destroyAllWindows()
    if FLAGS.outvideo is not None:
        writer.release()

    print("Num found frames: {}, {}, {}".format(
        len(camera_calib_frames[0].frame_numbers),
        len(camera_calib_frames[1].frame_numbers),
        len(camera_calib_frames[2].frame_numbers)))
    with open(FLAGS.outpickle, "wb") as fid:
        pickle.dump(camera_calib_frames, fid)


if __name__ == "__main__":
    app.run(main)
