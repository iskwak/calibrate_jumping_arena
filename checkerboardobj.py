import numpy as np
import cv2
# import glob
from matplotlib import pyplot as plt
from absl import app
from absl import flags
import pickle
from calibrationdata import CheckerboardDetectedFrames
import shared_flags
import utilities


FLAGS = flags.FLAGS
# flags.DEFINE_string("video", None, "Calibration Video")
flags.DEFINE_string("detection_video", None, "Video showing the detections")
# flags.DEFINE_string("outpickle", None, "Base pickle name for data.")
# flags.DEFINE_boolean("show_plot", False, "Shows plots of detections.")
flags.DEFINE_boolean("debug", False, "Detect corners in a subset of the frames, to speed up the process")
flags.DEFINE_string("test_file", None, "Filename for debug CheckerboardDetection obj pickle")

flags.mark_flag_as_required("calib_video")
flags.mark_flag_as_required("detected_frames")
flags.mark_flag_as_required("test_file")


def find_corners(calib_frames, frame, gray, frame_num):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

    ret, corners = cv2.findChessboardCorners(gray, calib_frames.checkerboard_dims, None)
    if ret == True:
        corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1,-1), criteria)
        # calib_frames.corners.append(corners)
        # calib_frames.corners2.append(corners2)
        # calib_frames.frame_number.append(frame_num)
        cv2.drawChessboardCorners(frame, calib_frames.checkerboard_dims, corners2, ret)
        calib_frames.add_data(frame_num, corners, corners2)


def main(argv):
    del argv

    cap = cv2.VideoCapture(FLAGS.calib_video)
    full_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(full_width / 3)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Arrays to store object points and image points from all the images.
    camera_calib_frames = [
        CheckerboardDetectedFrames("left", FLAGS.calib_video, (height, width)),
        CheckerboardDetectedFrames("right", FLAGS.calib_video, (height, width)),
        CheckerboardDetectedFrames("center", FLAGS.calib_video, (height, width))
    ]

    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if frame_num < 500:
            frame_num = frame_num + 1
            continue
        if frame_num >  1000:
            break
        if (frame_num % 10) != 0:
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

        else:
            break

        frame_num = frame_num + 1

    cap.release()
    cv2.destroyAllWindows()

    # test the serializtion here.

    serialized = camera_calib_frames[0].serialize_data()
    test_obj = CheckerboardDetectedFrames.from_data(serialized)

    assert(test_obj.camera_name == camera_calib_frames[0].camera_name)
    assert(test_obj.movie_name == camera_calib_frames[0].movie_name)
    assert(test_obj.frame_size == camera_calib_frames[0].frame_size)
    assert(test_obj.square_mm == camera_calib_frames[0].square_mm)
    assert(test_obj.checkerboard_dims == camera_calib_frames[0].checkerboard_dims)
    assert(len(test_obj.frame_numbers) == len(camera_calib_frames[0].frame_numbers))

    for i in range(len(test_obj.frame_numbers)):
        assert(test_obj.frame_numbers[i] == camera_calib_frames[0].frame_numbers[i])
        assert(np.all(test_obj.corners[i] == camera_calib_frames[0].corners[i]))
        assert(np.all(test_obj.corners2[i] == camera_calib_frames[0].corners2[i]))

    # next try saving and loading
    with open(FLAGS.test_file, "wb") as fid:
        #pickle.dump(vars(test_obj), fid)
        pickle.dump(test_obj.serialize_data(), fid)

    with open(FLAGS.test_file, "rb") as fid:
        reloaded_data = pickle.load(fid)

    reloaded_obj = CheckerboardDetectedFrames.from_data(reloaded_data)

    assert(test_obj.camera_name == reloaded_obj.camera_name)
    assert(test_obj.movie_name == reloaded_obj.movie_name)
    assert(test_obj.frame_size == reloaded_obj.frame_size)
    assert(test_obj.square_mm == reloaded_obj.square_mm)
    assert(test_obj.checkerboard_dims == reloaded_obj.checkerboard_dims)
    assert(len(test_obj.frame_numbers) == len(reloaded_obj.frame_numbers))

    for i in range(len(test_obj.frame_numbers)):
        assert(test_obj.frame_numbers[i] == reloaded_obj.frame_numbers[i])
        assert(np.all(test_obj.corners[i] == reloaded_obj.corners[i]))
        assert(np.all(test_obj.corners2[i] == reloaded_obj.corners2[i]))

    # the current usage model of the object is to fill it with all detections.
    # then flip the detections (update the calibratoin detections)
    #    for this one, lets just create a new object, and append to it.
    # filter the detections/sampel them.
    # run calibration on it
    # sample the detections again and run stereo calibration
    indexing = [0, 4, 6]
    moo = test_obj.filter_data(indexing)
    for i in range(len(indexing)):
        assert(moo.frame_numbers[i] == test_obj.frame_numbers[indexing[i]])
        assert(np.all(moo.corners[i] == test_obj.corners[indexing[i]]))
        assert(np.all(moo.corners2[i] == test_obj.corners2[indexing[i]]))

if __name__ == "__main__":
    app.run(main)
