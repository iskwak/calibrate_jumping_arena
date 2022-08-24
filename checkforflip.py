# Helper script to make sure that the flipped points are properly detected.
import numpy as np
import cv2
# import glob
from matplotlib import pyplot as plt
from absl import app
from absl import flags
import pickle
from calibrationdata import CalibrationFrames
import time


def main(argv):
    del argv

    with open("../../calibration/calib_frames_small2.pkl", "rb") as fid:
        calib_frames = pickle.load(fid)

    video_name = "../../calibration/calibrate_2022_07_06_14_55_42.mp4"

    cap = cv2.VideoCapture(video_name)
    full_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(full_width / 3)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    writer = cv2.VideoWriter("../../calibration/test.avi", fourcc, fps, (full_width, height))

    while cap.isOpened():
        current_calib_frames = calib_frames[0]
        for i in range(len(current_calib_frames.corners2)):
            bottom = current_calib_frames.corners2[i][-1][0][1]
            top = current_calib_frames.corners2[i][0][0][1]
            if bottom > top:
                frame_num = current_calib_frames.frame_numbers[i]
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                res, frame = cap.read()
                # cv2.imshow("Original", frame)

                corners2 = current_calib_frames.corners2[i]
                cv2.drawChessboardCorners(frame, current_calib_frames.squares_xy, corners2, 1)
                # cv2.imshow("With Corners", frame)
                # if cv2.waitKey(-1) & 0xFF == ord('q'):
                #     break
                writer.write(frame)

    cap.release()
    cv2.destroyAllWindows()
    writer.release()


if __name__ == "__main__":
    app.run(main)