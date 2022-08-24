import numpy as np
import cv2
from absl import app
from absl import flags
import random


FLAGS = flags.FLAGS
flags.DEFINE_list("calib_movies", None, "List of calibration videos to merge.")
flags.DEFINE_string("out_video", None, "Merged output video name.")

flags.mark_flag_as_required("calib_movies")
flags.mark_flag_as_required("out_video")

def main(argv):
    del argv

    movie_list = FLAGS.calib_movies

    # get initial flags to setup the writer obj
    cap = cv2.VideoCapture(FLAGS.calib_movies[0])
    full_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    writer = cv2.VideoWriter(FLAGS.out_video, fourcc, fps, (full_width, height))

    for i in range(len(FLAGS.calib_movies)):
        print("copying movie: {}".format(FLAGS.calib_movies[i]))
        cap = cv2.VideoCapture(FLAGS.calib_movies[i])

        while cap.isOpened():
            ret, frame = cap.read()
            if ret is True:
                writer.write(frame)
            else:
                break

        cap.release()

    writer.release()


if __name__ == "__main__":
    app.run(main)