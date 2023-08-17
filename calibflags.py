from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("calib_video", None, "Calibration Video")
flags.DEFINE_string("detected_corners", None, "Pickle for detected corners.")
flags.DEFINE_string("out_video", None, "output video")
flags.DEFINE_boolean("debug", False, "Detect corners in a subset of the frames, to speed up the process")

flags.DEFINE_string("calibrated_name", None, "Calibrated Camera Output File Name.")



