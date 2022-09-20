#from absl import app
from absl import flags

flags.DEFINE_string("calib_video", None, "Calibration video")
flags.DEFINE_string("detected_frames", None, "Pickle for detected frames")
flags.DEFINE_string("flipped_frames", None, "Calibration frames after flipping corner detections")
flags.DEFINE_string("filtered_frames", None, "Calibration frames with small squares removed")
flags.DEFINE_string("single_cam_sampled_frames", None, "Sampled frames for single camera calibration")
flags.DEFINE_string("stereo_cam_sampled_frames", None, "Sampled frames for stereo camera calibration")
flags.DEFINE_string("output_dir", None, "Output directory")
