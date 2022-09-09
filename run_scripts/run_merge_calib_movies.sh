CALIB_DIR=/workspace/calibration/calibration_videos

python merge_calibration_videos.py --out_video $CALIB_DIR/merged/calibration.avi --calib_movies $CALIB_DIR/cal_2021_11_10_15_40_32.mp4,$CALIB_DIR/cal2_2021_11_10_15_43_54.mp4,$CALIB_DIR/cal3_2021_11_10_15_46_14.mp4,$CALIB_DIR/cal5_2021_12_15_14_51_23.mp4,$CALIB_DIR/calibrate_2022_07_06_14_55_42.mp4
