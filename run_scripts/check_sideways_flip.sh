# script to run the full pipeline.
OUTDIR=/workspace/calibration/check_sideways
mkdir $OUTDIR

INPUTVIDEO=/workspace/calibration/calibration_videos/cal4_2021_12_15_14_49_19.mp4

time python find_frames.py --video $INPUTVIDEO --outvideo $OUTDIR/calib_detect_video.avi --outpickle $OUTDIR/calib_frames.pkl

time python flip_corners.py --calib_frames $OUTDIR/calib_frames.pkl --flipped_name $OUTDIR/flipped_frames.pkl --output_video $OUTDIR/flipped_movie.avi --input_video $INPUTVIDEO

# time python calibrate_cameras.py --calib_frames $OUTDIR/flipped_frames.pkl --calibrated_name $OUTDIR/calibrated_cameras.pkl --out_dir $OUTDIR/single_cam_calib --input_video ../calibration/calibrate_2022_07_06_14_55_42.mp4 --num_frames 100

# time python stereo_calibration.py --calib_frames $OUTDIR/flipped_frames.pkl --calibrated_name $OUTDIR/calibrated_cameras.pkl --input_video ../calibration/calibrate_2022_07_06_14_55_42.mp4 --out_dir $OUTDIR
