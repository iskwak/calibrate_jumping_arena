# script to run the full pipeline.
OUTDIR=../calibration/20220726
mkdir $OUTDIR


time python find_frames.py --video ../calibration/calibrate_2022_07_06_14_55_42.mp4 --outvideo $OUTDIR/calib_detect_video.avi --outpickle $OUTDIR/calib_frames.pkl

time python flip_corners.py --calib_frames $OUTDIR/calib_frames.pkl --flipped_name $OUTDIR/flipped_frames.pkl

time python calibrate_cameras.py --calib_frames $OUTDIR/flipped_frames.pkl --calibrated_name $OUTDIR/calibrated_cameras.pkl --out_dir $OUTDIR/single_cam_calib --input_video ../calibration/calibrate_2022_07_06_14_55_42.mp4 --num_frames 100

time python stereo_calibration.py --calib_frames $OUTDIR/flipped_frames.pkl --calibrated_name $OUTDIR/calibrated_cameras.pkl --input_video ../calibration/calibrate_2022_07_06_14_55_42.mp4 --out_dir $OUTDIR
