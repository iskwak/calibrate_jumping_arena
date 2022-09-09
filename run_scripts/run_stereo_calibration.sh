OUTDIR=../calibration/20220725

python stereo_calibration.py --calib_frames ../calibration/flipped_frames.pkl --calibrated_name ../calibration/calibrated_cameras.pkl --input_video ../calibration/calibrate_2022_07_06_14_55_42.mp4 --out_dir $OUTDIR
#time python stereo_calibration.py --calib_frames $OUTDIR/flipped_frames.pkl --calibrated_name $OUTDIR/calibrated_cameras.pkl --input_video ../calibration/calibrate_2022_07_06_14_55_42.mp4 --out_dir $OUTDIR
