# script to run the full pipeline.
OUTDIR=/workspace/outputs/test_calibration/
# mkdir $OUTDIR
INPUTVIDEO=/workspace/calibration/calibration_videos/merged/calibration.avi

#time python find_frames.py --calib_video $INPUTVIDEO --detection_video $OUTDIR/calib_detect_video.avi --detected_frames $OUTDIR/calib_frames.pkl

time python flip_corners.py --detected_frames $OUTDIR/data_dicts.pkl --flipped_frames $OUTDIR/flipped_frames.pkl --flipped_video $OUTDIR/flipped_movie.avi --calib_video $INPUTVIDEO

#time python filter_checkerboard_detections.py --flipped_name $OUTDIR/flipped_frames.pkl --filtered_name $OUTDIR/filtered_frames.pkl --calib_video $INPUTVIDEO --out_dir $OUTDIR/filtered_squares --threshold 9.5

#time python calibrate_cameras.py --calib_frames $OUTDIR/filtered_frames.pkl --calibrated_name $OUTDIR/calibrated_cameras.pkl --out_dir $OUTDIR/single_cam_calib --input_video $INPUTVIDEO --num_frames 150

#time python stereo_calibration.py --calib_frames $OUTDIR/flipped_frames.pkl --calibrated_name $OUTDIR/calibrated_cameras.pkl --input_video $INPUTVIDEO --out_dir $OUTDIR --num_frames 150
