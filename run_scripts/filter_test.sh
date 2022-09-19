# script to run the full pipeline.
OUTDIR=/workspace/calibration/20220913_stereo_test
#mkdir $OUTDIR
INPUTVIDEO=/workspace/calibration/calibration_videos/merged/calibration.avi

#time python filter_checkerboard_detections.py --flipped_name /workspace/calibration/20220726_bigvideo_test/flipped_frames.pkl --filtered_name $OUTDIR/filtered_frames.pkl --calib_video $INPUTVIDEO --out_dir $OUTDIR/filtered_squares --threshold 9.5

#time python calibrate_cameras.py --calib_frames $OUTDIR/filtered_frames.pkl --calibrated_name $OUTDIR/calibrated_cameras.pkl --out_dir $OUTDIR/single_cam_calib --input_video $INPUTVIDEO --num_frames 150

time python stereo_calibration.py --calib_frames $OUTDIR/filtered_frames.pkl --calibrated_name $OUTDIR/calibrated_cameras.pkl --input_video $INPUTVIDEO --out_dir $OUTDIR --num_frames 150
#time python stereo_calibration.py --calib_frames /workspace/calibration/20220726_bigvideo_test/flipped_frames.pkl --calibrated_name $OUTDIR/calibrated_cameras.pkl --input_video $INPUTVIDEO --out_dir $OUTDIR --num_frames 150