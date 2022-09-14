# script to run the full pipeline.
OUTDIR=/workspace/calibration/20220726_bigvideo_test
#mkdir $OUTDIR
INPUTVIDEO=/workspace/calibration/calibration_videos/merged/calibration.avi

#time python find_frames.py --video $INPUTVIDEO --outvideo $OUTDIR/calib_detect_video.avi --outpickle $OUTDIR/calib_frames.pkl

#time python flip_corners.py --calib_frames $OUTDIR/calib_frames.pkl --flipped_name $OUTDIR/flipped_frames.pkl --output_video $OUTDIR/flipped_movie.avi --input_video $INPUTVIDEO

#time python calibrate_cameras.py --calib_frames $OUTDIR/flipped_frames.pkl --calibrated_name $OUTDIR/calibrated_cameras.pkl --out_dir $OUTDIR/single_cam_calib --input_video $INPUTVIDEO --num_frames 150

time python stereo_calibration.py --calib_frames $OUTDIR/flipped_frames.pkl --calibrated_name $OUTDIR/calibrated_cameras.pkl --input_video $INPUTVIDEO --out_dir $OUTDIR --num_frames 150