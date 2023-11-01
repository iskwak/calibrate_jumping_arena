# Calibrating Jason's Hind Leg Extension Rig

Jason's Hind Leg Extension Rig currently has 3 cameras pointed to a head fixed mouse. The mouse is standing on its hind legs and on a platform that can be pushed down. The rig has three cameras that are pointed at the right side, left side, and center body of the mouse. This document describes the process for collecting calibration videos and using this repository for calibrating the rig. The 3 cameras, right, left, and center, will be refered as cameras 1, 2, and 3 respectively.

The goal of the calibration process is to estimate intrinsic and extrinsic parameters. The intrinsic parameters are the focal length, camera center, and lens distortion parameters. The extrinsic parameters are the rotation and translation vectors between cameras.

### More notes on the rig
#### Camera Configuration
The camera configuration has been difficult to calibrate properly. The right and left cameras are effectively pointing at each other, and have been difficult to compute extrinsic parameters. We have decided to calibration the right+center and left+center camera pairs, and then compute the extrinsic parameters from the right camera to the left camera (through the center camera).

At the time of writing this document, the right and center camera angle is under 90 degrees, and the left and center camera's angle is over 90 degrees. This has made it harder to collect videos where the target is visible to the center and left camera in all areas in the rig.

#### Video Format from Jason's Data Collection
Although the rig has 3 cameras, Jason's setup/code for collecting data will create a single video stream. Meaning, each camera's video stream will be concatenated together, producing a single video that is 3 camera frames wide. The calibration videos from the rig should be MJPEGs.

### Checkerboard
There are a variety of calibration targets available for calibrating a set of cameras. For this repo, it is assumed that we are using checkerboards to calibrate the targets. The most recent checkerboards are 5x4 squares.

## Calibration Steps
1) Collect calibration videos.
There are a few goals for the calibration videos.
* Ideally the checkerboard needs to be visible in the same 3D volume that the mouse will be at.
* The checkerboard should be seen in as many parts of the image as possible.
* The checkerboard will need to be visible by 2 cameras.
* The checkerboard is visible in a way that the corners of the checkerboard are easy to distinguish. If the target is shown at a glancing angle to a camera, it is difficult to accurately determine the corner locations.
* In order to avoid moving the checkerboard too fast and having blurry frames. It is advised to move the target slowly and stop moving the target periodically.
In these videos, we will leave the jumping platform attached to the rig. This helps the video collector estimate where the mouse will be during the experiments. The platform will not be fixed in place, so the collector can move the platform down (as if the mouse had extended its hind legs).

Originally the video calibration steps were to collect 5 videos, where each video had a specific purpose.
1) Video where the target is only visible to camera 1. This video will be used for calibrating the intrinsic parameters of camera 1.
* In this video, the target is moved along the height and width of the camera view frame, as well as the depth. The goal will be to have frames with the target in the full 3D volume of the rig platform, including when the platform is pushed down.
2) Video where the target is only visible to camera 2. This video will be used for calibrating the intrinsic parameters of camera 2. It will have the same collection guidelines as video 1.
3) Video where the target is only visible to camera 3. This video will be used for calibrating the intrinsic parameters of camera 2. It will have the same collection guidelines as video 1.
4) Video where the target is visible to camera 1 and camera 3. This video will be used for calibrating the extrinsic parameters between cameras 1 and 3.
* The target will need to be visible to cameras 1 and 3. So the target will need to be at an angle two both cameras.
* Like videos 1-3, the target will need to be visible to both cameras in the 3D volume that the mouse will be.
* As the target is moved around, then angle of the target will need to be adjusted because as the target is closer/further from either camera, the viewing angle can become too skewed.
5) Video where the target is visible to camera 1 and camera 3. This video will be used for calibrating the extrinsic parameters between cameras 2 and 3.
* This video is similar to video 4, but more difficult to collect. The angle between cameras 2 and 3 has been wide enough that it is difficult to collect video frames where the target is clearly visible from both cameras.

If these guidelines are followed properly, then there is no reason to record the video as 5 seperate videos. However past attempts, where only one video is collected has often missed parts of the guidelines described above. Ideally we'd like each video to be 2 or minutes long.

ADD EXAMPLE IMAGES OF THE TARGETS MOVEMENT

2) Detect corners.
This repository contains a python script called "findCheckerboards.py". This script will run opencv's findChessboardCorners function to find the corners of the checkerboard and save the corners in a pkl file. 

The parameters of this script can stored in a json file. Example json file:
```
{
    "base_dir": "/workspace/calibration/20230830_calibrationvideos",
    "calib_video": "raw/cal_2023_08_30_10_42_25.avi",
    "detected_corners": "detections/detect_0.pkl",
    "num_views": 3,
    "views": [0],
    "out_video_dir": "outputs",
    "squares_xy": [4,3],
    "square_mm": 5,
    "debug_image": true
}
```
Example command using the above config file:
```
python findCheckerboards.py --params /workspace/calibration/20230830_calibrationvideos/detect_0.json
```

Description of parameters:
* base_dir: the root of any relative paths for future parameters. For example, detected_corners is a realtive path and the full path would be <base_dir>/<detected_corners>
* calib_video: The calibration video to detect corners in.
* detected_corners: The file name to store the detected corners.
* num_views: The number of camera views
* views: A list of numbers, that are the views to detect corners.
* squares_xy: A list of numbers. This is the number of rows and columns of squares. NOTE: This is in opencv's format. If there are 5 column squares, but there are 4 detected corners along the column.
* square_mm: The real world dimensions for an individual squares edge.

(GET PICTURES OF SQUARES_XY SQUARE_MM THING)

3) Calibrate each individual camera.
This estimates the intrinsic parameters for a camaera.

Example parameters json file.
```
{
    "base_dir": "/workspace/calibration/20230830_calibrationvideos",
    "calib_video": "raw/cal_2023_08_30_10_42_25.avi",
    "detected_corners": "detections/detect_0.pkl",
    "calibration": ["calibrations/calibration_0.pkl"],
    "num_views": 3,
    "views": [0],
    "out_video_dir": "outputs",
    "squares_xy": [4,3],
    "square_mm": 5,
    "debug_image": true
}
```
Example command:
```
python calibrateCamera.py --params /workspace/calibration/20230830_calibrationvideos/calib_0.json
```

