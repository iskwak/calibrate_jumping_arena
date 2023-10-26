# calibrate_jumping_arena

Calibrating Jason's 3 camera rig setup.

Given a set of checkerboard calibration videos, the goal is to create a CalRigNPairwiseCalibrated calibration object for APT. The python code will use Opencv for calibrating the intrinsic parameters for cameras and extrinsic parameters for stereo. This code base does not do any bundle adjustment for N>2 camera calibration. This code base should be used in conjunction with SOMEMATLABCODEBASEGETLINKASAP for finalizing the multi camera calibration. 

NOTE: There will be off by one naming convention issues. In python the camera id's and camera indicies will match (cameras: 0, 1, 2). However, this naming convention will be off-by-one when using matlab.

## Steps
1) Collect calibration target videos from the rig.
* The code base assumes checkerboards and it is recommeneded to use checkerboards where number of squares in a row is different from the number of squares in a column.
* This code base assumes that the video streams are concatenated into a single video stream. All of the data I have recieved from Jason has been in this format.
* Make sure that the video streams are in the following order: right camera, left camera, center camera.
* At the moment, it was easier to calibrate and collect the data if each video had a specific purpose.
** For example, calibrating the intrinsic parameters of a single camera is one video. Then another video for calibrating the stereo extrinsinc parameters for a pair of cameras.
2) Detect checkerboard c



## Example Configuration Files
Example contents of detect_0.json
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

Example contents of detect_02.json
```
{
    "base_dir": "/workspace/calibration/20230830_calibrationvideos",
    "calib_video": "raw/cal_2023_08_30_10_49_34.avi",
    "detected_corners": "detections/detect_02.pkl",
    "num_views": 3,
    "views": [0,2],
    "out_video_dir": "outputs",
    "squares_xy": [4,3],
    "square_mm": 5,
    "debug_image": true,
    "group_detect": true
}
```

## Calibration Target Recommendations
I have found bits of anecdotal recommendations for the checkerboard size and dimensions. I also have some thoughts based of my experiences. It is unclear how fixed these recommendations are.

Goals of the checkerboard:
* Ideally it is easy to detect the corners of the checkerboard (either clicking by hand or automatically with opencv's checkerboard corner detector).
* From my experience, it is much better if the target locations fill the volume of space the mouse will be. For example, if the mouse is 20mm from one camera, 30mm from another, it helps if a checkerboard is detected at that location.
* The larger the target is, the more space the target will cover. However it makes it difficult to position the target when there are potentially rig components in the way.

For this rig, the checkerboard we used:
* 5x4 square target with 5mm edges.
* 


## Issues/Weaknesses
* I was in the process of adding json config file support for calibrating the rig. I was hoping to have a single monolithic json file for calibrating the rig (almost like a calibration project json file.). However this isn't the current case. Currently it is more like a json file for calibrating a camera or a pair of cameras. Additionally, the json file and command line calibration parameters do not match. I was rushing and fell behind on a few calibraiton parameter verificaiton steps.

Example calibration process in /groups/branson/bransonlab/kwaki/ForceData/calibration/20230830_calibrationvideos
```
-rw-r--r-- 1 kwaki branson 346 Sep 21 12:48 detect_02.json
-rw-r--r-- 1 kwaki branson 317 Sep 21 12:48 detect_0.json
-rw-r--r-- 1 kwaki branson 320 Sep 21 12:49 detect_12.json
-rw-r--r-- 1 kwaki branson 317 Sep 21 12:48 detect_1.json
-rw-r--r-- 1 kwaki branson 317 Sep 21 12:48 detect_2.json
```
