# Calibrating Jason's Hind Leg Extension Rig

Jason's Hind Leg Extension Rig currently has 3 cameras pointed to a head fixed mouse. The mouse is standing on its hind legs and on a platform that can be pushed down. The rig has three cameras that are pointed at the right side, left side, and center body of the mouse. This document describes the process for collecting calibration videos and using this repository for calibrating the rig.

The goal of the calibration process is to estimate intrinsic and extrinsic parameters. The intrinsic parameters are the focal length, camera center, and lens distortion parameters. The extrinsic parameters are the rotation and translation vectors between cameras.

### More notes on the rig
#### Camera Configuration
The camera configuration has been difficult to calibrate properly. The right and left cameras are effectively pointing at each other, and have been difficult to compute extrinsic parameters. We have decided to calibration the right+center and left+center camera pairs, and then compute the extrinsic parameters from the right camera to the left camera (through the center camera).

At the time of writing this document, the right and center camera angle is under 90 degrees, and the left and center camera's angle is over 90 degrees. This has made it harder to collect videos where the target is visible to the center and left camera in all areas in the rig.

#### Video Format from Jason's Data Collection
The videos are 

## Calibration Steps
1) Collect calibration videos.
* Collect a total of 5 videos to use for calibration. 3 videos will be for calibrating the intrinsic parameters for a single camera. 2 videos will be for collecting videos for calculating extrinsic parameters.
* 
Some notes: In the past we collected one large video for calibration, but have found this to be awkward. Instead by collecting one video for one purpose, it was easier to create the calibration videos. 

2) Detect corners.
