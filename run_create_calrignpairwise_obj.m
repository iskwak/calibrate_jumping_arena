calib_filenames = {'cam_01_opencv.mat', 'cam_02_opencv.mat', 'cam_02_opencv.mat'};
outname = 'multi_cam_calib.mat';

create_calrignpairwise(outname, calib_filenames);