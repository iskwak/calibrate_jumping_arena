%base_out = '/groups/branson/bransonlab/kwaki/ForceData/avian3dstuff/calibration/20220726/';
%base_out = '/groups/branson/bransonlab/kwaki/ForceData/avian3dstuff/calibration/20220726_bigvideo';
%base_out = '/groups/branson/bransonlab/kwaki/ForceData/avian3dstuff/calibration/20220726_bigvideo_test';
%base_out = '/groups/branson/bransonlab/kwaki/ForceData/avian3dstuff/calibration/20220913_stereo_test';
%base_out = '/groups/branson/bransonlab/kwaki/ForceData/avian3dstuff/calibration/20220913_stereo_test_copy';
base_out = '/groups/branson/bransonlab/kwaki/ForceData/avian3dstuff/calibration/20221011';

calib_filenames = { ...
    fullfile(base_out, 'cam_01_opencv.mat'), ...
    fullfile(base_out, 'cam_02_opencv.mat'), ...
    fullfile(base_out, 'cam_12_opencv.mat') ...
};
outname = fullfile(base_out, 'multi_cam_calib_updated_refactor.mat');

create_calrignpairwise(outname, calib_filenames);   