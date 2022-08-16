function create_calrignpairwise(outname, calib_filenames)
    % Take the output of the python opencv calibration mat files and construct
    % a struct to pass into CalRigNPairwiseCalibrated. Then save this mat for
    % use with multicamera projects.
    % The mat file created by scipy should be loaded by CalRig2Caltech
    calib_mats = cell(length(calib_filenames), 1);
    for i = 1:length(calib_filenames)
        calib_mats{i} = CalRig2CamCaltech(calib_filenames{i});
    end
    calib_mats{1} = CalRig2CamCaltech(calib_filenames{1});
    calib_mats{2} = CalRig2CamCaltech(calib_filenames{2});
    calib_mats{3} = CalRig2CamCaltech(calib_filenames{3});

    % create a struct to used for input to the CalRigNPairwiseCalibrated
    % constructor
    s.nviews = 3;
    s.calibrations = calib_mats;

    multicam = CalRigNPairwiseCalibrated(s);
    save(outname, 'multicam');
end
