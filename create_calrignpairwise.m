function create_calrignpairwise(outname, calib_filenames)
    % Take the output of the python opencv calibration mat files and construct
    % a struct to pass into CalRigNPairwiseCalibrated. Then save this mat for
    % use with multicamera projects.
    % The mat file created by scipy should be loaded by CalRig2Caltech
    calib_mats = cell(length(calib_filenames), 1);
    for i = 1:length(calib_filenames)
        calib_mats{i} = CalRig2CamCaltech(calib_filenames{i});
    end

    % create a struct to used for input to the CalRigNPairwiseCalibrated
    % constructor
    s.nviews = 3;
    s.calibrations = calib_mats;

    multicam = CalRigNPairwiseCalibrated(s);
    save(outname, 'multicam');
end


    % function obj = CalRigNPairwiseCalibrated(varargin)
    %   if nargin==1 
    %     if isstruct(varargin{1})
    %       s = varargin{1};
    %     end
    %   end
        
    %   ncam = s.nviews;
    %   obj.nviews = ncam;
    %   obj.crigStros = cell(ncam);
    %   crigs = s.calibrations;
    %   c = 1;
    %   % ordering of stereo crigs assumed
    %   for icam=1:ncam
    %   for jcam=icam+1:ncam
    %     obj.crigStros{icam,jcam} = crigs{c};
    %     c = c+1;
    %   end
    %   end
      
    %   assert(c==numel(crigs)+1);
      
    %   obj.viewNames = arrayfun(@(x)sprintf('view%d',x),(1:ncam)','uni',0);
    % end     