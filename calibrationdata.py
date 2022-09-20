import numpy as np
import pickle


class CheckerboardDetectedFrames:

    def __init__(self, camera_name, movie_name, frame_size, square_mm=3, checkerboard_dims=(7, 6), corners=None, corners2=None, frame_numbers=None):
        self._camera_name = camera_name
        self._movie_name = movie_name
        self._frame_size = frame_size # movie frame size

        self._corners = corners or [] # corner detector
        self._corners2 = corners2 or []# refined corners
        self._frame_numbers = frame_numbers or [] # frame numbers for the corners

        self._checkerboard_dims = checkerboard_dims # dimensions of the checkboard
        self._square_mm = square_mm # square edge length in mm


    # function to create the "flat" real world targets for camera calibration
    def setup_obj_points(self):
        object_points = []
        for i in range(len(self._frame_numbers)):
            objp = np.zeros((self._checkerboard_dims[0] * self._checkerboard_dims[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:self._checkerboard_dims[0], 0:self._checkerboard_dims[1]].T.reshape(-1,2)
            objp = objp * self._square_mm
            object_points.append(objp)
        return object_points

    def add_data(self, frame_num, corners, corners2):
        self._corners.append(corners)
        self._corners2.append(corners2)
        self._frame_numbers.append(frame_num)

    @property
    def camera_name(self):
        return self._camera_name
    
    @property
    def movie_name(self):
        return self._movie_name

    @property
    def frame_size(self):
        return self._frame_size

    @property
    def corners(self):
        return self._corners

    @property
    def corners2(self):
        return self._corners2

    @property
    def movie_name(self):
        return self._movie_name

    @property
    def frame_numbers(self):
        return self._frame_numbers

    @property
    def checkerboard_dims(self):
        return self._checkerboard_dims

    @property
    def square_mm(self):
        return self._square_mm

    def __getitem__(self, index):
        return self._corners[index], self._corners2[index], self._frame_numbers[index]

    def __len__(self):
        return len(self._frame_numbers)

    def __repr__(self):
        return "Camera: {}\nMovie name: {}\nNumber of detected frames: {}\nCheckerboard Dims: {}\nSquare Edge Length: {}\n".format(
            self._camera_name, self._movie_name, len(self._frame_numbers), self._checkerboard_dims, self._square_mm
        )

    def serialize_data(self):
        return vars(self)

    @classmethod
    def from_data(cls, data_dict):
        obj = CheckerboardDetectedFrames(
            data_dict["_camera_name"],
            data_dict["_movie_name"],
            data_dict["_frame_size"],
            square_mm=data_dict["_square_mm"], 
            checkerboard_dims=data_dict["_checkerboard_dims"],
            corners=data_dict["_corners"],
            corners2=data_dict["_corners2"],
            frame_numbers=data_dict["_frame_numbers"])
        return obj
    

    def filter_data(self, index):
        new_corners = []
        new_corners2 = []
        new_frame_numbers = []
        for i in range(len(index)):
            new_corners.append(self._corners[index[i]])
            new_corners2.append(self._corners2[index[i]])
            new_frame_numbers.append(self._frame_numbers[index[i]])

        obj = CheckerboardDetectedFrames(
            self._camera_name,
            self._movie_name,
            self._frame_size,
            square_mm=self._square_mm,
            checkerboard_dims=self._checkerboard_dims,
            corners=new_corners,
            corners2=new_corners2,
            frame_numbers=new_frame_numbers)
        return obj


def load_calibs(filename):
    with open(filename, "rb") as fid:
        checkerboard_data = pickle.load(fid)
        checkerboard_frames = []
        for i in range(len(checkerboard_data)):
            checkerboard_frames.append(CheckerboardDetectedFrames.from_data(checkerboard_data[i]))
    return checkerboard_frames

# class CalibratedCamData:
#     mtx = []
#     dist = []
#     rvecs = []
#     tvecs = []

#     def __init__(self, mtx, dist, rvecs, tvecs):
#         mtx = mtx
#         dist = dist
#         rvecs = rvecs
#         tvecs = tvecs
