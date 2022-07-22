import numpy as np


class CalibrationFrames:
    camera_name = ""
    view = -1
    corners = []
    corners2 = []
    frame_numbers = []
    grid_points = []
    squares_xy = (7, 6)
    square_mm = 3
    movie_name = ""
    frame_size = (0, 0)

    def __init__(self, name, movie_name, frame_size):
        self.camera_name = name
        self.move_name = movie_name
        self.frame_size = frame_size

        self.corners = []
        self.corners2 = []
        self.grid_points = []
        self.frame_numbers = []

    def setup_obj_points(self):
        object_points = []
        for i in range(len(self.frame_numbers)):
            objp = np.zeros((self.squares_xy[0] * self.squares_xy[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:self.squares_xy[0], 0:self.squares_xy[1]].T.reshape(-1,2)
            objp = objp * self.square_mm
            object_points.append(objp)
        return object_points


    def add_data(self, frame_num, corners, corners2):
        self.corners.append(corners)
        self.corners2.append(corners2)
        self.frame_numbers.append(frame_num)


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
