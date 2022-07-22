import numpy as np
import cv2 as cv
import glob
from matplotlib import pyplot as plt


def plot_corners(img, corners, corners2):
    plt.imshow(img)
    # plot points
    plt.plot(corners[:, 0, 0], corners[:, 0, 1])
    # plt.plot(corners2[:, 0, 0], corners2[:, 0, 1], color='red')
    plt.plot(corners[0, 0, 0], corners[0, 0, 1], color='red', marker='x')
    plt.show()


# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('*.jpg')
count = 0
for fname in images:
    print(fname)
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7,6), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (3,3), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        # cv.drawChessboardCorners(img, (7,6), corners2, ret)
        # cv.imwrite("%d.jpg" % count, img)
        # count = count + 1
        # cv.imshow('img', img)
        # cv.waitKey(500)
        plot_corners(img, corners, corners2)
        # plt.imshow(img)
        # plt.xticks([]), plt.yticks([])
        # plt.show()
    count = count + 1

cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
# print(ret)
# print(mtx)
# print(dist)
# print(rvecs)
# print(tvecs)

