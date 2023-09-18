import numpy as np
import cv2
# import glob
from matplotlib import pyplot as plt
import pickle
from absl import app


def main(argv):
    del argv

    numViews = 3
    #cap = cv2.VideoCapture('/workspace/calibration/test_squares/cal_2023_08_23_13_40_53.mp4')
    cap = cv2.VideoCapture('/workspace/calibration/test_squares/cal_2023_08_30_10_49_34.avi')
    #cap = cv2.VideoCapture('/workspace/calibration/test_squares/cal_2023_08_30_10_53_40.avi')
    fullWidth  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(fullWidth / 3)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fps = cap.get(cv2.CAP_PROP_FPS)

    #squares_xy = [(6,5)]
    #squares_xy = [(4,3), (4, 3), (4, 3)]
    squares_xy = [(4,3)]
    squareIdx = 0

    #frameChange = [fps * 17, fps * 60]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

    cameraIdx = [0, 2]
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    writer = cv2.VideoWriter('/workspace/calibration/test_squares/testBothDetected.avi', fourcc, fps, (width*len(cameraIdx), height*2))

    _, baseFrame = cap.read()
    baseFrameSplit = []

    for i in range(len(cameraIdx)):
        start = cameraIdx[i]*width
        end = (cameraIdx[i]+1)*width
        baseFrameSplit.append(baseFrame[:, start:end].copy())
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frameNum = 0
    while cap.isOpened():
        ret, frame = cap.read()
        # if frameNum >= frameChange[squareIdx]:
        #     squareIdx = 1

        if ret == True:
            colorSplitAny = []
            colorSplitOrg = []
            #graySplit = []
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            allcorners2 = []
            boardFound = [False, False]
            for i in range(len(cameraIdx)):
                start = cameraIdx[i]*width
                end = (cameraIdx[i]+1)*width
                
                graySplit = gray[:, start:end].copy()
                color = frame[:, start:end, :].copy()
                colorSplitOrg.append(frame[:, start:end, :].copy())

                ret, corners = cv2.findChessboardCorners(graySplit, squares_xy[squareIdx], None)
                if ret == True:
                    corners2 = cv2.cornerSubPix(graySplit, corners, (5, 5), (-1,-1), criteria)
                    cv2.drawChessboardCorners(color, squares_xy[squareIdx], corners2, ret)
                    allcorners2.append(corners2)
                    boardFound[i] = True
                colorSplitAny.append(color)

            fullFrame = np.concatenate(colorSplitAny, axis=1)
            if boardFound[0] == False or boardFound[1] == False:
                bottomFrame = np.concatenate(colorSplitOrg, axis=1)
            else:
                bottomFrame = np.concatenate(colorSplitAny, axis=1)
                cv2.drawChessboardCorners(baseFrameSplit[0], squares_xy[squareIdx], allcorners2[0], ret)
                cv2.drawChessboardCorners(baseFrameSplit[1], squares_xy[squareIdx], allcorners2[1], ret)
                
            fullFrame = np.concatenate([fullFrame, bottomFrame], axis=0)
                
            writer.write(fullFrame)
        else:
            break

        frameNum = frameNum + 1

    cap.release()
    cv2.destroyAllWindows()
    writer.release()
    baseFrame = np.concatenate(baseFrameSplit, axis=0)
    cv2.imwrite("/workspace/calibration/test_squares/testdetection.jpg", baseFrame)


if __name__ == "__main__":
    app.run(main)