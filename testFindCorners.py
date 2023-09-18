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
    cap = cv2.VideoCapture('/workspace/calibration/test_squares/cal_2023_08_24_11_47_28.mp4');
    fullWidth  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(fullWidth / 3)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fps = cap.get(cv2.CAP_PROP_FPS)

    #squares_xy = [(7,6), (9,8)]
    squares_xy = [(6,5)]
    squareIdx = 0

    #frameChange = fps * 9.5

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    writer = cv2.VideoWriter('/workspace/calibration/test_squares/test2.avi', fourcc, fps, (width*3, height))

    frameNum = 0
    while cap.isOpened():
        ret, frame = cap.read()
        # if frameNum >= frameChange:
        #     squareIdx = 1

        if ret == True:
            colorSplit = []
            #graySplit = []
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            for i in range(numViews):
                start = i*width
                end = (i+1)*width
                
                graySplit = gray[:, start:end].copy()
                color = frame[:, start:end, :].copy()
                #import pdb; pdb.set_trace()
                ret, corners = cv2.findChessboardCorners(graySplit, squares_xy[squareIdx], None)
                if ret == True:
                    corners2 = cv2.cornerSubPix(graySplit, corners, (3, 3), (-1,-1), criteria)
                    cv2.drawChessboardCorners(color, squares_xy[squareIdx], corners2, ret)
                colorSplit.append(color)
                #graySplit.append(gray[:, start:end])
            fullFrame = np.concatenate(colorSplit, axis=1)
            writer.write(fullFrame)
        else:
            break

        frameNum = frameNum + 1

    cap.release()
    cv2.destroyAllWindows()
    writer.release()
    #cap.set(cv2.CAP_PROP_POS_FRAMES, fps * 9.5)
    #ret, frame = cap.read()
    #cv2.imshow('moo', frame)
    #cv2.waitKey()




if __name__ == "__main__":
    app.run(main)