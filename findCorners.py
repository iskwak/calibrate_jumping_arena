import numpy as np
import cv2
# import glob
from matplotlib import pyplot as plt
from absl import app
from absl import flags
import pickle
from cornerdata import CheckerboardCorners
import calibflags

FLAGS = flags.FLAGS
flags.adopt_module_key_flags(calibflags)


def findCorners(detectedCorners, frame, frameNum, color):
    squares_xy = (7, 6)
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

    ret, corners = cv2.findChessboardCorners(frame, squares_xy, None)
    if ret == True:
        corners2 = cv2.cornerSubPix(frame, corners, (5, 5), (-1,-1), criteria)

        cv2.drawChessboardCorners(color, squares_xy, corners2, ret)
        detectedCorners['corners'].append(corners)
        detectedCorners['corners2'].append(corners2)
        detectedCorners['frameNumbers'].append(frameNum)


def main(argv):
    del argv

    numViews = 3
    cap = cv2.VideoCapture(FLAGS.calib_video)
    fullWidth  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(fullWidth / 3)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if FLAGS.out_video is not None:
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        writer = cv2.VideoWriter(FLAGS.out_video, fourcc, fps, (fullWidth, height))


    detectedCorners = []
    for i in range(numViews):
        detectedCorners.append({
            'corners': [],
            'corners2': [],
            'frameNumbers': []
        })

    if FLAGS.out_video is not None:
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        writer = cv2.VideoWriter(FLAGS.out_video, fourcc, fps, (fullWidth, height))


    frameNum = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if FLAGS.debug is True and frameNum > 1000:
            break
        if FLAGS.debug is True and (frameNum %  10) != 0:
            frameNum = frameNum + 1
            continue

        if ret == True:
            colorSplit = []
            graySplit = []
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            for i in range(numViews):
                start = i*width
                end = (i+1)*width
                colorSplit.append(frame[:, start:end, :])
                graySplit.append(gray[:, start:end])

            for i in range(numViews):
                findCorners(detectedCorners[i], graySplit[i], frameNum, colorSplit[i])

            if FLAGS.out_video is not None:                
                fullFrame = np.concatenate(colorSplit, axis=1)
                writer.write(fullFrame)
        else:
            break

        frameNum = frameNum + 1

    cap.release()
    cv2.destroyAllWindows()
    if FLAGS.out_video is not None:
        writer.release()


    cornerData = []
    for i in range(numViews):
        corners = np.array(detectedCorners[i]['corners'])
        corners2 = np.array(detectedCorners[i]['corners2'])
        checkerCorners = CheckerboardCorners(i, FLAGS.calib_video, (width, height), corners, corners2, detectedCorners[i]['frameNumbers'])
        cornerData.append(checkerCorners.toDict())

    print("Num found frames: {}, {}, {}".format(
        len(cornerData[0]['frameNumbers']),
        len(cornerData[1]['frameNumbers']),
        len(cornerData[2]['frameNumbers'])))
    with open(FLAGS.detected_corners, "wb") as fid:
        pickle.dump(cornerData, fid)


if __name__ == "__main__":
    app.run(main)
