import numpy as np
import cv2
import pickle
from cornerdata import MultiCamCheckerboardCorners
import calibflags
import os
import sys
import utilities


def findCheckerboards(params):
    numViews = params["num_views"]
    cap, width, fullWidth, height, fourcc, fps = utilities.loadVideo(os.path.join(params["base_dir"], params["calib_video"]), numViews)
    squares_xy = tuple(params["squares_xy"])
    cameraIds = params["views"]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

    # if params["debug_image"] is True:
    #     _, baseFrame = cap.read()
    #     baseFrameSplit = []

    #     for i in range(len(cameraIds)):
    #         start = cameraIds[i]*width
    #         end = (cameraIds[i]+1)*width
    #         baseFrameSplit.append(baseFrame[:, start:end].copy())
    #         cap.set(cv2.CAP_PROP_POS_FRAMES, 0)


    numFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    numCorners = squares_xy[0] * squares_xy[1]
    detectedCorners = {
        "corners": np.zeros((numViews, numFrames, numCorners, 1, 2)),
        "cornerCameraFlag": np.zeros((numViews, numFrames), dtype=bool),
        "frameNumbers": []
    }

    frameNum = 0
    foundFrameFlag = np.zeros((numFrames,), dtype=bool)
    while cap.isOpened():
        ret, frame = cap.read()

        if ret == True:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            for i in range(len(cameraIds)):
                start = cameraIds[i]*width
                end = (cameraIds[i]+1)*width
                
                graySplit = gray[:, start:end]
                #color = frame[:, start:end, :]

                ret, corners = cv2.findChessboardCorners(graySplit, squares_xy, None)
                if ret == True:
                    corners2 = cv2.cornerSubPix(graySplit, corners, (5, 5), (-1,-1), criteria)
                    detectedCorners["corners"][i, frameNum, :, :, :] = corners2
                    detectedCorners["cornerCameraFlag"][i, frameNum] = True
                    foundFrameFlag[frameNum] = True
        else:
            break

        frameNum = frameNum + 1
        if params["debug"] and frameNum > 500:
            break

    cap.release()

    # prune frames with no detections.
    detectedCorners["cornerCameraFlag"] = detectedCorners["cornerCameraFlag"][:, foundFrameFlag]
    detectedCorners["corners"] = detectedCorners["corners"][:, foundFrameFlag, :, :, :]
    detectedCorners["frameNumbers"] = np.where(foundFrameFlag)[0]

    #print("Num found frames " + str(len(detectedCorners["frameNumbers"])))
    for i in range(len(cameraIds)):
        print("Found {} frames for camera {}".format(detectedCorners["cornerCameraFlag"][i,:].sum(), i))
    for i in range(len(cameraIds)):
        for j in range(i,len(cameraIds)):
            if i==j:
                continue
            numFoundCorners = (detectedCorners["cornerCameraFlag"][i,:] * detectedCorners["cornerCameraFlag"][j,:]).sum()
            print("Found {} frames for cameras {} and {}".format(numFoundCorners, i, j))
    

    cornerData = MultiCamCheckerboardCorners(numViews, cameraIds, os.path.join(params["base_dir"], params["calib_video"]), detectedCorners["corners"], detectedCorners["cornerCameraFlag"], detectedCorners["frameNumbers"], (width, height), squares_xy, params["square_mm"])
    cornerData = cornerData.toDict()

    outname = os.path.join(params["base_dir"], params["detected_corners"])
    outpath, _ = os.path.split(outname)
    os.makedirs(outpath, exist_ok=True)
    with open(outname, "wb") as fid:
        pickle.dump(cornerData, fid)


if __name__ == "__main__":
    params = calibflags.parseArgs(sys.argv[1:])
    findCheckerboards(params)
