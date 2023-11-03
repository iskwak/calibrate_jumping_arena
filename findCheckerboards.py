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

    if params["out_video_dir"] is not None:
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        outName = "detections_"
        for i in range(len(cameraIds)):
            outName = outName + str(cameraIds[i])
        outname = os.path.join(params["base_dir"], params["out_video_dir"], outName+".avi")
        writer = cv2.VideoWriter(outname, fourcc, fps, (width*len(cameraIds), height*2))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

    if params["debug_image"] is True:
        _, baseFrame = cap.read()
        baseFrameSplit = []

        for i in range(len(cameraIds)):
            start = cameraIds[i]*width
            end = (cameraIds[i]+1)*width
            baseFrameSplit.append(baseFrame[:, start:end].copy())
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)


    numFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    numCorners = squares_xy[0] * squares_xy[1]
    detectedCorners = {
        "corners": np.zeros((numViews, numFrames, numCorners, 1, 2)),
        "corners2": np.zeros((numViews, numFrames, numCorners, 1, 2)),
        "frameNumbers": []
    }

    # temp stoarge for detected corners
    currentCorners = np.zeros((numViews, numCorners, 1, 2), dtype=np.float32)
    currentCorners2 = np.zeros((numViews, numCorners, 1, 2), dtype=np.float32)

    frameNum = 0
    while cap.isOpened():
        ret, frame = cap.read()

        if ret == True:
            colorSplitAny = []
            colorSplitOrg = []
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            allFound = True
            for i in range(len(cameraIds)):
                start = cameraIds[i]*width
                end = (cameraIds[i]+1)*width
                
                graySplit = gray[:, start:end]
                color = frame[:, start:end, :]
                colorSplitOrg.append(frame[:, start:end, :].copy())

                ret, corners = cv2.findChessboardCorners(graySplit, squares_xy, None)
                if ret == True:
                    corners2 = cv2.cornerSubPix(graySplit, corners, (5, 5), (-1,-1), criteria)
                    cv2.drawChessboardCorners(color, squares_xy, corners2, ret)
                    #allcorners2.append(corners2)
                    currentCorners[cameraIds[i], :, :, :] = corners
                    currentCorners2[cameraIds[i], :, :, :] = corners2
                else:
                    allFound = False

                colorSplitAny.append(color)

            fullFrame = np.concatenate(colorSplitAny, axis=1)

            if allFound == False:
                bottomFrame = np.concatenate(colorSplitOrg, axis=1)
            else:
                bottomFrame = np.concatenate(colorSplitAny, axis=1)
                for i in range(len(cameraIds)):
                    if params["debug_image"] is True:
                        cv2.drawChessboardCorners(baseFrameSplit[i], squares_xy, currentCorners2[cameraIds[i], :, :, :], ret)
                    # only add the corners if it has been detected in all the views
                    detectedCorners["corners"][cameraIds[i], frameNum, :, :, :] = currentCorners[cameraIds[i], :, :, :]
                    detectedCorners["corners2"][cameraIds[i], frameNum, :, :, :] = currentCorners2[cameraIds[i], :, :, :]
                    detectedCorners["frameNumbers"].append(frameNum)

            #fullFrame = np.concatenate([fullFrame, bottomFrame], axis=0)
                
            #writer.write(fullFrame)
        else:
            break

        frameNum = frameNum + 1
        if params["debug"] and frameNum > 100:
            break

    cv2.destroyAllWindows()
    
    if params["out_video_dir"] is not None:
        writer.release()
    if params["debug_image"] is True:
        baseFrame = np.concatenate(baseFrameSplit, axis=1)
        outName = "detections_"
        for i in range(len(cameraIds)):
            outName = outName + str(cameraIds[i])
        outname = os.path.join(params["base_dir"], params["out_video_dir"], outName+".jpg")
        os.makedirs(os.path.join(params["base_dir"], params["out_video_dir"]), exist_ok=True)
        cv2.imwrite(outname, baseFrame)


    print("Num found frames " + str(len(detectedCorners["frameNumbers"])))
    corners = detectedCorners["corners"][:, detectedCorners["frameNumbers"], :, :, :]
    corners2 = detectedCorners["corners2"][:, detectedCorners["frameNumbers"], :, :, :]
    outname = os.path.join(params["base_dir"], "test.png")
    utilities.plotSampled(cap, outname, detectedCorners["corners2"][cameraIds[0]], squares_xy, 0)

    cornerData = MultiCamCheckerboardCorners(numViews, cameraIds, params["calib_video"], corners, corners2, detectedCorners["frameNumbers"], (width, height), squares_xy, params["square_mm"])
    cornerData = cornerData.toDict()

    outname = os.path.join(params["base_dir"], params["detected_corners"])
    outpath, _ = os.path.split(outname)
    os.makedirs(outpath, exist_ok=True)
    with open(outname, "wb") as fid:
        pickle.dump(cornerData, fid)

    cap.release()


if __name__ == "__main__":
    params = calibflags.parseArgs(sys.argv[1:])
    findCheckerboards(params)
