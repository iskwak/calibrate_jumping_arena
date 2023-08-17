import numpy as np
import numpy.typing as npt
from typing import List, Tuple, Type, Optional


class MultiCamCheckerboardCorners:
    cameraIds = []
    movieNames = []
    numViews = -1
    corners = []
    corners2 = []
    frameNumbers = []
    squares_xy = (7,6)
    square_mm = 3
    cameraFilterIdx = []

    def __init__(self, cornerData: List[Type["CheckerBoardCorners"]]):
        self.numViews = len(cornerData)

        # extract/merge data.
        allFrameNumbers = []
        for i in range(self.numViews):
            currentFrameNumbers = cornerData[i].frameNumbers
            allFrameNumbers.append(np.asarray(currentFrameNumbers))

        idxs = [0] * self.numViews
        uniqueFrames = np.unique(np.concatenate(allFrameNumbers))
        self.cameraFilterIdx = np.zeros((uniqueFrames.shape[0], self.numViews))
        for i in range(len(uniqueFrames)):
            currentFrameIdx = uniqueFrames[i]

            for j in len(idxs):
                if allFrameNumbers[j][idxs[j]] == uniqueFrames[i]:
                    

        # allFrameNumbers = np.concatenate(allFrameNumbers)
        # allCameraFrameIdx = np.concatenate(allCameraFrameIdx)
        # sortingIdx = np.argsort(allFrameNumbers)
        # allFrameNumbers = allFrameNumbers(sortingIdx)
        # allCameraFrameIdx = allCameraFrameIdx(sortingIdx)

        for i in range(self.numViews):
            self.movieNames.append(cornerData[i].movieName)



class CheckerboardCorners:
    cameraId = -1
    movieName = ""
    frameSize = (0, 0)
    corners = []
    corners2 = []
    frameNumbers = []
    squares_xy = (7, 6)
    square_mm = 3


    def __init__(self, cameraId: int, movieName: str, frameSize: Optional[Tuple[int, int]]=(512,512),
                 corners: Optional[npt.ArrayLike]=None, corners2: Optional[npt.ArrayLike]=None,
                 frameNumbers: Optional[List[int]]=[]):
        self.cameraId = cameraId
        self.movieName = movieName
        self.frameSize = frameSize

        self.corners = corners
        self.corners2 = corners2
        self.frameNumbers = frameNumbers


    @classmethod
    def fromDict(cls, dataDict: dict) -> Type["CheckerBoardCorners"]:
        cornerData = cls(dataDict["cameraId"], dataDict["movieName"], dataDict["frameSize"],
                         dataDict["corners"], dataDict["corners2"], dataDict["frameNumbers"])
        cornerData.square_mm = dataDict["square_mm"]
        cornerData.squares_xy = dataDict["squares_xy"]
        return cornerData


    def setup_obj_points(self) -> npt.ArrayLike:
        objectPoints = []
        for i in range(len(self.frameNumbers)):
            objp = np.zeros((self.squares_xy[0] * self.squares_xy[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:self.squares_xy[0], 0:self.squares_xy[1]].T.reshape(-1,2)
            objp = objp * self.square_mm
            objectPoints.append(objp)
        return objectPoints


    def toDict(self) -> dict:
        dataDict = {}
        dataDict["cameraId"] = self.cameraId
        dataDict["corners"] = self.corners
        dataDict["corners2"] = self.corners2
        dataDict["frameNumbers"] = self.frameNumbers
        dataDict["movieName"] = self.movieName
        dataDict["frameSize"] = self.frameSize
        dataDict["squares_xy"] = self.squares_xy
        dataDict["square_mm"] = self.square_mm

        return dataDict


    # def load(self, dataDict: dict):
    #     self.cameraId = dataDict["cameraId"]
    #     self.movieName = dataDict["movieName"]
    #     self.frameSize = dataDict["frameSize"]
    #     self.corners = dataDict["corners"]
    #     self.corners2 = dataDict["corners2"]
    #     self.frameNumbers = dataDict["frameNumbers"]
    #     self.squares_xy = dataDict["squares_xy"]
    #     self.square_mm = dataDict["square_mm"]
