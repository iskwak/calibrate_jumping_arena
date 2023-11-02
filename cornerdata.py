import numpy as np
import numpy.typing as npt
from typing import List, Tuple, Type, Optional


class MultiCamCheckerboardCorners:
    numViews = -1
    cameraIds = []
    videoName = []
    corners = []
    corners2 = []
    frameNumbers = []
    frameSize = (0, 0)
    squares_xy = (7,6)
    square_mm = 5

    def __init__(self, numViews: int, cameraIds: List[int], videoName: str, corners: npt, corners2: npt, frameNumbers: List[int], frameSize: Optional[Tuple[int, int]]=(512,512), squares_xy: Optional[Tuple[int, int]]=(7,6), square_mm: Optional[int]=5):
        self.numViews = numViews
        self.cameraIds = cameraIds
        self.videoName = videoName

        self.corners = corners
        self.corners2 = corners2
        self.frameNumbers = frameNumbers

        self.frameSize = frameSize
        self.square_mm = square_mm
        self.squares_xy = squares_xy


    @classmethod
    def fromDict(cls, dataDict: dict) -> Type["MultiCamCheckerboardCorners"]:
        cornerData = cls(dataDict["numViews"], dataDict["cameraIds"], dataDict["videoName"],
                        dataDict["corners"], dataDict["corners2"], dataDict["frameNumbers"],
                        dataDict["frameSize"], dataDict["squares_xy"], dataDict["square_mm"])
        # cornerData.square_mm = dataDict["square_mm"]
        # cornerData.squares_xy = dataDict["squares_xy"]
        return cornerData


    def toDict(self) -> dict:
        dataDict = {
            "numViews": self.numViews,
            "cameraIds": self.cameraIds,
            "videoName": self.videoName,
            "corners": self.corners,
            "corners2": self.corners2,
            "frameNumbers": self.frameNumbers,
            "frameSize": self.frameSize,
            "squares_xy": self.squares_xy,
            "square_mm": self.square_mm
        }

        return dataDict


    def setupObjPoints(self) -> npt.ArrayLike:
        objectPoints = []
        for i in range(len(self.frameNumbers)):
            objp = np.zeros((self.squares_xy[0] * self.squares_xy[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:self.squares_xy[0], 0:self.squares_xy[1]].T.reshape(-1,2)
            objp = objp * self.square_mm
            objectPoints.append(objp)
        return objectPoints
