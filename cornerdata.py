import numpy as np
import numpy.typing as npt
from typing import List, Tuple, Type, Optional


class MultiCamCheckerboardCorners:
    numViews = -1
    cameraIds = []
    videoName = []
    corners = []
    cornerCameraFlag = []
    frameNumbers = []
    frameSize = (0, 0)
    squares_xy = (7,6)
    square_mm = 5

    def __init__(self, numViews: int, cameraIds: List[int], videoName: str, corners: npt, cornerCameraFlag: npt, frameNumbers: List[int], frameSize: Optional[Tuple[int, int]]=(512,512), squares_xy: Optional[Tuple[int, int]]=(7,6), square_mm: Optional[int]=5):
        self.numViews = numViews
        self.cameraIds = cameraIds
        self.videoName = videoName

        self.corners = corners.astype('float32')
        self.frameNumbers = frameNumbers
        self.cornerCameraFlag = cornerCameraFlag

        self.frameSize = frameSize
        self.square_mm = square_mm
        self.squares_xy = squares_xy


    @classmethod
    def fromDict(cls, dataDict: dict) -> Type["MultiCamCheckerboardCorners"]:
        cornerData = cls(dataDict["numViews"], dataDict["cameraIds"], dataDict["videoName"],
                        dataDict["corners"], dataDict["cornerCameraFlag"], dataDict["frameNumbers"],
                        dataDict["frameSize"], dataDict["squares_xy"], dataDict["square_mm"])
        return cornerData


    def toDict(self) -> dict:
        dataDict = {
            "numViews": self.numViews,
            "cameraIds": self.cameraIds,
            "videoName": self.videoName,
            "corners": self.corners,
            "frameNumbers": self.frameNumbers,
            "frameSize": self.frameSize,
            "squares_xy": self.squares_xy,
            "square_mm": self.square_mm,
            "cornerCameraFlag": self.cornerCameraFlag
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
