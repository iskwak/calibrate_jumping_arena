# Test merging detections into MultiCamCheckerboardCorners obj
import numpy as np
from matplotlib import pyplot as plt
from absl import app
from absl import flags
import pickle
import time
import os
import utilities
from cornerdata import CheckerboardCorners, MultiCamCheckerboardCorners
import calibflags


FLAGS = flags.FLAGS
flags.adopt_module_key_flags(calibflags)


def main(argv):
    del argv

    with open(FLAGS.detected_corners , "rb") as fid:
        cornerDict = pickle.load(fid) 
        cornerData = []
        for i in range(len(cornerDict)):
            cornerData.append(CheckerboardCorners.fromDict(cornerDict[i]))
    
    multicamCheckers = MultiCamCheckerboardCorners(cornerData)
    print(multicamCheckers.numViews)
    
    
if __name__ == "__main__":
    app.run(main)
