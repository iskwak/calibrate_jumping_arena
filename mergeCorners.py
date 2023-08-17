# merge detected corners into one large matrix. a bit easier to deal with
# for some downstream tasks.


import numpy as np
import cv2
# import glob
from matplotlib import pyplot as plt
from absl import app
from absl import flags
import pickle
from cornerdata import CheckerBoardCorners

import calibflags

FLAGS = flags.FLAGS
flags.adopt_module_key_flags(calibflags)
