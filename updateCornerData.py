import numpy as np
from absl import app
from absl import flags
import pickle
from cornerdata import CheckerboardCorners
import calibflags

FLAGS = flags.FLAGS
flags.DEFINE_string("fixed", None, "updated corner pickle output name")
flags.adopt_module_key_flags(calibflags)


def main(argv):
    del argv
    # there were some changes to how the CheckerboardCorners class serializes and
    # data. helper script to tweak and resave the dictionary

    with open(FLAGS.detected_corners, "rb") as fid:
        cornerDicts = pickle.load(fid)

    numViews = len(cornerDicts)
    newCornerDicts = []
    for i in range(numViews):
        tempCorners = CheckerboardCorners(i, FLAGS.calib_video)
        tempCorners.corners = cornerDicts[i]['corners']
        tempCorners.corners2 = cornerDicts[i]['corners2']
        tempCorners.frameNumbers = cornerDicts[i]['frameNumbers']

        newCornerDicts.append(tempCorners.toDict())

    with open(FLAGS.fixed, "wb") as fid:
        pickle.dump(newCornerDicts, fid)


if __name__ == "__main__":
    app.run(main)
