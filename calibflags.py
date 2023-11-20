import argparse
import json

# FLAGS = flags.FLAGS
# flags.DEFINE_string("calib_video", None, "Calibration Video")
# flags.DEFINE_string("detected_corners", None, "Pickle for detected corners.")
# flags.DEFINE_string("out_video", None, "output video")
# flags.DEFINE_boolean("debug", False, "Detect corners in a subset of the frames, to speed up the process")

# flags.DEFINE_string("calibrated_name", None, "Calibrated Camera Output File Name.")
# flags.DEFINE_integer("numviews", 3, "Number of total views")

# flags.DEFINE_string("params", None, "json param file")


def parseArgs(argv, description="",paramtype=None):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--out_video", type=str)
    parser.add_argument("--calib_video", type=str)
    parser.add_argument("--detected_corners", type=str)
    parser.add_argument("--calib_data", type=str)
    parser.add_argument("--num_views", type=int, default=3)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_images", action="store_true")
    parser.add_argument("--params", type=str)

    args=vars(parser.parse_args(argv))
    if args["params"] is not None:
        with open(args["params"], "r") as f:
            json_params = json.load(f)
            if paramtype is not None:
              if 'general' in json_params:
                args.update(json_params['general'])
              if paramtype in json_params:
                json_params = json_params[paramtype]
        args.update(json_params)

    return args
