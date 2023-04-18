#!/usr/bin/env python2
""" This script converts a ROS bag file into the HDF5 file format."""
from __future__ import print_function

import rosbag_pandas

DEFAULT_BAGFILE = '/home/charlie/alc/alc_utils/test/res/dataset0/1563991410951/1563991410951/config-0/results/recording.bag'
DEFAULT_OUTPUT_FILE = "/tmp/alc/converted_bag_files/rosbag.hdf5"


def rosbag_to_hdf5(bagfile, output_file, **kwargs):
    dataframe = rosbag_pandas.bag_to_dataframe(bagfile)
    dataframe.to_hdf(output_file, "my_bag_file", **kwargs)


if __name__ == "__main__":
    rosbag_to_hdf5(DEFAULT_BAGFILE, DEFAULT_OUTPUT_FILE)


#     parser = argparse.ArgumentParser(description='Utility script which reads data from a ROS Bag file, formats the '
#                                                  'data, then saves the formatted data in a pickle format.')
#     parser.add_argument('--bagfile', help='Path to ROS bag file to convert.', required=True)
#     parser.add_argument('--formatter_path', help='Path to DataFormatter class to use for formatting data.', required=True)
#     parser.add_argument('--formatted_shape', help='Desired shape of the formatted input data.', required=True)
#     parser.add_argument('--output_file', help='Path of output pickle file.', required=True)
#     parser.add_argument('--parameter_file',
#                         help='Path to JSON parameter file. Parameters are passed to DataLoader.',
#                         default=None)
#     args = parser.parse_args()
#
#     params = {}
#     if args.parameter_file is not None:
#         with open(args.parameter_file, 'r') as param_fp:
#             params = json.load(param_fp)
#
#     # Convert formatted shape argument (string) to list of integers. Assumes values are comma-separated
#     formatted_shape = [int(s) for s in args.formatted_shape.split(',')]
#
#     convert_training_data_from_bagfile(args.bagfile,
#                                        formatted_shape,
#                                        args.formatter_path,
#                                        params,
#                                        args.output_file)
