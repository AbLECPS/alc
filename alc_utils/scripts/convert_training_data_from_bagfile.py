#!/usr/bin/env python
""" This script extracts training data from a ROS bag file, formats it, and stores it in a simpler pickle format.
This simpler format is useful for running unit tests"""
from __future__ import print_function

import json
import argparse
import os
import alc_utils.common as alc_common
import pickle

DEFAULT_TEST_DATA_COUNT = 10
DEFAULT_BAGFILE = "/hdd0/alc_workspace/uuv-sim/project/model/2019_04_13_00_42_48/config-2/results/recording.bag"
DEFAULT_NETWORK_INPUT_SHAPE = (66, 200, 3)
DEFAULT_OUTPUT_DIRECTORY = "/tmp/alc/converted_bag_files"


def convert_training_data_from_bagfile(bagfile, input_shape, formatter_path, params, output_file):
    # Construct data URI
    bagfile_dir = os.path.split(bagfile)[0]
    data_uri = {"directory": bagfile_dir}

    # Load rosbag
    formatter_params = {"input_shape": input_shape}
    data_formatter = alc_common.load_formatter(
        formatter_path, **formatter_params)
    formatted_data = alc_common.load_training_datasets(
        [data_uri], data_formatter, **params)

    with open(output_file, 'w') as pkl_fp:
        pickle.dump(formatted_data, pkl_fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Utility script which reads data from a ROS Bag file, formats the '
                                                 'data, then saves the formatted data in a pickle format.')
    parser.add_argument(
        '--bagfile', help='Path to ROS bag file to convert.', required=True)
    parser.add_argument(
        '--formatter_path', help='Path to DataFormatter class to use for formatting data.', required=True)
    parser.add_argument(
        '--formatted_shape', help='Desired shape of the formatted input data.', required=True)
    parser.add_argument(
        '--output_file', help='Path of output pickle file.', required=True)
    parser.add_argument('--parameter_file',
                        help='Path to JSON parameter file. Parameters are passed to DataLoader.',
                        default=None)
    args = parser.parse_args()

    params = {}
    if args.parameter_file is not None:
        with open(args.parameter_file, 'r') as param_fp:
            params = json.load(param_fp)

    # Convert formatted shape argument (string) to list of integers. Assumes values are comma-separated
    formatted_shape = [int(s) for s in args.formatted_shape.split(',')]

    convert_training_data_from_bagfile(args.bagfile,
                                       formatted_shape,
                                       args.formatter_path,
                                       params,
                                       args.output_file)
