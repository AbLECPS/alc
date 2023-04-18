#!/usr/bin/env python
# Author: Charlie Hartsell <charles.a.hartsell@vanderbilt.edu>
"""This module defines a PyTorch Dataset class for reading PNG files"""
from __future__ import print_function

import sensor_msgs.msg
import glob
import os
import cv2
import re
import random
import numpy as np
import torch.utils.data
import alc_utils.common as alc_common
import alc_utils.config as alc_config

DATASET_NAME = "DatasetPNG"

topic_name_to_file_prefix_dict = {
    "/sss_sonar/left/data/raw/compressed": "raw_image_left_",
    "/sss_sonar/left/data/ground_truth/compressed": "truth_image_left_",
    "/sss_sonar/right/data/raw/compressed": "raw_image_right_",
    "/sss_sonar/right/data/ground_truth/compressed": "truth_image_right_"
}

input_topic_to_output_topic_dict = {
    "/sss_sonar/left/data/raw/compressed": "/sss_sonar/left/data/ground_truth/compressed",
    "/sss_sonar/right/data/raw/compressed": "/sss_sonar/right/data/ground_truth/compressed"
}

input_topics = [
    "/sss_sonar/left/data/raw/compressed",
    "/sss_sonar/right/data/raw/compressed"
]


class DatasetPNG(torch.utils.data.Dataset):
    """This class handles loading data from CSV files stored in the specified data directory."""

    def __init__(self, data_dir_list, data_formatter, **kwargs):
        super(DatasetPNG, self).__init__()

        # Store arguments, read topic names, and read optional parameters
        self._data_dir_list = data_dir_list
        self._formatter = data_formatter
        self._topic_names = self._formatter.get_topic_names()
        self._useful_fraction = kwargs.get(
            "useful_data_fraction", alc_config.training_defaults.USEFUL_DATA_FRACTION)
        self._rng_seed = kwargs.get(
            "rng_seed", alc_config.training_defaults.RNG_SEED)

        # Input checks
        if len(self._topic_names) == 0:
            raise ValueError(
                "PNG DataInterpreter received empty list of topic names to load.")

        # Create and seed private copy of RNG
        self._rng = random.Random()
        self._rng.seed(self._rng_seed)

        # Build PNG file index
        self._file_index = self._build_index(data_dir_list)

    def __len__(self):
        return len(self._file_index)

    def __getitem__(self, idx):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            raise RuntimeError(
                "DatasetROS does not support multi-process loading")

        # Read filenames from index and load PNG file to a ROS message
        directory, input_topic, index = self._file_index[idx]
        input_filename, output_filename = _construct_png_filenames(
            directory, input_topic, index)
        input_msg = _read_png_to_compressed_img_msg(input_filename)
        output_msg = _read_png_to_compressed_img_msg(output_filename)

        # Construct topic dictionary for DataFormatter
        topic_to_msg_dict = {}
        topic_to_msg_dict[input_topic] = input_msg
        topic_to_msg_dict[input_topic_to_output_topic_dict[input_topic]] = output_msg

        # Get properly formatted input/output for neural network and add to data arrays
        f_ins = self._formatter.format_input(topic_to_msg_dict)
        f_outs = self._formatter.format_training_output(topic_to_msg_dict)
        if (f_ins is None) or (f_outs is None) or (len(f_ins) == 0) or (len(f_outs) == 0):
            raise IOError("DataFormatter returned an invalid value.")

        # FIXME: Check for numpy array type is a workaround to support older DataFormatters.
        #       Newer formatters should return a python list. Lists containing only one element are fine.
        if isinstance(f_ins, np.ndarray):
            return f_ins, f_outs
        else:
            if (len(f_ins) > 1) or (len(f_outs) > 1):
                raise IOError(
                    "DataFormatter returned too many values. Expected 1.")
            return f_ins[0], f_outs[0]

    def _build_index(self, data_dirs):
        # Find all subdirectories which contain at least one PNG file
        png_file_index = []
        dirs_with_pngs = []
        for data_dir in data_dirs:
            for root, dirs, files in os.walk(data_dir):
                files.sort()
                dirs.sort()
                for filename in files:
                    if filename.endswith(".png"):
                        print("Found sub-directory containing PNG files: %s" % root)
                        dirs_with_pngs.append(
                            alc_common.strip_trailing_separator(root))
                        break

        # Load data from each directory containing PNGs individually
        for current_dir in dirs_with_pngs:
            print("Loading PNG files from directory %s..." % current_dir)

            # Find the greatest data point index from among the valid input topics
            greatest_index = -1
            for input_topic in input_topics:
                file_prefix = topic_name_to_file_prefix_dict[input_topic]
                full_file_prefix = "%s%s%s" % (
                    current_dir, os.path.sep, file_prefix)
                for filename in glob.glob("%s*.png" % full_file_prefix):
                    reg_expression_match = re.search(
                        "%s(.*?)\\.png" % full_file_prefix, filename)
                    file_index = int(reg_expression_match.group(1))
                    if file_index > greatest_index:
                        greatest_index = file_index

            # Scan for PNG files from 0 to the maximum index found in this directory
            for data_point_index in range(0, greatest_index):
                # For each valid input PNG, must have a matching labeled output PNG file
                for input_topic in input_topics:
                    # Construct input & output PNG filenames
                    input_png_file, output_png_file = _construct_png_filenames(current_dir,
                                                                               input_topic,
                                                                               data_point_index)

                    # If both input and output PNG exist, add info to index
                    if os.path.isfile(input_png_file) and os.path.isfile(output_png_file):
                        # Process this point and add to the batch only if it is deemed "useful"
                        if self._rng.random() > self._useful_fraction:
                            continue

                        # FIXME: This is inefficient and should be removed when possible
                        # Before adding files to index, make sure DataFormatter will return a valid value for these PNGs
                        input_msg = _read_png_to_compressed_img_msg(
                            input_png_file)
                        output_msg = _read_png_to_compressed_img_msg(
                            output_png_file)
                        # Construct topic dictionary for DataFormatter
                        topic_to_msg_dict = {}
                        topic_to_msg_dict[input_topic] = input_msg
                        topic_to_msg_dict[input_topic_to_output_topic_dict[input_topic]] = output_msg
                        # Get properly formatted input/output for neural network and add to data arrays
                        f_ins = self._formatter.format_input(topic_to_msg_dict)
                        f_outs = self._formatter.format_training_output(
                            topic_to_msg_dict)
                        if (f_ins is None) or (f_outs is None) or (len(f_ins) == 0) or (len(f_outs) == 0):
                            continue

                        png_file_index.append(
                            (current_dir, input_topic, data_point_index))

        return png_file_index


def _construct_png_filenames(directory, input_topic, index):
    input_file_prefix = topic_name_to_file_prefix_dict[input_topic]
    input_png_file = "%s/%s%d.png" % (directory, input_file_prefix, index)
    output_topic = input_topic_to_output_topic_dict[input_topic]
    output_file_prefix = topic_name_to_file_prefix_dict[output_topic]
    output_png_file = "%s/%s%d.png" % (directory, output_file_prefix, index)
    return input_png_file, output_png_file


def _read_png_to_compressed_img_msg(filepath):
    if not os.path.isfile(filepath):
        raise IOError(
            "Specified PNG filepath (%s) is not a valid file" % filepath)

    # Read PNG to CV2, then convert to ROS CompressedImage message for compatibility with DataFormatter
    cv2_image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    comp_img_msg = sensor_msgs.msg.CompressedImage()
    comp_img_msg.format = "jpeg"
    comp_img_msg.data = np.array(cv2.imencode('.jpg', cv2_image)[1]).tostring()

    return comp_img_msg
