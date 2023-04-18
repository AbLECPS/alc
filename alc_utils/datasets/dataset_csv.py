#!/usr/bin/env python
# Author: Charlie Hartsell <charles.a.hartsell@vanderbilt.edu>
"""This module defines a PyTorch Dataset class for reading CSV files"""
from __future__ import print_function

import pandas as pd
import os
import random
import torch.utils.data
import alc_utils.config as alc_config

DATASET_NAME = "DatasetCSV"


class DatasetCSV(torch.utils.data.IterableDataset):
    """This class handles loading data from CSV files stored in the specified data directory."""

    def __init__(self, data_dir_list, data_formatter, **kwargs):
        super(DatasetCSV, self).__init__()

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
                "CSV DataInterpreter received empty list of topic names to load.")

        # Create and seed private copy of RNG
        self._rng = random.Random()
        self._rng.seed(self._rng_seed)

        # Build CSV file index
        self._csv_files = self._find_csv_files(data_dir_list)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            raise RuntimeError(
                "DatasetROS does not support multi-process loading")

        # Load and store data from each bag file found in the data directory
        for file_name in self._csv_files:
            # Open CSV file and read only desired topics
            print("Reading CSV file {}...".format(file_name))
            csv_data = pd.read_csv(file_name, usecols=self._topic_names)
            for csv_row in csv_data:
                yield csv_row

    def _find_csv_files(self, data_dirs):
        # Scan the provided data directory for any CSV files. Sort file and directory names so that order is consistent and reproducible.
        file_names = []
        for root, dirs, files in os.walk(data_dirs):
            files.sort()
            dirs.sort()
            for filename in files:
                if filename.endswith(".csv"):
                    full_filename = os.path.join(root, filename)
                    file_names.append(full_filename)
                    print("Found CSV file: %s" % (full_filename))

        return file_names
