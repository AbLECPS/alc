#!/usr/bin/env python
# Author: Charlie Hartsell <charles.a.hartsell@vanderbilt.edu>
"""This file defines the base AssuranceMonitor class.
Specific AssuranceMonitor implementations should inherit from this class"""

import pickle
import os
import numpy as np
import alc_utils.common
import functools
import datetime
import shutil
import json
import glob
from inspect import getargspec

DEFAULT_ASSURANCE_MONITOR_BASENAME = "assurance_monitor.pkl"


class AssuranceMonitor(object):
    """Base assurance monitor class."""
    _variant = "base"

    def __init__(self):
        self.data_formatter = None

    @property
    def variant(self):
        return self._variant

    # FIXME: make_unique_subdir should probably default to TRUE,
    #  but some of the derived classes seem to mess with this...
    def save(self, save_dir, data_formatter_path=None, lec_storage_metadata=None, make_unique_subdir=True):
        """Saves an assurance monitor in the specified directory by pickling the entire class instance.

        Dervied assurance monitors can override this default save behavior if necessary."""
        # Create unique sub-directory if desired
        if make_unique_subdir:
            _save_dir = self._make_unique_subdir(save_dir)
        else:
            _save_dir = save_dir

        # Copy the storage metadata of the associated LEC to this assurance monitor's output directory
        # Each assurance monitor is tied to a single LEC, so this allows us to fetch the corresponding LEC later
        if lec_storage_metadata is not None:
            lec_storage_metadata_path = os.path.join(
                _save_dir, "lec_storage_metadata.json")
            with open(lec_storage_metadata_path, 'w') as metadata_fp:
                json.dump(lec_storage_metadata, metadata_fp)

        # Save assurance monitor (complete class instance) as pickle file
        assurance_monitor_path = os.path.join(
            _save_dir, DEFAULT_ASSURANCE_MONITOR_BASENAME)
        self.reset_mode()

        with open(assurance_monitor_path, 'wb') as pkl_fp:
            pickle.dump(self, pkl_fp)

        # If a particular data formatter is associated with this class, save it alongside the monitor
        print('in save')
        print(data_formatter_path)
        if data_formatter_path is not None:
            copy_data_formatter_path = os.path.join(
                _save_dir, "data_formatter.py")
            shutil.copy(data_formatter_path, copy_data_formatter_path)

        # TODO: Create and save assurance monitor metadata file

        # Return path to saved file
        return _save_dir

    @staticmethod
    def _make_unique_subdir(orig_dir):
        # Create unique sub-directory if desired
        datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        unique_subdir = os.path.join(
            orig_dir, "assurance_monitor_" + datetime_str)
        os.mkdir(unique_subdir)
        return unique_subdir

    @staticmethod
    def load(file_path):
        """Loads an AssuranceMonitor instance from a pickle file or directory at the specified path.

        The loaded class instance can be of a any type derived from this class as long as the default save behavior
        was used. Dervied assurance monitors can override this default load behavior if necessary."""
        # If provided path is a file, assume it is a pickle file.
        # If it is a directory, assume directory contains assurance monitor files.
        # Otherwise, raise exception
        if os.path.isfile(file_path):
            _assurance_monitor = AssuranceMonitor._load_pkl_file(file_path)
        elif os.path.isdir(file_path):
            # Check if this directory contains a saved AssuranceMonitor file.
            dir_path = file_path
            assert os.path.isdir(
                dir_path), 'Provided path (%s) is not a valid directory' % dir_path

            if os.path.isfile(os.path.join(dir_path, DEFAULT_ASSURANCE_MONITOR_BASENAME)):
                _dir_path = dir_path
            else:
                # If this directory does NOT contain a saved AssuranceMonitor, search for a subdirectory which does.
                sub_dirs = []
                candidates = glob.glob(os.path.join(
                    dir_path, "assurance_monitor*"))
                for candidate in candidates:
                    if os.path.isdir(candidate):
                        sub_dirs.append(candidate)

                if len(sub_dirs) == 0:
                    raise IOError(
                        "Failed to find a valid AssuranceMonitor directory in path: %s" % dir_path)
                elif len(sub_dirs) > 1:
                    raise IOError("AssuranceMonitor found multiple valid directories in path (%s), "
                                  "but only accepts one valid directory. " % dir_path)

                # If one and only one valid subdirectory was found, try to load assurance monitor from it
                _dir_path = sub_dirs[0]

            _assurance_monitor = AssuranceMonitor._load_dir(_dir_path)
        else:
            raise IOError("Path provided for loading AssuranceMonitor (%s) is not a valid file or directory."
                          % file_path)

        return _assurance_monitor

    @staticmethod
    def _load_pkl_file(pkl_file_path):
        assert os.path.isfile(
            pkl_file_path), 'Assurance Monitor file {} is not a valid file'.format(pkl_file_path)
        assert ".pkl" in pkl_file_path, 'Assurance Monitor file {} is not a Pickle file'.format(
            pkl_file_path)

        # Load and return assurance monitor instance from pickle file
        with open(pkl_file_path, 'rb') as pkl_fp:
            _assurance_monitor = pickle.load(pkl_fp)

        return _assurance_monitor

    @staticmethod
    def _load_dir(dir_path):
        # Load the assurance monitor pickle file
        pkl_file_path = os.path.join(
            dir_path, DEFAULT_ASSURANCE_MONITOR_BASENAME)
        _assurance_monitor = AssuranceMonitor._load_pkl_file(pkl_file_path)

        # Perform any additional, class-specific loading procedures
        _assurance_monitor._load_extra(dir_path)

        # If a data formatter exists in this directory, load it in the assurance monitor instance
        data_formatter_path = os.path.join(dir_path, "data_formatter.py")
        if os.path.isfile(data_formatter_path):
            _assurance_monitor.data_formatter = alc_utils.common.load_am_data_formatter(
                data_formatter_path)
            _assurance_monitor.reset_mode()

        return _assurance_monitor

    def _load_extra(self, dir_path):
        """Derived classes should override this function if they require additional class-specific
        loading procedures. Function is called after class instance has been loaded from pickle file."""
        pass

    def train(self, *args, **kwargs):
        raise NotImplementedError(
            "Abstract train() function of AssuranceMonitor base class called.")

    def _evaluate(self, *args, **kwargs):
        raise NotImplementedError(
            "Abstract evaluate() function of AssuranceMonitor base class called.")

    def evaluate(self, input_data, *args, **kwargs):
        """Convenience function which applies DataFormatter if input_data is not already formatted,
         then calls evaluate function for particular AssuranceMonitor instance."""
        format_data = kwargs.get("format_data", True)
        if format_data and (self.data_formatter is not None):
            # FIXME: The "isinstance" check is a workaround to support code using older assurance monitors.
            #  Should be removed when no longer needed
            if isinstance(input_data, dict):
                args1 = getargspec(self.data_formatter.format_input)
                num_params = len(args1.args)
                if (num_params == 2):
                    formatted_input = self.data_formatter.format_input(
                        input_data)
                elif (num_params == 3):
                    formatted_input = self.data_formatter.format_input(
                        input_data, None)
            else:
                formatted_input = input_data
        else:
            formatted_input = input_data

        if formatted_input is None:
            return None
        else:
            return self._evaluate(formatted_input, *args, **kwargs)

    # Format data given as numpy array into appropriate shape for assurance monitor
    # Data should be a flattened from original_dimension -> desired_dimension.
    # Dimensions 0:(desired_dimension - 1) are preserved.
    # Dimensions (desired_dimension - 1):original_dimension are flattened into a vector.
    @staticmethod
    def _format_data(data, desired_dimension=2):
        # If data is None or <= desired dimension, return immediately
        if (data is None) or len(data.shape) <= desired_dimension:
            return data

        # Reshape to desired dimension
        flattened_vector_len = functools.reduce(
            lambda x, y: x * y, data.shape[(desired_dimension - 1):])
        # Convert shape tuple to list
        final_shape = []
        for i in data.shape[0:(desired_dimension - 1)]:
            final_shape.append(i)
        final_shape.append(flattened_vector_len)
        return np.reshape(data, final_shape)

    # Set the mode to training in the data-formatter
    def set_training_mode(self):
        if self.data_formatter and hasattr(self.data_formatter, 'mode'):
            self.data_formatter.mode = 'TRAINING'

    # Unset mode in the data-formatter
    def reset_mode(self):
        if self.data_formatter and hasattr(self.data_formatter, 'mode'):
            self.data_formatter.mode = ''
