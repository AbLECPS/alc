#!/usr/bin/env python2
"""Recursively explore/summarize contents of an HDF5 file."""
from __future__ import print_function

import h5py

DEFAULT_INPUT_FILE = "/tmp/alc/converted_bag_files/rosbag.hdf5"


def summarize_hdf5(input_file, **kwargs):
    f = h5py.File(input_file, 'r')
    print("TOP-LEVEL KEYS: %s" % f.keys())
    explore_group(f)


def explore_group(h5_group):
    print("Exploring group: %s" % h5_group.name)
    print("\t Group info: %s" % str(h5_group))
    try:
        for key in h5_group.keys():
            explore_group(h5_group[k])
    except AttributeError:
        return


if __name__ == "__main__":
    summarize_hdf5(DEFAULT_INPUT_FILE)
