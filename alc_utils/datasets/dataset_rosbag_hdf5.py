#!/usr/bin/env python
# Author: Charlie Hartsell <charles.a.hartsell@vanderbilt.edu>
import os
import itertools
import random
import rospy
import rosbag
import h5py
import numpy as np
import torch
from torch.utils import data
from alc_utils import config as alc_config

DATASET_NAME = "DatasetROSHDF5"


class DatasetROSHDF5(data.Dataset):
    """Custom PyTorch Dataset class for reading ROS Bag files"""

    def __init__(self, data_dir_list, data_formatter, **kwargs):
        super(DatasetROSHDF5, self).__init__()

        # Save arguments
        self.formatter = data_formatter
        self.topic_names = self.formatter.get_topic_names()
        self.data_dir_list = data_dir_list
        self.kwargs = kwargs

        # Read optional parameters
        self._start_time_s = kwargs.get("start_time_s", 0.0)
        self._useful_fraction = kwargs.get(
            "useful_data_fraction", alc_config.training_defaults.USEFUL_DATA_FRACTION)
        self._rng_seed = kwargs.get(
            "rng_seed", alc_config.training_defaults.RNG_SEED)

        # Create and seed private copy of RNG
        self._rng = random.Random()
        self._rng.seed(self._rng_seed)

        # Transform all ROS Bag files into HDF5 files and store file metadata
        self._h5_file_info = self.transform_dataset_to_hdf5()

        # Open each generated h5 file for reading and calculate full dataset size
        self._h5_files = []
        self._dataset_len = 0
        self.is_input_tensor = False
        self.is_output_tensor = False
        for filename, _, datafile_len in self._h5_file_info:
            self._h5_files.append(h5py.File(filename, 'r'))
            self._dataset_len += datafile_len

    def __len__(self):
        return self._dataset_len

    def __getitem__(self, idx):
        # FIXME: Get worker info throws AttributeError in some situations.
        #  Suspect that older versions of PyTorch did not have this feature. Commented out for now.
        #worker_info = data.get_worker_info()
        # if worker_info is not None:
        #    raise RuntimeError("DatasetROSHDF5 does not support multi-process loading")

        h5_file_idx, item_idx = self._translate_item_index(idx)
        h5_file = self._h5_files[h5_file_idx]
        input_val = h5_file["input"][item_idx]
        output_val = h5_file["output"][item_idx]
        if (self.is_input_tensor):
            input_val = torch.from_numpy(input_val)
        if (self.is_output_tensor):
            output_val = torch.from_numpy(output_val)
        return input_val, output_val

    def __del__(self):
        # Close and delete generated H5 files
        for h5_file in self._h5_files:
            h5_file.close()
        for filename, _, _ in self._h5_file_info:
            os.remove(filename)

    def _translate_item_index(self, idx):
        """Translate an index of an item in the complete dataset into the specific H5 file it belongs to,
        and the corresponding index within that file."""
        for _, h5_file_idx, file_len in self._h5_file_info:
            # Check if item index falls in this H5 file.
            # If not, decrement idx by file length and move to next file.
            if idx < file_len:
                return h5_file_idx, idx
            else:
                idx -= file_len

        raise IndexError(
            "Requested item index does not exist in the loaded dataset.")

    def transform_dataset_to_hdf5(self):
        # Scan the provided data directory for any bag files.
        # Sort file and directory names so that order is consistent and reproducible.
        bag_file_names = []
        for data_dir in self.data_dir_list:
            for root, dirs, files in os.walk(data_dir):
                files.sort()
                dirs.sort()
                for filename in files:
                    if filename.endswith(".bag"):
                        full_filename = os.path.join(root, filename)
                        bag_file_names.append(full_filename)
                        print("Found bag file: %s" % full_filename)

        h5_file_infos = []
        file_index = 0
        for bag_file in bag_file_names:
            input_data = []
            output_data = []
            for data_point in self._load_rosbag_data(bag_file):
                # Formatter expects dictionary mapping topic names to message. Construct this.
                topics_dict = {}
                for j, topic in enumerate(self.topic_names):
                    topics_dict[topic] = data_point[j]

                # Get properly formatted input/output for neural network and add to data arrays
                f_ins = self.formatter.format_input(topics_dict)
                f_outs = self.formatter.format_training_output(topics_dict)
                if (f_ins is None) or (f_outs is None) or (len(f_ins) == 0) or (len(f_outs) == 0):
                    continue

                # FIXME: Check for numpy array type is a workaround to support older DataFormatters.
                #       Newer formatters should return a python list. Lists containing only one element are fine.
                if isinstance(f_ins, np.ndarray):
                    input_data.append(f_ins)
                    output_data.append(f_outs)
                else:
                    if (torch.is_tensor(f_ins[0])):
                        self.is_input_tensor = True
                        f_ins1 = []
                        for f in f_ins:
                            f_ins1.append(f.numpy())
                        f_ins = f_ins1

                    if (torch.is_tensor(f_outs[0])):
                        self.is_output_tensor = True
                        f_outs1 = []
                        for f in f_outs:
                            f_outs1.append(f.numpy())
                        f_outs = f_outs1

                    input_data.extend(f_ins)
                    output_data.extend(f_outs)

            datafile_len = len(input_data)
            if datafile_len != len(output_data):
                raise RuntimeError(
                    "Lengths of input and output datasets did not match when processing bag file %s." % bag_file)
            if datafile_len <= 0:
                print ("Bag file %s resulted in an empty dataset after formatting.")
                continue

            # Create HDF5 file for this ROS Bag. Overwrite any old H5 file with the same name
            bag_filename_extensionless = os.path.splitext(
                os.path.abspath(bag_file))[0]
            h5_basename = os.path.split(bag_filename_extensionless)[
                1] + "_" + str(file_index) + ".h5"
            h5_filename = os.path.join(
                alc_config.WORKSPACE_CACHE_DIR, h5_basename)
            h5_file = h5py.File(h5_filename, 'w')

            # Write datasets to H5 file and close file
            print('shape ', input_data[0].shape)
            print('dtype ', input_data[0].dtype)
            h5_file.create_dataset("input", (datafile_len,) + input_data[0].shape,
                                   dtype=input_data[0].dtype,
                                   data=input_data,
                                   chunks=True)  # ,
            # compression="gzip",
            # compression_opts=1)
            h5_file.create_dataset("output", (datafile_len,) + output_data[0].shape,
                                   dtype=output_data[0].dtype,
                                   data=output_data,
                                   chunks=True)  # ,
            # compression="gzip",
            # compression_opts=1)
            # Store information about this H5 file
            h5_file_infos.append((h5_filename, file_index, datafile_len))
            file_index += 1

        return h5_file_infos

    def _load_rosbag_data(self, bag_file):
        # FIXME: More description
        """This function loads the desired topics from all ROS bag files found in the specified data directory."""
        # Load and store data from each bag file found in the data directory
        # Create topics dict
        topic_to_msg_gen_dict = {}

        # Open bag file/Init rosbag class and set start time
        print("Reading bag file {}...".format(bag_file))
        bag = rosbag.Bag(bag_file)
        start_time = rospy.Time(self._start_time_s)

        # Load each topic's message generator one at a time to avoid mixing multiple topics in single list
        for topic_name in self.topic_names:
            topic_to_msg_gen_dict[topic_name] = bag.read_messages(
                topics=[topic_name], start_time=start_time)

        # Gather messages on each topic into 2-D list (indexed by [data_point_number][topic_name_number])
        msg_data_available = True
        while msg_data_available:
            first_topic_timestamp = None
            data_point = []
            for i, topic_name in enumerate(self.topic_names):
                # Get ROS message generator corresponding to this topic
                msg_gen = topic_to_msg_gen_dict[topic_name]

                # Special case for first topic
                if i == 0:
                    # Get next ROS message from generator
                    try:
                        (_, msg_data, first_topic_timestamp) = next(msg_gen)
                    except StopIteration:
                        # If this ROS Bag is out of messages, set flag to break loop and move on to next bag file.
                        msg_data_available = False
                        break

                    # Skip messages if not deemed "useful" to reduce total size of training set.
                    # This is helpful for quickly debugging/tuning parameters when training networks
                    if random.random() > self._useful_fraction:
                        break

                    # Store all messages deemed "useful"
                    data_point.append(msg_data)

                # For remaining topics, find message which corresponds with timestamp of first topic message
                # ie. Perform approximate time sync of multiple message streams
                else:
                    # Find and store closest matching message based on timestamp from first-topic's message
                    # Also store updated message generator to avoid wasting time repeating search of older messages
                    try:
                        closest_msg, updated_msg_gen, time_error = find_nearest_msg(msg_gen,
                                                                                    first_topic_timestamp)
                    except StopIteration:
                        raise IOError("ROS Bag Interpreter encountered an empty message generator while "
                                      "processing bag file at %s.\n Please make sure selected bag files contain "
                                      "messages for all required topics." % bag_file)

                    topic_to_msg_gen_dict[topic_name] = updated_msg_gen
                    _, msg_data, _ = closest_msg
                    data_point.append(msg_data)

            # If we read some data, yield this data point. Otherwise, continue on to next data point immediately
            if len(data_point) > 0:
                yield data_point


def find_nearest_msg(msgs, time):
    """ Find message among set of ROS messages that is closest to the given time (ROS Time type)
    Assumes that messages are in a generator object and sorted by ascending timestamps
    Typically, this function is used as a way of 'zipping' two message streams with the closest matching time stamps
    With this in mind, an updated msgs generator is returned which allows the caller to skip all earlier messages which
    no longer need to be considered.

    Args:
        msgs (generator((str, ROS msg, ROS timestamp))): ROS message generator sorted by ascending timestamps.
        time (ROS timestamp): ROS timestamp to be matched.

    Returns:
        (closest_msg, updated_msgs_gen, min_dt_abs)

        a) closest_msg (ROS msg): The ROS message from the provided message generator which most closely matches
            the provided timestamp.
        b) updated_msgs_gen (generator): Updated ROS message generator.
            Subset of the provided 'msgs' generator containing all messages from 'closest_msg' onwards (inclusive).
        c) min_dt_abs (float): Absolute time error between 'closest_msg' and provided 'time' argument.
    """
    min_dt_abs = None
    closest_msg = None

    try:
        while True:
            # Determine time-difference (absolute) between this message and specified timestamp
            msg = next(msgs)
            (_, _, msg_time) = msg
            dt_s = time.secs - msg_time.secs
            dt_ns = time.nsecs - msg_time.nsecs
            dt = dt_s + (dt_ns / 1000000000.0)
            dt_abs = abs(dt)

            # Since list is assumed to be in ascending time order, if the time-difference (dt_abs) of this message
            # is not less than the current minimum error, then we have gone past the closest message.
            if (closest_msg is None) or (dt_abs < min_dt_abs):
                min_dt_abs = dt_abs
                closest_msg = msg
            else:
                updated_msgs_gen = itertools.chain([closest_msg, msg], msgs)
                break

    except StopIteration:
        # Message generator ran out of messages.
        # If currently stored 'closest_msg' is not None, then it is the closest message. Update generator and return.
        # Otherwise, no messages were present in generator. Pass exception up the call stack.
        if closest_msg is not None:
            # This generator will contain only one message (closest_msg), then yield StopIteration again.
            # Probably a rare case, but could happen if the message stream ended before the desired time
            updated_msgs_gen = itertools.chain([closest_msg], msgs)
        else:
            raise

    return closest_msg, updated_msgs_gen, min_dt_abs
