#!/usr/bin/env python
# Author: Charlie Hartsell <charles.a.hartsell@vanderbilt.edu>
"""This module defines a PyTorch IterableDataset class for reading ROS Bag files"""
import os
import itertools
import random
import rospy
import rosbag
import numpy as np
import torch
from torch.utils import data
from alc_utils import config as alc_config

DATASET_NAME = "DatasetROS"


class DatasetROS(data.IterableDataset):
    """Custom PyTorch IterableDataset class for reading ROS Bag files"""

    def __init__(self, data_dir_list, data_formatter, **kwargs):
        super(DatasetROS, self).__init__()

        # Scan each directory for all data files and load messages from each
        self.formatter = data_formatter
        self.topic_names = self.formatter.get_topic_names()
        self.data_dir_list = data_dir_list
        self.kwargs = kwargs

    def __iter__(self):
        worker_info = data.get_worker_info()
        if worker_info is not None:
            raise RuntimeError(
                "DatasetROS does not support multi-process loading")

        for data_point in load_rosbag_data(self.data_dir_list, self.topic_names, **self.kwargs):
            # Formatter expects dictionary mapping topic names to message. Construct this.
            topics_dict = {}
            for j, topic in enumerate(self.topic_names):
                topics_dict[topic] = data_point[j]

            # Get properly formatted input/output for neural network and add to data arrays
            f_ins = self.formatter.format_input(topics_dict)
            f_outs = self.formatter.format_training_output(topics_dict)
            if (f_ins is None) or (f_outs is None) or (len(f_ins) == 0) or (len(f_outs) == 0):
                continue

            #f_vals = []
            #f_outvals = []
            #i = -1
            # for f in f_ins:
            #    i +=1
            #    if ((not f_outs[i]) or (f_outs[i] == 0)):
            #        continue
            #    f_vals.append(f)
            #    f_outvals.append(f_outs[i])

            # if (len(f_vals)==0):
            #    continue

            #f_ins = f_vals
            #f_outs = f_outvals

            # FIXME: Check for numpy array type is a workaround to support older DataFormatters.
            if isinstance(f_ins, np.ndarray):
                yield f_ins, f_outs
            else:
                for f_in, f_out in zip(f_ins, f_outs):
                    yield f_in, f_out


def load_rosbag_data(data_dirs, data_identifiers, **kwargs):
    # FIXME: More description
    """This function loads the desired topics from all ROS bag files found in the specified data directory."""
    # Read optional parameters
    start_time_s = kwargs.get("start_time_s", 0.0)
    useful_fraction = kwargs.get(
        "useful_data_fraction", alc_config.training_defaults.USEFUL_DATA_FRACTION)
    rng_seed = kwargs.get("rng_seed", alc_config.training_defaults.RNG_SEED)
    topic_names = data_identifiers

    # Create and seed private copy of RNG
    _rng = random.Random()
    _rng.seed(rng_seed)

    # Scan the provided data directory for any bag files.
    # Sort file and directory names so that order is consistent and reproducible.
    bag_file_names = []
    for data_dir in data_dirs:
        for root, dirs, files in os.walk(data_dir):
            files.sort()
            dirs.sort()
            for filename in files:
                if filename.endswith(".bag"):
                    full_filename = os.path.join(root, filename)
                    bag_file_names.append(full_filename)
                    print("Found bag file: %s" % full_filename)

    # Load and store data from each bag file found in the data directory
    for bag_file in bag_file_names:
        # Create topics dict
        topic_to_msg_gen_dict = {}

        # Open bag file/Init rosbag class and set start time
        print("Reading bag file {}...".format(bag_file))
        bag = rosbag.Bag(bag_file)
        start_time = rospy.Time(start_time_s)

        # Load each topic's message generator one at a time to avoid mixing multiple topics in single list
        for topic_name in topic_names:
            topic_to_msg_gen_dict[topic_name] = bag.read_messages(
                topics=[topic_name], start_time=start_time)

        # Gather messages on each topic into 2-D list (indexed by [data_point_number][topic_name_number])
        msg_data_available = True
        while msg_data_available:
            first_topic_timestamp = None
            data_point = []
            for i, topic_name in enumerate(topic_names):
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
                    if random.random() > useful_fraction:
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
            #print 'len data point '+str(len(data_point))
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
