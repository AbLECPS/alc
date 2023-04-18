# !/usr/bin/env python
# Author: Charlie Hartsell <charles.a.hartsell@vanderbilt.edu>
"""This module provides commonly used utility functions for the ALC project"""
from __future__ import print_function
from future.utils import viewitems, iteritems

import os
import sys
import errno
import re
import json
import six
import stat
import alc_utils.config as alc_config


# Function which produces a hash based on the name and contents of a specified directory
# From https://stackoverflow.com/questions/24937495/how-can-i-calculate-a-hash-for-a-filesystem-directory-using-python/31691399
def hash_directory(path):
    """Hashes the names and contents of all files in the specified directory. Returns a hexadecimal hash value."""
    # Import necessary libraries
    import hashlib

    digest = hashlib.sha1()

    # Remove any trailing path separators
    if path.endswith(os.path.sep):
        path = path[:-1]

    # Make sure directory exists
    if not os.path.isdir(path):
        raise ValueError(
            "Hash directory function received directory that does not exist (%s)." % path)

    # Hash all files in directory
    for root, dirs, files in os.walk(path):
        # Sort directory and file names so they are visited in a consistent order
        dirs.sort()
        files.sort()
        for names in files:
            file_path = os.path.join(root, names)

            # Hash the path and add to the digest to account for empty files/directories
            digest.update(hashlib.sha1(
                file_path[len(path):].encode()).digest())

            # Per @pt12lol - if the goal is uniqueness over repeatability, this is an alternative method using 'hash'
            # digest.update(str(hash(file_path[len(path):])).encode())

            if os.path.isfile(file_path):
                with open(file_path, 'rb') as f_obj:
                    while True:
                        buf = f_obj.read(1024 * 1024)
                        if not buf:
                            break
                        digest.update(buf)

    return digest.hexdigest()


def generate_pkl_file(bag_file):
    """Generates a pickle file from a ROS bag file.

    Generated pickle file will have the same name as the specified bag file with ".bag" replaced by ".pkl".
    Conversion is very slow on large files and is not recommended for use on bag files > 1 GB."""
    # Import necessary libraries
    import rosbag
    import pickle

    # Check input and load bag file
    if not (bag_file.endswith('.bag')):
        return None
    if not (os.path.isfile(bag_file)):
        return None
    bag = rosbag.Bag(bag_file)

    msgs_dict = {}
    for topic, msg, time in bag.read_messages():
        # init each topic on first encounter
        if msgs_dict.get(topic) is None:
            msgs_dict[topic] = []

        # Turn each message into a dictionary and append to topic array
        msg_dict = {'data': msg, 'time': time}
        msgs_dict[topic].append(msg_dict)

    # Close bag file
    bag.close()

    # Write pickle dump file
    _, bag_file_basename = os.path.split(bag_file)
    pkl_filename = bag_file_basename[:-4] + '.pkl'
    with open(pkl_filename, 'wb') as pkl_fp:
        pickle.dump(msgs_dict, pkl_fp, protocol=pickle.HIGHEST_PROTOCOL)

    return pkl_filename


def source_env_vars(vars_dict):
    """Loads variables from the specified dictionary and stores them in "os.environ"."""
    for key, value in viewitems(vars_dict):
        if value is None:
            os.environ[key] = ''
        else:
            os.environ[key] = str(value)


# Get IP of a network interface
def get_ip_address(ifname="eth0"):
    """Attempts to find the IP address of the specified network interface."""
    import socket
    import fcntl
    import struct

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return socket.inet_ntoa(fcntl.ioctl(
        s.fileno(),
        0x8915,  # SIOCGIFADDR
        struct.pack('256s', ifname[:15])
    )[20:24])


def get_complete_training_metadata(network_metadata):
    """Returns a complete network metadata containing all datasets this network has been trained on.

    For networks trained only once (starting from an empty architecture),
    the complete metadata is the same as the network metadata.

    For networks trained more than once (starting from a parent network model),
    need to recursively search parent metadata as well until parent is "None"."""
    training_data_metadata = []

    while network_metadata is not None:
        # Read training set info from this network's metadata, and add to metadata list
        next_training_metadata = network_metadata.get(
            "dataset_storage_metadata", [])
        training_data_metadata.extend(next_training_metadata)

        # Update network_metadata to parent network's metadata for recursive search
        network_metadata = network_metadata.get("parent_model_metadata", None)

    return training_data_metadata


def dataset_metadata_is_equal(metadata1, metadata2):
    """Checks if the two provided dataset storage metadata dictionaries are equal."""
    # Compare dataset hashes if available
    hash1 = metadata1.get("hash", None)
    hash2 = metadata2.get("hash", None)
    if (hash1 is not None) and (hash2 is not None):
        if hash1 == hash2:
            return True
        else:
            return False

    # Otherwise, assume datasets are not equal
    return False


def load_python_module(module_path):
    """Load and return the specified file at <module_path> as a Python module."""
    print (module_path)
    if module_path.endswith(".py"):
        # Import the python file directly
        module_dir_name, module_base_name = os.path.split(module_path)
        (module_name, extension) = os.path.splitext(module_base_name)
        print(module_dir_name)
        if not (module_dir_name in sys.path):
            print('added')
            sys.path.append(module_dir_name)
        print (module_name)
        module = __import__(module_name)
    else:
        # Import by package.module name (eg. "os.path")
        import importlib
        print (module_path)
        module = importlib.import_module(module_path)
    return module


def load_am_data_formatter(formatter_path):
    import imp
    foo = imp.load_source('data_formatter', formatter_path)
    return foo.DataFormatter()


def load_formatter(formatter_path, **formatter_params):
    """Load DataFormatter class from provided python module path.
     DataFormatter provides input/output formatting functions for neural network."""
    import imp
    if formatter_path.endswith(".py"):
        formatter_module = imp.load_source('data_formatter', formatter_path)
    else:
        formatter_module = imp.load_source(
            'data_formatter', os.path.join(formatter_path, 'data_formatter.py'))
    return formatter_module.DataFormatter(**formatter_params)


# For test cases, arguments may either be passed directly or through a config file.
# If a config file contains a value for a particular argument, this should be preferred over the current value
def parse_test_config(arg_dict, config_file):
    import json
    import alc_utils.config

    # Make sure ALC environment variables are available
    source_env_vars(alc_utils.config.env)

    # Read config file if available, or set config_dict to empty
    if config_file is not None:
        with open(config_file, 'r') as config_fp:
            config_dict = json.load(config_fp)
    else:
        config_dict = {}

    # For each argument key, overwrite value with the value in provided config file
    # If config file does not contain a particular key, keep the value specified in the arg_dict
    # Similar to python dictionary 'update' built-in, but also expands environmental variables in string values
    parsed_args = {}
    for key, value in viewitems(arg_dict):
        updated_val = config_dict.get(key, value)
        expanded_val = expandvars_recursive(updated_val)
        parsed_args[key] = expanded_val

    return parsed_args


def expandvars_recursive(value):
    """If provided value is a string, expand environment variables (using os.path.expandvars) recursively"""
    if isinstance(value, six.string_types):
        # Expand any environment variables recursively in-case any env variables reference other env variables
        expanded_val = os.path.expandvars(value)
        while not (expanded_val == value):
            value = expanded_val
            expanded_val = os.path.expandvars(value)
        value = expanded_val

    return value


def mkdir_p(path):
    """Function to mimic the functionality of the 'mkdir -p' terminal command.
    Recursively creates directories to the specified path, and does not complain if directory already exists."""
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def chown_r(path, uid, gid):
    """Recursively change the ownership of a directory and all files/sub-directories.
    Similar to 'chown -R <UID>:<GID>' on UNIX systems."""
    os.chown(path, uid, gid)
    for root, dirs, files in os.walk(path):
        for _dir in dirs:
            os.chown(os.path.join(root, _dir), uid, gid)
        for _file in files:
            os.chown(os.path.join(root, _file), uid, gid)


def chmod_r(path, permissions, add=False):
    """Recursively change the permissions of a directory and all files/sub-directories.
    Similar to 'chmod -R <PERMISSIONS>' on UNIX systems.
    If 'add' flag is true, will add to existing permissions similar to 'chmod -R +<PERMISSIONS>'"""
    if add:
        st = os.stat(path)
        os.chmod(path, st.st_mode | permissions)
        for root, dirs, files in os.walk(path):
            for _dir in dirs:
                full_path = os.path.join(root, _dir)
                st = os.stat(full_path)
                os.chmod(full_path, st.st_mode | permissions)
            for _file in files:
                full_path = os.path.join(root, _file)
                st = os.stat(full_path)
                os.chmod(full_path, st.st_mode | permissions)
    else:
        os.chmod(path, permissions)
        for root, dirs, files in os.walk(path):
            for _dir in dirs:
                os.chmod(os.path.join(root, _dir), permissions)
            for _file in files:
                os.chmod(os.path.join(root, _file), permissions)


def strip_trailing_separator(_path):
    """Normalize and strip trailing separator from specified path. Return updated path string."""
    _path = os.path.normpath(_path)
    if _path.endswith(os.path.sep):
        _path = _path[:-1]
    return _path


def strip_leading_separator(_path):
    """Normalize and strip leading separator from specified path. Return updated path string."""
    _path = os.path.normpath(_path)
    if _path.startswith(os.path.sep):
        _path = _path[1:]
    return _path


def check_directory_match(local_dir_fullname, desired_dir_name, expected_hash, ignore_hash_check=False):
    # Strip any trailing path separators
    local_dir_fullname = strip_trailing_separator(local_dir_fullname)
    desired_dir_name = strip_trailing_separator(desired_dir_name)

    # Check if directory names match
    if not (local_dir_fullname.endswith(desired_dir_name)):
        return False

    # If ignore_hash_check is set, name match is sufficient
    if ignore_hash_check:
        return True

    # If directory names match, use hash check to ensure data is correct. Otherwise, assume data is incorrect.
    if expected_hash is None:
        return False
    else:
        local_dir_hash = hash_directory(local_dir_fullname)
        # Don't need to re-download if hashes match. Otherwise, re-download directory.
        if expected_hash == local_dir_hash:
            print("Directory %s is already present at path %s and matches the expected hash" %
                  (desired_dir_name, local_dir_fullname))
            return True


def search_for_uri_locally(object_uri, additional_search_paths=[], ignore_hash_check=False):
    _object_dir = object_uri["directory"]
    hash_value = object_uri.get("hash", None)
    # Directory name is typically a relative path, but if it is absolute, check if it already exists.
    if os.path.isabs(_object_dir) and os.path.isdir(_object_dir):
        if check_directory_match(_object_dir, _object_dir, hash_value, ignore_hash_check):
            return _object_dir

    # Determine local directories to search where data may already be stored
    search_paths = alc_config.DOWNLOAD_SEARCH_DIRS
    for additional_path in additional_search_paths:
        if additional_path is not None:
            search_paths.append(additional_path)

    # For each search path, first check the path itself, then check all sub-directories
    for search_path in search_paths:
        # Check if this search path contains the data we want to download
        if check_directory_match(search_path, _object_dir, hash_value, ignore_hash_check):
            return search_path

        # Check if any subdirectory of this search path contains the data we want to download
        for root, dirs, files in os.walk(search_path):
            for dirname in dirs:
                dir_path = os.path.join(root, dirname)
                if check_directory_match(dir_path, _object_dir, hash_value, ignore_hash_check):
                    return dir_path

    # Matching directory not found in any search path
    return None


def dict_convert_key_case(orig_dict, desired_case):
    """Convert all string-type keys in the provided dictionary to the desired case (UPPER/lower)."""
    desired_case = desired_case.lower()
    new_dict = {}

    if desired_case == "upper":
        for key, value in viewitems(orig_dict):
            if isinstance(key, six.string_types):
                new_dict[key.upper()] = value
            else:
                new_dict[key] = value

    elif desired_case == "lower":
        for key, value in viewitems(orig_dict):
            if isinstance(key, six.string_types):
                new_dict[key.lower()] = value
            else:
                new_dict[key] = value

    else:
        raise ValueError(
            "Unrecognized case (%s). Allowed values are 'upper' or 'lower'." % desired_case)

    return new_dict


def load_params(param_file=None, param_dict=None, default_params=None):
    """Load parameters either from a file or directly from a provided dictionary, then add/update entries in default set
    of parameters."""
    # Initialize parameters to default
    if default_params is not None:
        params = default_params
    else:
        params = {}

    # Check if param file was defined or param dictionary was provided. If neither, return without loading
    if param_file is not None:
        print("Loading training parameters from file: %s" % param_file)
        with open(param_file, 'r') as param_fp:
            param_dict = json.load(param_fp)
    elif param_dict is not None:
        print("Loading training parameters from provided dictionary")
    else:
        print(
            "No parameter file or dictionary provided - using default training parameters.")
        return

    # Update existing parameter dictionary with loaded parameters
    params.update(param_dict)
    return params


def normalize_string(_str):
    """Normalize a string by converting to lower-case, removing whitespace, and converting '-' to '_'."""
    _normalized_str = _str.lower()
    _normalized_str = re.sub(r"\s+", "", _normalized_str)
    _normalized_str = _normalized_str.replace('-', '_')
    return _normalized_str


def load_training_datasets(training_data_uris,
                           data_formatter,
                           validation_data_uris=None,
                           testing_data_uris=None,
                           dataset_name=alc_config.training_defaults.DATASET_NAME,
                           useful_data_fraction=alc_config.training_defaults.USEFUL_DATA_FRACTION,
                           training_data_fraction=alc_config.training_defaults.TRAINING_DATA_FRACTION,
                           rng_seed=alc_config.training_defaults.RNG_SEED,
                           dataset_params=None,
                           **params):
    """This function handles loading desired data from data storage files, formatting the data into the desired
    format for use by other modules, and splitting the data into training/validation sets if desired. It performs the
    following tasks (in order):

        1) Loads the specified DataFormatter class.
        2) Searches for ROS bag files in the specified directories.
        3) Loads the desired topics from any bag files found (based on topic names from DataFormatter).
           If multiple topics are desired, but messages have different timestamps, then messages are paired to the
           closest matching timestamp.
        4) Formats the data according to the DataFormatter provided.
        5) Splits the formatted data into training/testing sets (if desired).

    Args:
        training_data_uris (dict(list)): List of URIs to search for training data files.
        data_formatter (DataFormatter): DataFormatter class to be used for formatting.
                                    Should be initialized before passing.
        dataset_name (str): Name of the DataInterpreter class to be used.
        validation_data_uris (Optional[dict(list)]): List of URIs to search for validation data files.
        testing_data_uris (Optional[dict(list)]): List of URIs to search for testing data files.
        useful_data_fraction (Optional[float]): Percentage of the complete dataset to keep as 'useful' data. Remaining
                                           data is discarded. This is often used for testing/debugging when working
                                           with large datasets.
        training_data_fraction (Optional[float]): Percentage of the useful dataset to use for training. Remaining data
                                             is used for validation. A value of 'None' indicates that no splitting
                                             should be performed.
        rng_seed (Optional[int]): Seed for splitting datasets into useful/training/testing and shuffling data.
        dataset_params (Optional[dict]): Keyword arguments to pass on to Dataset class.
        **params (Optional[dict]): Additional parameters to pass to PyTorch DataLoader

    Returns:
        (training_dataset, validation_dataset, testing_dataset)

        a) training_dataset:
        b) validation_dataset:
        c) testing_dataset:

    Raises:
        IOError: If no data files are found in the specified directories

    """
    import torch.utils.data
    from torch.utils.data.sampler import SubsetRandomSampler

    # Basic sanity checks
    if len(training_data_uris) == 0:
        raise IOError("load_data function received empty dataset URI list.")

    # Add standard keyword args for dataset and formatter
    if dataset_params is None:
        dataset_params = {}
    dataset_params.update({"useful_data_fraction": useful_data_fraction,
                           "rng_seed": rng_seed})

    dataset = load_dataset(
        training_data_uris, data_formatter, dataset_name, **dataset_params)

    validation_dataset = None
    if validation_data_uris is not None and len(validation_data_uris) > 0:
        print ('validation data uris {0}'.format(validation_data_uris))
        validation_dataset = load_dataset(
            validation_data_uris, data_formatter, dataset_name, **dataset_params)
    # else:
    # validation_file_dirs = training_file_dirs[-4:]
    # training_file_dirs = training_file_dirs[:-4]
    # validation_file_dirs = training_file_dirs
    # dataset = dataset_class(training_file_dirs, data_formatter, **dataset_params)
    # validation_dataset = dataset_class(validation_file_dirs, data_formatter, **dataset_params)

    testing_dataset = None
    if testing_data_uris is not None and len(testing_data_uris) > 0:
        print ('***********got testing data uris *************')
        testing_dataset = load_dataset(
            testing_data_uris, data_formatter, dataset_name, **dataset_params)
    # else:
    #    print ('***********no uris for testing data*************')
    #    raise RuntimeError("Testing data uris is none")

    # FIXME: Don't like how this is done. Should be improved.
    # Filter parameters relevant for PyTorch DataLoader
    data_loader_params = {"batch_size": params.get("batch_size", alc_config.training_defaults.BATCH_SIZE),
                          "shuffle": params.get("shuffle", alc_config.training_defaults.SHUFFLE)}
    optional_params = ["sampler", "batch_sampler", "num_workers", "collage_fn",
                       "pin_memory", "drop_last", "timeout", "worker_init_fn"]
    for param_name in optional_params:
        param_val = params.get(param_name, None)
        if param_val is not None:
            data_loader_params[param_name] = param_val

    # If desired, randomly split dataset into training and validation subsets.
    # This is only supported for certain Dataset types (Check and raise exception if unsupported).
    print(
        "***********training data fraction {0}".format(training_data_fraction))
    if (training_data_fraction is not None) and (training_data_fraction < 1.0):
        # Feature-compatibility checks
        if isinstance(dataset, torch.utils.data.IterableDataset):
            raise RuntimeError("Random splitting of dataset into training & validation sub-sets (specified with 'TRAINING_DATA_FRACTION' parameter) "
                               "is not supported for IterableDatasets. Must be a standard (map-style) Dataset.")
        elif validation_dataset is not None:
            raise RuntimeError("Cannot use random splitting of dataset into training & validation sub-sets (specified with 'TRAINING_DATA_FRACTION' parameter) "
                               "when explicit validation data set has been provided.")

        # Get length of complete dataset
        try:
            dataset_len = len(dataset)
        except TypeError:
            raise RuntimeError("Random splitting of dataset into training & validation sub-sets (specified with 'TRAINING_DATA_FRACTION' parameter) "
                               "is not supported for datasets without a valid 'len' function.")

        print('******************************************')
        print('******************************************')
        indices = list(range(dataset_len))
        training_len = int(round(dataset_len * training_data_fraction))
        validation_len = dataset_len - training_len
        print("***********training_len  {0}".format(training_len))
        print("***********validation_len  {0}".format(validation_len))
        if data_loader_params.get('shuffle', False):
            import numpy as np
            np.random.seed(rng_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[:
                                             training_len], indices[training_len:]
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        import copy
        data_loader_param_new = copy.deepcopy(data_loader_params)
        del data_loader_param_new['shuffle']

        training_data = torch.utils.data.DataLoader(
            dataset, sampler=train_sampler, **data_loader_param_new)
        validation_data = torch.utils.data.DataLoader(
            dataset, sampler=valid_sampler, **data_loader_param_new)

        print("***********len training data  {0}".format(len(training_data)))
        print(
            "***********len validation data  {0}".format(len(validation_data)))

        #training_dataset, validation_dataset = torch.utils.data.random_split(dataset, [training_len, validation_len])
    else:
        training_dataset = dataset
        training_data = torch.utils.data.DataLoader(
            training_dataset, **data_loader_params)
        validation_data = None
        if validation_dataset is not None:
            print ('***********using validation data as validation data *************')
            validation_data = torch.utils.data.DataLoader(
                validation_dataset, **data_loader_params)
        else:
            print ('***********using training data as validation data *************')
            validation_data = torch.utils.data.DataLoader(
                training_dataset, **data_loader_params)

    testing_data = None
    import copy
    data_loader_param_test = copy.deepcopy(data_loader_params)
    data_loader_param_test['shuffle'] = False
    if testing_dataset is not None:
        testing_data = torch.utils.data.DataLoader(
            testing_dataset, **data_loader_param_test)

    # FIXME: For now, use validation data as testing data if no testing data available.
    #  This should be removed later
    if testing_data is None:
        print('testing data is none. using validation data as testing dataa')
        testing_data = validation_data

    return training_data, validation_data, testing_data


def load_dataset(data_uris, data_formatter, dataset_name, **dataset_params):
    import alc_utils.datasets
    # Load desired Dataset class, then use this class to load actual data
    data_dirs = uris_to_dir_list(data_uris)
    dataset_class = alc_utils.datasets.load_dataset_class(dataset_name)
    return dataset_class(data_dirs, data_formatter, **dataset_params)


def load_data(data_uris,
              data_formatter,
              dataset_name=alc_config.training_defaults.DATASET_NAME,
              dataset_params=None,
              **params):
    import torch.utils.data

    # Basic sanity checks
    if len(data_uris) == 0:
        raise IOError("load_data function received empty dataset URI list.")

    # Add standard keyword args for dataset and formatter
    if dataset_params is None:
        dataset_params = {}

    dataset = load_dataset(data_uris, data_formatter,
                           dataset_name, **dataset_params)

    # FIXME: Don't like how this is done. Should be improved.
    # Setup required PyTorch DataLoader parameters, then filter for optional parameters
    data_loader_params = {"batch_size": params.get("batch_size", alc_config.training_defaults.BATCH_SIZE),
                          "shuffle": params.get("shuffle", alc_config.training_defaults.SHUFFLE)}
    opt_param_names = ["sampler", "batch_sampler", "num_workers", "collage_fn",
                       "pin_memory", "drop_last", "timeout", "worker_init_fn"]
    for param_name in opt_param_names:
        param_val = params.get(param_name, None)
        if param_val is not None:
            data_loader_params[param_name] = param_val

    # Invoke PyTorch DataLoader
    return torch.utils.data.DataLoader(dataset, **data_loader_params)


def uris_to_dir_list(dataset_uris):
    """This function takes a list of dataset URIs, finds the local directory corresponding to each URI, then
    returns the list of local directories where the data can be found.
    Assumes all data URIs are available locally and will throw an exception otherwise."""
    # FIXME: This is for compatibility with older API. Remove when no longer needed.
    if isinstance(dataset_uris[0], six.string_types):
        # Older API specified dataset as list of directories (strings) containing desired data.
        # Current API specifies dataset as list of dataset URI's (dictionaries) to desired data.
        # If strings are provided, use list of paths to datasets directly
        dir_list = dataset_uris
    elif isinstance(dataset_uris[0], dict):
        # Search for local directories containing each data URI
        # If no local directory found, raise an error. Don't attempt to download.
        dir_list = []
        for data_uri in dataset_uris:
            # FIXME: Shouldn't skip hash check
            local_dir = search_for_uri_locally(
                data_uri, ignore_hash_check=True)
            if local_dir is None:
                raise IOError(
                    "load_data function could not find local copy of dataset URI: %s" % str(data_uri))
            else:
                dir_list.append(local_dir)
    else:
        raise TypeError("load_data function received unexpected type for data URI list argument. Got type %s. "
                        "Expected string or dict." % type(dataset_uris[0]))

    # Sort directories for reproducibility.
    dir_list.sort()

    return dir_list


def filter_parameters(param_dict, filter_map):
    """Function which filters a dictionary of parameters to only contain those listed in the filter_map dictionary.
    Also renames parameters based on map."""
    filtered_param_dict = {}
    for orig_name, new_name in iteritems(filter_map):
        try:
            param_val = param_dict[orig_name]
        except KeyError:
            continue

        filtered_param_dict[new_name] = param_val

    return filtered_param_dict


def get_custom_dict(orig_dict, recursive=True):
    """Returns a copy of the provided dictionary where non-serializable objects are replaced with the string 'CUSTOM'.
    By default, operates recursively if input contains other dictionaries. Otherwise, assumes they are serializable.
    For list objects, if list contains any non-serializable objects, entire list is replaced with string 'CUSTOM'."""
    new_dict = {}
    for name, value in iteritems(orig_dict):
        if isinstance(value, (int, float, bool, str)):
            new_dict[name] = value
        elif isinstance(value, dict):
            if recursive:
                new_dict[name] = get_custom_dict(value)
            else:
                new_dict[name] = value
        elif isinstance(value, list):
            new_dict[name] = value
            for item in value:
                if not isinstance(item, (int, float, bool, str)):
                    new_dict[name] = 'CUSTOM'
                    break
        else:
            new_dict[name] = 'CUSTOM'
    return new_dict
