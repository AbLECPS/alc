import alc_utils.common as alc_common
import alc_utils.config as alc_config
import alc_utils.datasets

import argparse
import os
import json
import numpy as np

import torch

import rosbag
from std_msgs.msg import Int32, String


def test_hash_directory():
    real_dir_hash = 'da39a3ee5e6b4b0d3255bfef95601890afd80709'
    folder_name = 'folder_to_hash'
    path_to_folder = ''
    if not os.path.isdir(folder_name):
        path_to_folder = os.path.join(os.getcwd(), folder_name)
        os.mkdir(path_to_folder)
    dir_hash = alc_common.hash_directory(folder_name)
    if len(path_to_folder) > 0:
        os.rmdir(path_to_folder)
    assert(real_dir_hash == dir_hash)


def test_generate_pkl_file():
    bag = rosbag.Bag('test.bag', 'w')

    try:
        s = String()
        s.data = 'foo'

        i = Int32()
        i.data = 42

        bag.write('chatter', s)
        bag.write('numbers', i)
    finally:
        bag.close()

    pkl_fn = alc_common.generate_pkl_file('./test.bag')
    os.remove("./test.bag")
    os.remove("./test.pkl")
    assert(pkl_fn == 'test.pkl')


def test_source_env_vars():
    env_dict = {
        "ALC_COMMON": "1",
        "ALC_UTILS": "2",
        "ALC_VARS": "3"
    }

    alc_common.source_env_vars(env_dict)

    for key in env_dict:
        assert(key in os.environ)
        assert(os.environ[key] is env_dict[key])


def test_filter_parameters():
    param_dict = {
        "In_Dict": 1,
        "Filter_Me": 1,
        "Filter_Me_Too": 1
    }

    filter_map = {
        "In_Dict": "Input_Dictionary",
        "Filter_Me": "You're Filtered",
        "Filter_Me_Too": "Filtered Too"
    }

    filtered = alc_common.filter_parameters(param_dict, filter_map)
    for idx, key in enumerate(filter_map):
        val = filter_map[key]
        new_keys = filtered.keys()
        assert(val in new_keys)


def test_normalize_string():
    strings_to_test = ['TEST-this', 'test white space',
                       'test-this and WHITE space', 'TESTTHIS', 'lowercase']

    expected_strings = ['test_this', 'testwhitespace',
                        'test_thisandwhitespace', 'testthis', 'lowercase']
    for s in strings_to_test:
        assert(alc_utils.common.normalize_string(s) in expected_strings)


def test_get_complete_training_metadata(network_metadata):
    training_data_metadata = alc_common.get_complete_training_metadata(
        network_metadata)
    for data in training_data_metadata:
        assert(data.has_key('hash'))
        assert(data['hash'] is not None)


def test_dataset_metadata_is_equal(network_metadata):
    training_data_metadata = alc_common.get_complete_training_metadata(
        network_metadata)
    prev_data = 0
    next_data = 0
    for idx, data in enumerate(training_data_metadata):
        next_data = data
        if idx == 1:
            break
        assert(alc_common.dataset_metadata_is_equal(data, data) is True)
        prev_data = data
    assert(alc_common.dataset_metadata_is_equal(prev_data, next_data) is False)


def test_load_python_module():
    assert(alc_common.load_python_module(
        '../../alc_utils/ml_library_adapters/keras_library_adapter.py').__name__ == 'keras_library_adapter')
    assert(alc_common.load_python_module(
        '../../alc_utils/ml_library_adapters/pytorch_semseg_adapter.py').__name__ == 'pytorch_semseg_adapter')


def test_parse_test_config(monkeypatch):
    with open('data.json', 'w') as outfile:
        data = {}
        data['upload'] = True
        data['training_params'] = {"mean": 100, "threshold": 450}
        json.dump(data, outfile)
    monkeypatch.setattr('argparse._sys.argv', ['python'])
    DEFAULT_NETWORK_MODEL_DIR = os.path.join(
        alc_config.ALC_HOME, "alc_utils/network_models/dave2")
    DEFAULT_SFTP_CONFIG_FILE = os.path.join(
        alc_config.ALC_HOME, "alc_utils/test/test_sftp_config.json")
    DEFAULT_OUTPUT_DIRECTORY = "/tmp/alc/network_training_test"

    parser = argparse.ArgumentParser(
        description='Test ALC utilities download, network trainer, and data loader functions')
    parser.add_argument(
        '--training_uris', help='JSON file containing uri references to desired training data.')
    parser.add_argument(
        '--validation_uris', help='JSON file containing uri references to desired validation data.', default=None)
    parser.add_argument(
        '--testing_uris', help='JSON file containing uri references to desired testing data.', default=None)
    parser.add_argument('--training_params',
                        help='Dictionary of training parameters.', default={})
    parser.add_argument('--network_model_dir',
                        help='Directory containing Keras network model and data formatter to be loaded.',
                        default=DEFAULT_NETWORK_MODEL_DIR)
    parser.add_argument(
        '--sftp_config', help='SFTP config file to use for test', default=DEFAULT_SFTP_CONFIG_FILE)
    parser.add_argument(
        '--output_dir', help='Output directory to save the test output.', default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument(
        '--config_file', help='JSON file specifying desired arguments.', default="data.json")
    parser.add_argument(
        '--output_hash_file', help='Text file containing hash value of a correct result.', default=None)
    parser.add_argument(
        '--upload', help='Flag to upload results.', action='store_true')
    args = parser.parse_args()

    parsed_args = alc_common.parse_test_config(vars(args), args.config_file)
    os.remove("data.json")
    assert(parsed_args["upload"] == True)
    assert(parsed_args["training_params"]['threshold'] == 450)
    assert(parsed_args["training_params"]['mean'] == 100)


def test_strip_trailing_separator():
    path = "/some/thing/else/"
    res = "/some/thing/else"
    assert(alc_common.strip_trailing_separator(path) == res)


def test_strip_leading_separator():
    path = "/some/thing/else"
    res = "some/thing/else"
    assert(alc_common.strip_leading_separator(path) == res)


def test_check_directory_match():
    exp_hash = 'da39a3ee5e6b4b0d3255bfef95601890afd80709'
    folder_name = 'folder_to_hash/'
    local_dir_fullname = os.path.join(os.getcwd(), 'folder_to_hash')
    os.mkdir(local_dir_fullname)
    assert(alc_common.check_directory_match(
        'folder_to_hash/', folder_name, exp_hash))
    os.rmdir(local_dir_fullname)


def test_search_for_uri_locally():
    exp_hash = 'da39a3ee5e6b4b0d3255bfef95601890afd80709'
    local_dir_fullname = os.path.join(os.getcwd(), 'folder_to_hash')
    os.mkdir(local_dir_fullname)
    object_uri = {
        "directory": local_dir_fullname,
        "hash": exp_hash
    }
    assert(alc_common.search_for_uri_locally(object_uri) != None)
    os.rmdir(local_dir_fullname)
    assert(alc_common.search_for_uri_locally(object_uri) == None)


def test_load_data_dave2(create_dave2_dataset, dave2_data_formatter):
    batch_size = 2
    size_of_input = torch.Size([batch_size, 66, 200, 3])
    size_of_output = torch.Size([batch_size, 1])
    data_uris = ["file://some/dummy/location"]

    data_loader = alc_common.load_data(data_uris, dave2_data_formatter)
    assert(data_loader is not None)
    for xb, yb in data_loader:
        assert(xb.size() == size_of_input)
        assert(yb.size() == size_of_output)
        break


def test_load_data_semseg(create_semseg_dataset, semseg_data_formatter):
    batch_size = 2
    size_of_input = torch.Size([batch_size, 100, 512, 3])
    size_of_output = torch.Size([batch_size, 100, 512, 3])
    data_uris = ["file://some/dummy/location"]

    data_loader = alc_common.load_data(data_uris, semseg_data_formatter)
    assert(data_loader is not None)
    for xb, yb in data_loader:
        assert(xb.size() == size_of_input)
        assert(yb.size() == size_of_output)
        break


def test_load_params():
    params_dict = {
        "learning_rate": 0.01,
        "epsilon": 0.001
    }
    with open('data.json', 'w') as outfile:
        data = {}
        data['upload'] = True
        data['training_params'] = {"mean": 100, "threshold": 450}
        json.dump(data, outfile)
    params = alc_common.load_params(
        param_file="data.json", param_dict=params_dict)
    os.remove("data.json")
    assert(params == {'upload': True, 'training_params': {
           "mean": 100, "threshold": 450}})
    params = alc_common.load_params(param_file=None, param_dict=params_dict)
    assert(params == params_dict)


def test_load_training_datasets(create_semseg_dataset, semseg_data_formatter):
    batch_size = 2
    size_of_input = torch.Size([batch_size, 100, 512, 3])
    size_of_output = torch.Size([batch_size, 100, 512, 3])
    data_uris = ["file://some/dummy/location"]

    training_data, validation_data, testing_data = alc_common.load_training_datasets(
        data_uris, semseg_data_formatter, data_uris, data_uris)

    assert(training_data is not None)
    for xb, yb in training_data:
        assert(xb.size() == size_of_input)
        assert(yb.size() == size_of_output)
        break

    assert(validation_data is not None)
    for xb, yb in validation_data:
        assert(xb.size() == size_of_input)
        assert(yb.size() == size_of_output)
        break

    assert(testing_data is not None)
    for xb, yb in testing_data:
        assert(xb.size() == size_of_input)
        assert(yb.size() == size_of_output)
        break
