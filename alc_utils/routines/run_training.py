#!/usr/bin/env python
# Author: Charlie Hartsell <charles.a.hartsell@vanderbilt.edu>
"""This module provides common routines for training Neural Networks. (Supervised Learning)"""
from __future__ import print_function

from alc_utils import file_uploader, network_interface
from alc_utils import common as alc_common
from alc_utils import config as alc_config
import os
import json
import yaml


# FIXME: Complete dataset is already downloaded when this function is called,
#        but only certain directories are needed if training from a parent model
def run_training(param_dict,
                 training_data_uri_list,
                 data_dirs,
                 lec_model_dir,
                 output_dir,
                 parent_model_dir=None,
                 validation_data_uri_list=None,
                 testing_data_uri_list=None,
                 evaluation_only=False):
    # Convert param dict to lower-case keys for consistency
    param_dict = alc_common.dict_convert_key_case(param_dict, "lower")

    # Check if training/testing/validation datasets were provided, and normalize accordingly
    training_data_uri_list = normalize_uri_list(training_data_uri_list)
    if (validation_data_uri_list is None) and (testing_data_uri_list is None):
        print("No validation or testing datasets provided. Will continue with training dataset only.")
    elif (validation_data_uri_list is None) and not (testing_data_uri_list is None):
        testing_data_uri_list = normalize_uri_list(testing_data_uri_list)
        td = param_dict.get('training_data_fraction', 1.0)
        if (td == 1.0):
            print(
                "No validation dataset provided. Will using testing dataset for test and validation.")
            validation_data_uri_list = testing_data_uri_list
    elif (validation_data_uri_list is not None) and (testing_data_uri_list is None):
        print("No testing dataset provided. Will using validation dataset for test and validation.")
        validation_data_uri_list = normalize_uri_list(validation_data_uri_list)
        testing_data_uri_list = validation_data_uri_list
    else:
        validation_data_uri_list = normalize_uri_list(validation_data_uri_list)
        testing_data_uri_list = normalize_uri_list(testing_data_uri_list)

    # Ensure output directory exists
    alc_common.mkdir_p(output_dir)

    # Load LEC network interface
    ml_library_adapter = param_dict.get(
        "ml_library", alc_config.ML_LIBRARY_ADAPTER_PATH)
    nn_interface = network_interface.NetworkInterface(ml_library_adapter)

    # Run Training
    model_eval, training_history = nn_interface.train(lec_model_dir, training_data_uri_list, output_dir,
                                                      validation_data_uris=validation_data_uri_list,
                                                      testing_data_uris=testing_data_uri_list, param_dict=param_dict)

    # FIXME: currently custom functions are saved as 'CUSTOM'
    param_dict_custom = alc_common.get_custom_dict(param_dict)

    # FIXME: Parameters are also written to the network metadata. Don't really need to write them twice
    # Write parameters to file into trained model directory

    param_filename = os.path.join(output_dir, 'params.yml')
    with open(param_filename, 'w') as yaml_fp:
        yaml.dump(param_dict_custom, yaml_fp, default_flow_style=False)

    # FIXME: Is this necessary? Could be when custom architectures etc. are used
    # Copy LEC model file to trained model directory
    # shutil.copy2(lec_model_path, output_dir)

    # Print directory where training results can be found
    print("Trained model directory: {}".format(output_dir))

    # Create and return storage metadata
    uploader = file_uploader.FileUploader()

    storage_metadata = uploader.upload_with_params(
        output_dir, param_dict_custom)

    # Add training result info to metadata and return
    storage_metadata["results"] = {
        "model_evaluation": model_eval, "training_history": training_history}

    return storage_metadata


def run_evaluation(param_dict,
                   eval_data_uri_list,
                   lec_model_dir,
                   output_dir):
    # Convert param dict to lower-case keys for consistency and normalize dataset URI's accordingly
    param_dict = alc_common.dict_convert_key_case(param_dict, "lower")
    eval_data_uri_list = normalize_uri_list(eval_data_uri_list)

    # Ensure output directory exists
    alc_common.mkdir_p(output_dir)

    # Load NN model
    ml_library_adapter = param_dict.get(
        "ml_library", alc_config.ML_LIBRARY_ADAPTER_PATH)
    nn_model = network_interface.NetworkInterface(ml_library_adapter)
    nn_model.load(lec_model_dir)

    # Load evaluation dataset
    eval_data = alc_common.load_data(
        eval_data_uri_list, nn_model.formatter, **param_dict)

    # Run Training
    eval_results = nn_model.evaluate(eval_data, param_dict)

    # Write evaluation results to file in output directory
    eval_filename = os.path.join(output_dir, 'evaluation_results.json')
    with open(eval_filename, 'w') as eval_fp:
        json.dump(eval_results, eval_fp)

    # FIXME: currently custom functions are saved as 'CUSTOM'
    param_dict_custom = alc_common.get_custom_dict(param_dict)
    # Write parameters to file into trained model directory
    param_filename = os.path.join(output_dir, 'params.yml')
    with open(param_filename, 'w') as yaml_fp:
        yaml.dump(param_dict_custom, yaml_fp, default_flow_style=False)

    # Print directory where training results can be found
    print("Evaluation output directory: %s" % output_dir)

    # Create and return storage metadata
    uploader = file_uploader.FileUploader()
    storage_metadata = uploader.upload_with_params(
        output_dir, param_dict_custom)

    return storage_metadata


def normalize_uri_list(uri_list):
    # Ideally it should be list of dictionaries, but a json string or a single dictionary are also accepted.
    if type(uri_list) is str:
        uri_list = json.loads(uri_list)
    if type(uri_list) is dict:
        uri_list = [uri_list]
    if not (type(uri_list) is list):
        raise TypeError(
            "Specified dataset storage metadata has unexpected type (%s)" % str(type(uri_list)))
    return uri_list
