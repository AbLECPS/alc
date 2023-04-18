#!/usr/bin/env python
# Author: Charlie Hartsell <charles.a.hartsell@vanderbilt.edu>
"""This python script provides functions for training an Assurance Monitor
Each assurance monitor is tied to a particular, pre-trained Neural Network"""
from __future__ import print_function

import alc_utils.assurance_monitor
from alc_utils import config as alc_config
from alc_utils import common as alc_common
import os
import json


# FIXME: Is it required/beneficial to have a separate orchestrator for Assurance Monitor training,
#        or can the functionality of this script be moved to the AssuranceMonitor base class?
def run_ood_detector_training(param_dict,
                                   training_data_dirs,
                                   lec_model_dir,
                                   output_dir,
                                   am_data_formatter_path=None,
                                   validation_data_dirs=None,
                                   testing_data_dirs = None):
    # Convert param dict to lower-case keys for consistency, and load any defaults
    param_dict = alc_common.dict_convert_key_case(param_dict, "lower")
    param_dict = alc_common.load_params(param_dict=param_dict,
                                        default_params=alc_config.training_defaults.var_dict_lower)


    # Load appropriate DataFormatter. Inherit from trained LEC if no formatter specified
    if am_data_formatter_path is None:
        data_formatter_path = os.path.join(lec_model_dir, "data_formatter.py")
    else:
        data_formatter_path = am_data_formatter_path

    input_shape = param_dict.get("input_shape", None)

    # Load and format data.
    formatter_params = {"input_shape": input_shape, "mode":"TRAINING"}
    
    data_formatter = alc_common.load_formatter(data_formatter_path, **formatter_params)

    training_data, validation_data, testing_data = alc_common.load_training_datasets(training_data_dirs, 
                                                                                        data_formatter,
                                                                                        validation_data_uris=validation_data_dirs,
                                                                                        testing_data_uris=testing_data_dirs,
                                                                                        **param_dict)
    
    # Create and train assurance monitor
    print("Initializing Assurance Monitor and starting training.")
    
    assurance_monitor_type = param_dict.get("assurance_monitor_type",
                                            alc_config.training_defaults.ASSURANCE_MONITOR_TYPE).lower()
    assurance_monitor_type = param_dict.get("type", assurance_monitor_type).lower()
    monitor = alc_utils.assurance_monitor.load_assurance_monitor(assurance_monitor_type)
    _save_dir = monitor._make_unique_subdir(output_dir)
    
    monitor.train(training_data, validation_data,testing_data,_save_dir, **param_dict)

    # Save assurance monitor
    print("Saving trained assurance monitor...")
    save_dir = monitor.save(_save_dir, data_formatter_path=data_formatter_path, make_unique_subdir=False)
    print("Saved trained assurance monitor in output directory: %s" % save_dir)

    return output_dir
