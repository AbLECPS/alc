# !/usr/bin/env python
"""Utility class for training a Neural Network model, given a set of training data."""
# Author: Charlie Hartsell <charles.a.hartsell@vanderbilt.edu>
from __future__ import print_function

import alc_utils.network_interface
import alc_utils.config as alc_config
import alc_utils.assurance_monitor


# FIXME: **DEPRECATED** Remove this class when no longer needed
class NetworkTrainer:
    """ **DEPRECATED** This class provides an interface for training empty neural network architecture, or continuing
    training from an existing neural network model. """

    def __init__(self, ml_library_adapter=alc_config.ML_LIBRARY_ADAPTER_PATH):
        self.network_interface = alc_utils.network_interface.NetworkInterface(
            ml_library_adapter)

    def train_network(self, model_dir, data_uri_list, output_dir, param_file=None, param_dict=None):
        """ **DEPRECATED** Simple pass through to NetworkInterface."""
        # Pass-through to NetworkInterface
        model_eval, training_result = self.network_interface.train(
            model_dir, data_uri_list, output_dir, param_file=None, param_dict=None)
        return 0, model_eval, training_result
