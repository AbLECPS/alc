# !/usr/bin/env python
"""Utility class for executing Neural Network models and their associated Assurance Monitors"""
# Author: Charlie Hartsell <charles.a.hartsell@vanderbilt.edu>
from __future__ import print_function

import os
import alc_utils.network_interface
import alc_utils.assurance_monitor
import alc_utils.config as alc_config


class NetworkPredictor:
    """ **DEPRECATED**: Use NetworkInterface instead. Remove this class when existing usages have been updated.

    This class provides an interface for loading trained neural network models (and their corresponding assurance
        monitors), and using these models to make predictions. """

    def __init__(self, ml_library_adapter_path=alc_config.ML_LIBRARY_ADAPTER_PATH):
        self.network_interface = alc_utils.network_interface.NetworkInterface(
            ml_library_adapter_path)
        self.assurance_monitor = None

    def load_model(self, model_dir, use_assurance=False, assurance_monitor_path=None):
        """ This function loads the specified neural network model and the corresponding DataFormatter (as well as
        a trained assurance monitor, if desired)

        Args:
            model_dir (str): Directory where network model is located. At a minimum, this directory should
                contain the following files:
                    1) "model.keras" - Trained network model
                    2) "data_formatter.py" - Network DataFormatter class
            use_assurance (Optional[bool]): If True, function will also try to load the corresponding
                assurance monitor
            assurance_monitor_path (Optional[str]): Path to a saved AssuranceMonitor corresponding to this network.

        Returns:
            None

        Raises:
            IOError: If the required files do not exist in the provided directory.
        """
        # Determine relevant filenames and check validity
        if assurance_monitor_path is None:
            assurance_monitor_path = os.path.join(
                model_dir, "assurance_monitor.pkl")
        if use_assurance:
            if not os.path.isfile(assurance_monitor_path):
                raise IOError(
                    'Assurance monitor file %s is not a valid file' % assurance_monitor_path)

        # Load model
        self.network_interface.load(model_dir)

        # Load assurance monitor if desired
        if use_assurance:
            self.assurance_monitor = alc_utils.assurance_monitor.AssuranceMonitor.load(
                assurance_monitor_path)

    def get_input_shape(self):
        return self.network_interface.get_input_shape()

    def predict(self, input_data, *args, **kwargs):
        return self.network_interface.predict(input_data, *args, **kwargs)

    def run_assurance(self, _raw_input, predicted_output):
        # Format input with DataFormatter loaded by NetworkInterface
        formatted_input = self.network_interface.formatter.format_input(
            _raw_input)

        assurance_result = None
        if self.assurance_monitor is not None:
            assurance_result = self.assurance_monitor.evaluate(
                formatted_input, predicted_output)

        return assurance_result

    # FIXME: This function should be moved inside NetworkInterface. Leave wrapper here for compatibility
    #        Assurance Monitoring code may stay here?
    def run(self, _raw_input, **kwargs):
        """This function takes in unformatted inputs, formats the data using the loaded DataFormatter,
        then inferences with the NN model and returns the predicted value and assurance monitor result

        Args:
            _raw_input: Unformatted input data.
                The network's DataFormatter is used to format this data before prediction with the trained model.

        Returns:
            (result_code, predicted_output, assurance_result)

            a) result_code (int): 0 if successful. <0 otherwise.
            b) predicted_output (numpy array): Predicted output from the network model.
            c) assurance_result: Result from assurance monitor evaluate function. See assurance monitor docs for details
        """
        # Sanity checks
        if self.network_interface.formatter is None:
            return -1, None, None

        # Use trained network for prediction
        predicted_output = self.network_interface.predict(_raw_input, **kwargs)

        # Run assurance monitor
        assurance_result = self.run_assurance(_raw_input, predicted_output)

        return 0, predicted_output, assurance_result
