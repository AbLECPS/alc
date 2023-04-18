# !/usr/bin/env python
"""NetworkInterface provides a platform-agnostic interface for manipulating neural network models"""
# Author: Charlie Hartsell <charles.a.hartsell@vanderbilt.edu>
from __future__ import print_function

import os
import json
import shutil
import alc_utils.assurance_monitor
import alc_utils.config as alc_config
import alc_utils.common as alc_common
import alc_utils.ml_library_adapters

_NETWORK_METADATA_FILENAME = "model_metadata.json"
_DATA_FORMATTER_FILENAME = "data_formatter.py"


class NetworkInterface:
    """Provides a platform-agnostic interface for manipulating neural network models."""

    def __init__(self, library_adapter=alc_config.ML_LIBRARY_ADAPTER_PATH):
        """Initialize the NetworkInterface and load an instance of the desired LibraryAdapter class.
        LibraryAdapter class must implement the required interfaces to a particular learning framework
        (eg. Keras, TensorFlow, PyTorch, etc.)."""
        # Variable init
        self._lib_adapter = alc_utils.ml_library_adapters.load_library_adapter(
            library_adapter)
        self.formatter = None
        self._formatter_path = None
        self._lec_model_path = None
        self._metadata = {"state": "empty"}

    def get_metadata(self):
        """Return the current network metadata in a dictionary."""
        return self._metadata

    def get_input_shape(self, *args, **kwargs):
        """Return the input shape of the currently loaded network model"""
        return self._lib_adapter.get_input_shape(*args, **kwargs)

    def load(self, model_dir, params=None):
        """Load a network model (trained or un-trained) and metadata from the specified directory.
        Also load and initialize the corresponding DataFormatter.

        Args:
            model_dir (str): Directory containing network model and associated files.
            params (dict): Dictionary of parameters specified as name, value pairs.

        Returns:
            3-tuple containing:
                network metadata (dict): Metadata about the loaded network model
                network model path (str): Filepath to the loaded network model
                formatter path (str): Filepath to the loaded DataFormatter class
        """
        # Identify relevant files and load network
        self._metadata, self._lec_model_path, self._formatter_path = self.find_files(
            model_dir)
        self._lib_adapter.load_model(model_dir, params=params)

        # If we are loading a trained model, should have an associated network metadata file.
        # Otherwise, this must be an untrained network architecture.
        # Update or initialize metadata as appropriate
        if self._metadata:
            # Support older metadata files before "state" parameter was added
            if self._metadata.get("state", None):
                self._metadata["state"] = "trained"
        else:
            self._metadata = {"state": "untrained",
                              "input_shape": self.get_input_shape()}

        # Load formatter
        self.formatter = alc_common.load_formatter(
            self._formatter_path, **self._metadata)

        return self._metadata, self._lec_model_path, self._formatter_path

    def save(self, save_dir, history=None):
        """Saves the currently loaded NN model, DataFormatter, metadata, and any training history
        to the specified directory.

        Args:
            save_dir (str): Directory where model should be saved. This function assumes the directory already exists.
            history (dict): History of training results (ie. loss, accuracy, etc. over each epoch).

        Returns:
            None
        """
        # Make sure directory exists
        alc_common.mkdir_p(save_dir)

        # Save the trained model
        model_rel_path = self._lib_adapter.save_model(save_dir)

        # Save copy of formatter and the architecture (if available) to output directory
        data_formatter_path = self._formatter_path
        formatter_save_path = os.path.join(save_dir, _DATA_FORMATTER_FILENAME)
        shutil.copy(data_formatter_path, formatter_save_path)
        if self._lec_model_path.endswith(".py"):
            architecture_save_path = os.path.join(
                save_dir, os.path.basename(self._lec_model_path))
            shutil.copy(self._lec_model_path, architecture_save_path)

        # Save the training history (record of loss/metrics at each epoch) if available
        if history:
            history_path = os.path.join(save_dir, "history.json")
            with open(history_path, 'w') as json_fp:
                json.dump(history, json_fp)

        # Update and save metadata
        self._metadata["model_relative_path"] = model_rel_path
        self._metadata["data_formatter_relative_path"] = _DATA_FORMATTER_FILENAME
        metadata_path = os.path.join(save_dir, "model_metadata.json")
        with open(metadata_path, 'w') as json_fp:
            json.dump(self._metadata, json_fp)

        # FIXME: This is a temporary workaround due to flawed assumptions in the assurance monitor training process.
        #   Should be removed when possible.
        # Save a copy of the Assurance Monitor network architecture to the output directory
        lec_model_dir = os.path.dirname(os.path.realpath(data_formatter_path))
        am_net_path = os.path.join(lec_model_dir, "am_net.py")
        am_net_save_path = os.path.join(save_dir, "am_net.py")
        if os.path.exists(am_net_path):
            shutil.copy(am_net_path, am_net_save_path)

    def predict(self, input_data, format_input=True, *args, **kwargs):
        """Use the currently loaded network model for forward inference.
        Wrapper around LibraryAdapter 'predict' function.

        Args:
            input_data: Input data to the Neural Network. Can be raw data or pre-formatted.
            format_input (optional[bool]): If true, format input_data before passing to NN.

        Returns:
            Output from NN. Type is variable based on the library used and currently loaded NN model.
        """
        if format_input:
            formatted_input = self.formatter.format_input(input_data)
        else:
            formatted_input = input_data
        return self._lib_adapter.predict(formatted_input, *args, **kwargs)

    def _train(self, training_data, testing_data, training_params):
        return self._lib_adapter.train(training_data, testing_data, training_params)

    def evaluate(self, evaluation_data, evaluation_params):
        """Evaluate (calculate loss, accuracy, etc.) currently loaded neural network against provided evaluation
        dataset. Wrapper around LibraryAdapter 'evaluate' function.

        Args:
            evaluation_data (tuple[iterable]): Set of data points for evaluation.
                    2-tuple including (input_data, output_labels).
            evaluation_params (dict): Parameters to use for evaluation.

        Returns:
            evaluation_metrics (dict): Dictionary of values calculated for each evaluation metric.
        """
        return self._lib_adapter.evaluate(evaluation_data, evaluation_params)

    def train(self, model_dir, training_data_uris, output_dir, param_file=None, param_dict=None,
              validation_data_uris=None, testing_data_uris=None):
        """This function will train the currently loaded neural network on the provided dataset.

        Args:
            model_dir (str): Directory where LEC model and related files are located.
            training_data_uris (dict(list)): List of data URIs specifying dataset to be used for training.
            output_dir (str): Directory where trained LEC and related files should be saved.
            param_file (Optional[str]): Path to a json file to load training hyper-parameters from.
            param_dict (Optional[dict]): Dictionary mapping hyper-parameter names to the desired values.
            validation_data_uris (Optional[dict(list)]): List of data URIs specifying dataset to be used for validation.
            testing_data_uris (Optional[dict(list)]): List of data URIs specifying dataset to be used for testing.

        Returns:
            2-tuple containing:
                evaluation_results: Results of trained network on final test dataset.
                training_history: Incremental results (per epoch) of network against validation dataset.

        Raises:
            IOError: If no network model could be successfully loaded for training.
        """
        # Load parameters, if any specified
        training_params = self.load_params(
            param_file=param_file, param_dict=param_dict)

        # Load model
        loaded_metadata, loaded_model_path, loaded_formatter_path = self.load(
            model_dir, params=training_params)
        print("Loaded model")

        # Make sure we have a valid model loaded.
        # If the model has already been trained, current metadata should become parent metadata.
        model_state = self._metadata["state"]
        if (model_state is None) or (model_state == "empty"):
            raise IOError(
                "train method called, but no valid network is loaded.")
        elif model_state == "trained":
            self._metadata = {"state": "untrained",
                              "input_shape": self.get_input_shape(),
                              "parent_model_metadata": self._metadata}
        elif model_state == "untrained":
            pass
        else:
            raise IOError("train method called, but loaded network metadata has an unrecognized 'state' value (%s)."
                          % model_state)

        # Load training & validation datasets
        training_data, validation_data, testing_data = alc_common.load_training_datasets(training_data_uris,
                                                                                         self.formatter,
                                                                                         validation_data_uris=validation_data_uris,
                                                                                         testing_data_uris=testing_data_uris,
                                                                                         **training_params)

        # Train model
        print("\n\nStarting Training...")
        training_history = self._train(
            training_data, validation_data, training_params)
        print("Training complete.")
        self._metadata["state"] = "trained"

        # Load testing dataset (if provided) and evaluate network
        if testing_data is None:
            if validation_data is None:
                print("No testing dataset provided. Skipping network evaluation.")
                model_eval = None
            else:
                # Evaluate model with testing data
                print(
                    "No testing dataset provided. Will use validation dataset for evaluation.")
                print("\n\nStarting evaluation...")
                model_eval = self.evaluate(validation_data, training_params)
                print("Evaluation Results:\n%s" % str(model_eval))
        else:
            # Evaluate model with testing data
            print("\n\nStarting evaluation...")
            model_eval = self.evaluate(testing_data, training_params)
            print("Evaluation Results:\n%s" % str(model_eval))

        # Update metadata with training info
        self._metadata["dataset_storage_metadata"] = training_data_uris
        self._metadata["validation_dataset_uris"] = validation_data_uris
        self._metadata["testing_dataset_uris"] = testing_data_uris
        self._metadata["training_parameters"] = alc_common.get_custom_dict(
            training_params)
        self._metadata["training_method"] = "supervised"
        # FIXME: Shouldn't duplicate this information in metadata.
        #       Remove when no longer necessary (is it used anywhere?)
        self._metadata["rng_seed"] = training_params["rng_seed"]
        self._metadata["training_data_fraction"] = training_params["training_data_fraction"]

        # Save model
        self.save(output_dir, training_history)

        # Train assurance monitor if desired
        train_am = training_params.get("lec_assurance_monitor", False)
        if str(train_am).lower() == "true":
            self.train_assurance_monitor(
                training_data, validation_data, self._formatter_path, training_params, output_dir)

        return model_eval, training_history

    # FIXME: This probably doesn't belong in this class. Can go in ALC Common or similar
    @staticmethod
    def train_assurance_monitor(testing_data, training_data, formatter_path, param_dict, output_dir):
        # FIXME: Make this work for classification networks too
        # Train assurance monitor
        assurance_monitor_type = param_dict.get(
            "type", alc_config.training_defaults.ASSURANCE_MONITOR_TYPE)
        assurance_monitor_type = param_dict.get(
            "assurance_monitor_type", assurance_monitor_type)
        monitor = alc_utils.assurance_monitor.load_assurance_monitor(
            assurance_monitor_type)
        monitor.train(training_data, testing_data, output_dir, **param_dict)

        # Save assurance monitor
        _save_dir = monitor._make_unique_subdir(output_dir)
        monitor_save_dir = monitor.save(
            _save_dir, data_formatter_path=formatter_path)
        print("Trained assurance monitor directory: %s" % monitor_save_dir)

    @staticmethod
    def load_params(param_file=None, param_dict=None, default_params=alc_config.training_defaults.var_dict_lower):
        # TODO: This is broken out as a separate function because it will likely have to do more in the future
        #       Also, useful to make available outside of this class
        return alc_common.load_params(param_file=param_file, param_dict=param_dict, default_params=default_params)

    def find_files(self, lec_model_dir):
        """Locate and return paths to the relevant neural network model files in the provided directory.

        Args:
            lec_model_dir (str): Directory where LEC model is located.

        Returns:
            3-tuple containing:
                metadata (dict): Metadata about LEC (or None if no metadata available)
                model_file (str): Filepath to LEC model.
                data_formatter_file (str): Filepath to DataFormatter associated with this LEC.
        """
        # Remove any trailing path separators and ensure LEC model directory is valid
        if lec_model_dir.endswith(os.path.sep):
            lec_model_dir = lec_model_dir[:-1]
        if not (os.path.isdir(lec_model_dir)):
            raise ValueError(
                'Provided model directory (%s) is not a valid directory' % lec_model_dir)

        # Check if this model directory contains a metadata file.
        # If so, then get information from metadata (if provided).
        # Otherwise, use defaults.
        metadata_file = os.path.join(lec_model_dir, _NETWORK_METADATA_FILENAME)
        if os.path.isfile(metadata_file):
            # Read metadata
            with open(metadata_file, 'r') as metadata_fp:
                metadata = json.load(metadata_fp)

            # Get LEC model file from metadata if available, or use the ML library adapter to search default file names.
            model_rel_path = metadata.get("model_relative_path", None)
            if model_rel_path is not None:
                model_file = os.path.join(lec_model_dir, model_rel_path)
            else:
                model_file = self._lib_adapter.find_model_file(lec_model_dir)

            # Get file paths from metadata if available, or use defaults
            data_formatter_rel_path = metadata.get(
                "data_formatter_relative_path", _DATA_FORMATTER_FILENAME)
            data_formatter_file = os.path.join(
                lec_model_dir, data_formatter_rel_path)
            if not (os.path.isfile(data_formatter_file)):
                raise IOError(
                    "No file found at expected Data Formatter path (%s)." % data_formatter_file)

        else:
            # No metadata file. Use defaults.
            model_file = self._lib_adapter.find_model_file(lec_model_dir)
            data_formatter_file = os.path.join(
                lec_model_dir, _DATA_FORMATTER_FILENAME)
            metadata = None

        return metadata, model_file, data_formatter_file
