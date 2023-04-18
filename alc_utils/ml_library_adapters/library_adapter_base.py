#!/usr/bin/env python
# Author: Charlie Hartsell <charles.a.hartsell@vanderbilt.edu>


# FIXME: Some of the abstract functions could be partially implemented to reduce workload when writing a derived class.
class LibraryAdapterBase(object):
    """LibraryAdapterBase class. Defines the interfaces necessary for the NetworkInterface class to interact with
    specific Machine Learning libraries"""

    _variant = "base"

    def __init__(self):
        """Perform any required library initialization"""
        pass

    def get_input_shape(self):
        """Return the input shape of the currently loaded model, or `None` if no model is loaded."""
        raise NotImplementedError(
            "Abstract get_input_shape() function of LibraryAdapterBase called.")

    def get_output_shape(self):
        """Return the output shape of the currently loaded model, or `None` if no model is loaded."""
        raise NotImplementedError(
            "Abstract get_output_shape() function of LibraryAdapterBase called.")

    def load_model(self, lec_model_dir, model_filename=None, params=None):
        """Loads a ML network model at the specified path. Network can be an empty architecture, or a model which
        has already been trained (usually for continuing training).

            Args:
                lec_model_dir (str): Path to directory containing network model and related files.
                model_filename (Optional [str]): Name of model file to load within provided directory.
                params (Optional [dict]): Dictionary of parameters stored as name-value pairs

            Raises:
                IOError: If specified model path is not a valid file.
                AttributeError: If model architecture is provided as a python module, but does not define a 'model' variable
                    or a 'get_model()' function.
                ValueError: If a trained network model is provided, but is not a recognized file type.
        """
        raise NotImplementedError(
            "Abstract load_model() function of LibraryAdapterBase called.")

    def save_model(self, save_dir, model_name=None):
        """This function saves the current state of the loaded model to the specified directory.

            Args:
                save_dir (str): Directory where model should be saved. This function assumes the directory already exists.
                model_name (Optional[str]): Desired name of the saved model file.
                    If no value provided, will use the preferred default option.

            Returns:
                model_name (str): Name of the saved model file. (ie. path relative to *save_dir* argument)
       """
        raise NotImplementedError(
            "Abstract load_model() function of LibraryAdapterBase called.")

    def train(self, training_data, validation_data, params):
        """This function performs training on the currently loaded model using the provided training and validation
        datasets.

            Args:
                training_data (Iterable[various]): Iterable training dataset. Type of each iterated value is variable
                    depending on the DataFormatter used, and should be compatible with the selected ML library.
                validation_data (Iterable[various]): Validation dataset. May be 'None'. Otherwise, will be iterable
                    similar to training_data
                params (dict[str]): User-specified parameters

            Returns:
                history (dict): Dictionary of available training history results (eg. loss at each epoch, etc.)
        """
        raise NotImplementedError(
            "Abstract train() function of LibraryAdapterBase called.")

    def evaluate(self, test_data, params):
        """This function performs training on the currently loaded model using the provided training and validation
        datasets.

            Args:
                test_data (Iterable[various]): Iterable test dataset. Type of each iterated value is variable
                    depending on the DataFormatter used, and should be compatible with the selected ML library.
                params (dict[str]): User-specified parameters

            Returns:
                results (dict): Dictionary of available evaluation results/metrics (eg. loss at each epoch, etc.)
        """
        raise NotImplementedError(
            "Abstract evaluate() function of LibraryAdapterBase called.")

    def predict(self, formatted_input, batch_mode=False):
        """Use loaded network model for forward inference on provided data.
        If batch_mode == True, perform inferencing and return complete batch.
        Otherwise, perform inferencing and return single data point.

            Args:
                formatted_input (various): Formatted input data point(s).
                batch_mode (bool): Flag indicating if a single point or multiple points was provided

            Returns:
                prediction: Output prediction from model
        """
        raise NotImplementedError(
            "Abstract predict() function of LibraryAdapterBase called.")

    def find_model_file(self, model_dir, model_filename=None):
        """Search for a trained LEC model in the given directory using the specified filename, if provided.
        Otherwise, search with default model names.
        Typically used when model file name was not explicitly defined in metadata.

            Args:
                model_dir (str): Directory where model is stored.
                model_filename (Optional[str]): Filename of the stored model within provided directory.

            Returns:
                model_file: Full path (eg. "model_dir/filename") to ML model file.
        """
        raise NotImplementedError(
            "Abstract find_model_file() function of LibraryAdapterBase called.")
