#!/usr/bin/env python
"""Adapter class for using the Keras machine-learning framework with TensorFlow as the backend engine."""
# Author: Charlie Hartsell <charles.a.hartsell@vanderbilt.edu>
from __future__ import print_function

import os
import warnings
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import threading
import torch
import alc_utils.common as alc_common
from library_adapter_base import LibraryAdapterBase
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

ADAPTER_NAME = "KerasLibraryAdapter"

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass


# Dictionary mapping (& filtering) the ALC training hyperparameters to their Keras counterparts
# Key is the ALC param name. Value is Keras param name
training_parameter_filter_map = {
    "epochs": "epochs",
    "verbose": "verbose",
    "callbacks": "callbacks",
    # FIXME: Something is wrong with the Keras pip package. "validation_freq" is a valid param, but causes exception.
    # "validate_interval": "validation_freq",
    "class_weight": "class_weight",
    "sample_weight": "sample_weight",
    "initial_epoch": "initial_epoch",
    "max_queue_size": "max_queue_size",
    "workers": "workers",
    "use_multiprocessing": "use_multiprocessing"
}

evaluation_parameter_filter_map = {
    "verbose": "verbose",
    # "callbacks": "callbacks", # Keras throws an error if this parameter is used. Not sure why.
    # "sample_weight": "sample_weight", # Not used by 'evaluate_generator'
    "max_queue_size": "max_queue_size",
    "workers": "workers",
    "use_multiprocessing": "use_multiprocessing"
}
# removed this line from evaluation_parameter_filter_map as callbacks were not accepted
# "callbacks": "callbacks",
# "batch_size": "batch_size",

compile_parameter_filter_map = {
    "loss": "loss",
    "optimizer": "optimizer",
    "metrics": "metrics",
    "loss_weights": "loss_weights",
    "sample_weight_mode": "sample_weight_mode",
    "weighted_metrics": "weighted_metrics",
    "target_tensors": "target_tensors"
}

def KerasGeneratorAdapter(dataloader):
    data = {}
    data['x']=[]
    data['y']=[]
    
    
    train_x = []
    train_y = []
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        for j in range(len(inputs)):
            data['x'].append(inputs[j].cpu().detach().numpy())
            data['y'].append(targets[j].cpu().detach().numpy())

    data['x'] = np.array(data['x'])
    data['y'] = np.array(data['y'])
    return data


# TODO: Comment/doc-string this better
class KerasLibraryAdapter(LibraryAdapterBase):
    """Adapter class for using the Keras machine-learning framework with TensorFlow as the backend engine."""
    # Standard names of Keras model files (trained/untrained)
    # First name in list is preferred option when saving model if no model name is provided.
    default_model_filenames = ["model.h5", "model.keras"]
    default_architecture_filename = "LECModel.py"
    default_weights_filenames = ["model_weights.h5", "model_weights.keras"]

    _variant = "keras"

    def __init__(self):
        super(KerasLibraryAdapter, self).__init__()
        self.model = None
       
        physical_devices = tf.config.list_physical_devices('GPU') 
        for gpu_instance in physical_devices:
            try: 
                tf.config.experimental.set_memory_growth(gpu_instance, True)
            except:
                pass


    def get_input_shape(self):
        if self.model is None:
            return None
        else:
            # Remove the initial "batch_size" dimension
            return self.model.input_shape[1:]

    def get_output_shape(self):
        if self.model is None:
            return None
        else:
            # Remove the initial "batch_size" dimension
            return self.model.output_shape[1:]

    ############### I/O Functions ###############

    def load_model(self, lec_model_dir, model_filename=None, params=None):
        """Loads a Keras network model at the specified path. Network can be an empty architecture, or a model which
        has already been trained (usually for continuing training).

        Args:
            lec_model_dir (str): Path to directory containing network model and related files.
            model_filename (Optional [str]): Name of model file to load within provided directory.

        Raises:
            IOError: If specified model path is not a valid file.
            AttributeError: If model architecture is provided as a python module, but does not define a 'model'
                variable or a 'get_model()' function.
            ValueError: If a trained network model is provided, but is not a recognized file type.
        """
        # Input checks
        if not (os.path.isdir(lec_model_dir)):
            raise IOError(
                "Specified LEC model path (%s) is not a valid directory" % lec_model_dir)

        # Find model file within provided directory
        model_path = self.find_model_file(
            lec_model_dir, model_filename=model_filename)
        weights_path = self.find_weights_file(lec_model_dir)

        # Load model based on the files found above
        self.model = None
        if model_path.endswith(".py"):
            self._load_from_arch_weights(model_path, weights_path, params)
        elif model_path is not None:
            self._load_from_serialized(model_path)
        else:
            raise ValueError(
                "Failed to find a compatible Keras model in directory %s." % lec_model_dir)

    def _load_from_arch_weights(self, arch_file, weights_file, params):
        model_module = alc_common.load_python_module(arch_file)

        # Load model from architecture file
        if hasattr(model_module, "get_model"):
            # Try to load model from function with params.
            # Various methods for passing parameters. Try them in order from most to least preferred.
            # Order: Keyword arguments, positional argument, no parameters
            try:
                self.model = model_module.get_model(**params)
            except TypeError:
                # FIXME: Both of these methods are for backwards compatibility. Remove when no longer necessary.
                try:
                    self.model = model_module.get_model(params)
                except TypeError:
                    self.model = model_module.get_model()
        else:
            # Try to load model directly from variable if get_model function is not defined
            print("LEC model (%s) does not contain a get_model() function." % arch_file)
            warnings.warn(
                "Loading LEC model directly from 'model' variable is deprecated.", DeprecationWarning)
            self.model = model_module.model

        # Load model weights, if present
        if weights_file is not None:
            self.model.load_weights(weights_file)

    def _load_from_serialized(self, model_file):
        #with self.tf_graph.as_default():
        #    keras.backend.set_session(self.session)
            # Keras apparently does not support unicode paths. Make sure path is 'str' type.
        self.model = keras.models.load_model(str(model_file))

    def save_model(self, save_dir, model_weights_name=default_weights_filenames[0]):
        """This function saves the current state of the loaded model to the specified directory.

        Args:
            save_dir (str): Directory where model should be saved. This function assumes the directory already exists.
            model_weights_name (Optional[str]): Desired name of the file for saving model weights.
                If no value provided, will use the preferred default option.

        Returns:
            model_name (str): Name of the saved model file. (ie. path relative to *save_dir* argument)
       """
        model_abs_path = os.path.join(save_dir, model_weights_name)

        # Save model weights
        # Set TF Session & Graph in case this function is called from a separate thread than the __init__
        #with self.tf_graph.as_default():
        #    keras.backend.set_session(self.session)
        self.model.save_weights(model_abs_path)

        return model_weights_name

    ############### Training/Evaluation Functions ###############

    def train(self, training_data, validation_data, params):
        # Set Numpy and TensorFlow RNG seeds for reproducible training
        np.random.seed(params["rng_seed"])
        tf.random.set_seed(params["rng_seed"])
        
        physical_devices = tf.config.list_physical_devices('GPU') 
        for gpu_instance in physical_devices:
            try: 
                tf.config.experimental.set_memory_growth(gpu_instance, True)
            except:
                pass
        

        compile_params = alc_common.filter_parameters(
                params, compile_parameter_filter_map)
        self.model.compile(**compile_params)

            # Train model
        training_data_keras = KerasGeneratorAdapter(training_data)
        training_params = alc_common.filter_parameters(
                params, training_parameter_filter_map)
        if validation_data is not None:
            validation_data_keras = KerasGeneratorAdapter(validation_data)

                # Keras thinks 'None' and 0 are the same thing. Add a catch for this
            if len(validation_data_keras) == 0:
                raise ValueError(
                   "train() function was provided a validation dataset, but length of this set is 0.")

            training_result = self.model.fit(training_data_keras['x'],training_data_keras['y'],
                                                    steps_per_epoch=len(
                                                    training_data_keras),
                                                    validation_data=(validation_data_keras['x'],validation_data_keras['y']),
                                                    validation_steps=len(
                                                    validation_data_keras),
                                                    **training_params)
        else:

            training_result = self.model.fit(training_data_keras['x'], training_data_keras['y'],
                                                        steps_per_epoch=len(
                                                        training_data_keras),
                                                        **training_params)
        return training_result.history

    def evaluate(self, test_data, params):
        # Input checks
        if test_data is None:
            return None

        physical_devices = tf.config.list_physical_devices('GPU') 
        for gpu_instance in physical_devices:
            try: 
                tf.config.experimental.set_memory_growth(gpu_instance, True)
            except:
                pass

        # Convert test data type to be compatible with Keras
        test_data_keras = KerasGeneratorAdapter(test_data)


        # Run evaluator on generator
        eval_params = alc_common.filter_parameters(
               params, evaluation_parameter_filter_map)
        score = self.model.evaluate(test_data_keras['x'],test_data_keras['y'],
                                            steps=len(test_data_keras),
                                            **eval_params)

        # Place evaluation scores into dictionary indexed by the name of each evaluation metric
        model_eval = {}
        if not isinstance(score, list):
            # Evaluate function may return a scalar loss value if metrics set is empty.
            model_eval["loss"] = score
        else:
            i = 0
            for metric_name in self.model.metrics_names:
                model_eval[metric_name] = score[i]
                i += 1

        return model_eval

    def predict(self, formatted_input, batch_mode=False):
        """Use loaded network model for forward inference on provided data.
        If batch_mode == True, perform inferencing and return complete batch.
        Otherwise, perform inferencing and return single data point."""

        if batch_mode:
            formatted_input = np.array(formatted_input)
            result = self.model.predict(formatted_input)
        else:
            formatted_input = np.array([formatted_input])
            result = self.model.predict(formatted_input)[0]
        return result

    def find_model_file(self, model_dir, model_filename=None):
        """Search for a trained LEC model in the given directory using the specified filename, if provided.
         Otherwise, search with default model names.
        Typically used when model file name was not explicitly defined in metadata."""
        # Find model file in case this network was saved with old model.save() technique
        if model_filename is not None:
            if os.path.isfile(os.path.join(model_dir, model_filename)):
                return os.path.join(model_dir, model_filename)
        else:
            for model_filename in self.default_model_filenames:
                if os.path.isfile(os.path.join(model_dir, model_filename)):
                    return os.path.join(model_dir, model_filename)

        # Next check if architecture file exists
        arch_file = os.path.join(model_dir, self.default_architecture_filename)
        if os.path.isfile(arch_file):
            return arch_file

        raise IOError(
            "Failed to find LEC model file in provided directory (%s)." % model_dir)

    def find_weights_file(self, model_dir):
        # Find model weights file
        weights_file = None
        for name in self.default_weights_filenames:
            if os.path.isfile(os.path.join(model_dir, name)):
                weights_file = os.path.join(model_dir, name)
                break

        return weights_file


class KerasGeneratorAdapter1(object):
    """This class acts as an adapter between generic iterator types and the required Keras generator type. Keras
    requires generators to have a "len" function and be thread safe, but iterators in general may not have these
    features. Also, if the original iterator returns a PyTorch Tensor, this will be converted to a Numpy ndarray."""

    def __init__(self, orig_data_gen):
        self._orig_data_gen = orig_data_gen
        self._data_iter_instance = iter(self._orig_data_gen)
        self._length = self._calc_len()
        self._lock = threading.Lock()

    def __iter__(self):
        return self

    # Python 3 compatability
    def __next__(self):
        return self.next()

    def __len__(self):
        return self.len()

    def next(self):
        with self._lock:
            try:
                retval = next(self._data_iter_instance)
            except StopIteration as e:
                # We reached the end of the generator. Create a new copy to restart from the beginning
                # TODO: If data_generator_factory returns an empty generator, then this will raise StopIteration again.
                #       Is this desired behavior?
                self._data_iter_instance = iter(self._orig_data_gen)
                retval = next(self._data_iter_instance)

        # FIXME: get rid of this and the torch import
        # If torch.Tensor provided, convert to Numpy ndarray
        if torch.is_tensor(retval):
            retval = retval.cpu().detach().numpy()

        return retval

    # Function which returns length of original generator
    def len(self):
        return self._length

    # Keras requires that the number of items in the iterator is known.
    # If iterator includes a "len" function, this is trivial.
    # Otherwise, this function iterates through all items once to count the length.
    # Inefficient, but don't know a good way to get around this.
    def _calc_len(self):
        try:
            iter_len = len(self._orig_data_gen)
        except TypeError:
            iter_len = 0
            for _ in self._orig_data_gen:
                iter_len += 1

        if iter_len is None:
            raise RuntimeError(
                "__len__ function returned unexpected 'None' value.")

        return iter_len

import numpy as np
import tensorflow.keras as k




    


class KerasGeneratorAdapter2(k.utils.Sequence):
    
    def __init__(self, gen):
        self.gen = gen
        self.iter = iter(gen)
    
    def __getitem__(self, _):
        try:
            x,y = next(self.iter)
        except StopIteration:
            self.iter = iter(self.gen)
            x, y = next(self.iter)
        return x,y
    
    def __len__(self):
        return len(self.gen)