# !/usr/bin/env python
"""Adapter class for using the PyTorch machine-learning framework and pytorch_semseg library with the ALC Toolchain."""
# Author: Charlie Hartsell <charles.a.hartsell@vanderbilt.edu>
from __future__ import print_function
from future.utils import viewitems

import os
import sys
import numpy as np
import torch
import torch.cuda
import torch.nn
import logging
import glob
import random
import time
import tqdm
import alc_utils.config as alc_config
import alc_utils.common as alc_common
from alc_utils.ml_library_adapters import LibraryAdapterBase

# FIXME: relative imports are fragile. Get rid of this
#root = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(root, 'pytorch_semseg')
sys.path.append(module_path)
import ptsemseg.models
import ptsemseg.loss
import ptsemseg.loader
import ptsemseg.utils
import ptsemseg.metrics
import ptsemseg.augmentations
import ptsemseg.schedulers
import ptsemseg.optimizers

ADAPTER_NAME = "PytorchLibraryAdapter"


# TODO: Comment/doc-string this better
class PytorchLibraryAdapter(LibraryAdapterBase):
    """Adapter class for using the PyTorch machine-learning framework
    and pytorch_semseg library with the ALC Toolchain."""
    default_model_filenames = ["model.pkl", "LECModel.py"]

    _variant = "pytorch"

    def __init__(self):
        super(PytorchLibraryAdapter, self).__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.epoch = None
        self.iter_count = None
        self.iterations_per_epoch = None
        self.loaded_model_state = None

        # Set default parameters. Dictionary will be updated when a model is loaded.
        self.model_name = "segnet"
        self.model_parameters = {"img_norm": True,
                                 "dataset": "aa_side_scan_sonar",
                                 "n_classes": 4,
                                 "img_size": (100, 512)}

        # FIXME: This shouldn't be configured here.
        #   Should be passed to the LibraryAdapterBase and available automatically
        # Configure logging levels and handlers
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.DEBUG)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        stream_handler.setFormatter(stream_formatter)
        self._logger.addHandler(stream_handler)

    def get_input_shape(self):
        if self.model is None:
            return None
        else:
            return self.model_parameters["img_size"]

    ############### I/O Functions ###############

    def load_model(self, lec_model_dir, model_filename=None, params=None):
        """Loads a PyTorch-SemSeg model from the specified directory. Network can be an empty architecture,
        or a model which has already been trained (usually for continuing training).

        Args:
            lec_model_dir (str): Path to directory containing network model.

        Raises:
            IOError: If specified model path is not a valid file.
            AttributeError: If model archtecture is provided as a python module, but does not define a 'model' variable
                or a 'ptsemseg.loader.get_model()' function.
            ValueError: If a trained network model is provided, but is not a recognized file type.
        """
        # Check if a <model_name> is specified in saved model file
        model_file_name = self.find_model_file(
            lec_model_dir, model_name=self.model_name)
        if model_file_name is None:
            IOError("PyTorch-SemSeg adapter could not find a valid LEC model file in the specified directory (%s)."
                    % lec_model_dir)

        # If the model is a python file, it is an empty architecture and should contain a 'get_model()' function
        elif model_file_name.endswith(".py"):
            self.loaded_model_state = None
            model_module = alc_common.load_python_module(model_file_name)

            # FIXME: Don't like passing params to get_model, which then passes other parameters back. Revise this
            # Try to load model with params
            try:
                self.model, loaded_parameters = model_module.get_model(params)
                self.model_parameters.update(loaded_parameters)
            except TypeError:
                # If "get_model" function does not accept an argument, load without params
                self.model_name, loaded_parameters = model_module.get_model()
                self.model_parameters.update(loaded_parameters)
            except AttributeError:
                print(
                    "LEC model (%s) does not contain a valid get_model() function." % model_file_name)

        # If the model is a pickle file, it has already been trained and should contain all necessary info
        elif model_file_name.endswith(".pkl"):
            self.loaded_model_state = torch.load(model_file_name)
            self.model_name = self.loaded_model_state.get(
                "model_name", self.model_name)
            self.model_parameters.update(self.loaded_model_state["parameters"])

        # Setup Model
        self._logger.info("Setting up the model: " + str(self.model_name))
        print("Setting up the model: " + str(self.model_name))

        # Load network model (w/ uninitialized weights)
        # FIXME: the "version" kwarg does nothing in this function. Why is it used?
        self.model = ptsemseg.models.get_model({"arch": self.model_name},
                                               self.model_parameters["n_classes"],
                                               version=self.model_parameters["dataset"])

        # Initialize model weights from saved file, if available
        # FIXME: Is there any more init required for an empty model? Check train.py
        if self.loaded_model_state is not None:
            state = ptsemseg.utils.convert_state_dict(
                self.loaded_model_state["model_state"])
            self.model.load_state_dict(state)

        # Enable parallel (multi-GPU) processing
        self._logger.info("Using (%s) GPUs." % torch.cuda.device_count())
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        self.model = torch.nn.DataParallel(
            self.model, device_ids=range(torch.cuda.device_count()))

        # Move model to the desired device.
        self.model.to(self.device)
        self._logger.info("Ready...")

    def save_model(self, save_dir,  model_name=default_model_filenames[0]):
        """This function will save the current state of the loaded model to the specified directory.

        Args:
            save_dir (str): Directory where model should be saved. This function assumes the directory already exists.
            model_name (Optional[str]): Desired name of the saved model file.
                If no value provided, will use the preferred default option.

        Returns:
            model_name (str): Name of the saved model file. (ie. path relative to *save_dir* argument)
       """
        model_save_path = os.path.join(save_dir, model_name)
        state = {
            "epoch": self.epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "model_name": self.model_name,
            "parameters": self.model_parameters
            # "best_iou": best_iou,
        }
        torch.save(state, model_save_path)

        return model_save_path

    ############### Training/Evaluation Functions ###############

    def train(self, training_data, validation_data, params):
        print("Training with compute device '%s'." % self.device)

        # Setup seeds
        torch.manual_seed(params['rng_seed'])
        torch.cuda.manual_seed(params['rng_seed'])
        np.random.seed(params['rng_seed'])
        random.seed(params['rng_seed'])

        # Setup Augmentations
        augmentations = params.get('augmentations', None)
        data_aug = ptsemseg.augmentations.get_composed_augmentations(
            augmentations)

        # Setup optimizer, lr_scheduler and loss function
        # FIXME: Need to read ALL optimizer params here
        if params["optimizer"].lower() == "sgd":
            cfg = {"training": {"optimizer": {
                "name": params["optimizer"], "lr": 0.1}}}
        else:
            cfg = {"training": {"optimizer": {"name": params["optimizer"]}}}
        optimizer_cls = ptsemseg.optimizers.get_optimizer(cfg)
        optimizer_params = {
            k: v for k, v in cfg['training']['optimizer'].items() if k != 'name'}

        self.optimizer = optimizer_cls(
            self.model.parameters(), **optimizer_params)
        self._logger.info("Using optimizer {}".format(self.optimizer))

        # FIXME: Currently use default scheduler and loss function. Is there a config that should be passed here?
        # self.scheduler = ptsemseg.schedulers.get_scheduler(self.optimizer, cfg['training']['lr_schedule'])
        self.scheduler = ptsemseg.schedulers.get_scheduler(
            self.optimizer, None)

        # loss_fn = ptsemseg.loss.get_loss_function(cfg)
        # FIXME: Any other training parameters?
        loss_fn = ptsemseg.loss.get_loss_function({"training": {"loss": None}})
        self._logger.info("Using loss {}".format(loss_fn))

        # If model was loaded from non-empty architecture, then load states to resume training
        if self.loaded_model_state is not None:
            self._logger.info(
                "Loading model and optimizer states from trained model.")
            self.optimizer.load_state_dict(
                self.loaded_model_state["optimizer_state"])
            self.scheduler.load_state_dict(
                self.loaded_model_state["scheduler_state"])

        time_meter = ptsemseg.metrics.averageMeter()

        # # Determine desired_iterations based on desired number of epochs and number of batches in the training set
        # self.desired_iterations = params["epochs"] * len(training_data)

        # If provided training_data has a len function, use it. Otherwise, determine length after 1 epoch is complete.
        try:
            self.iterations_per_epoch = len(training_data)
        except TypeError:
            self.iterations_per_epoch = None

        # Train network with one batch of data at a time.
        # Repeat training loop for desired number of iterations.
        # TODO: Compute and return IoU metric
        # best_iou = -100.0
        self.epoch = 0
        self.iter_count = 0
        eval_history = {}
        while self.epoch < params["epochs"]:
            batch_count = 0
            for (input_batch, output_batch) in training_data:
                # Increment/Init variables
                self.iter_count += 1
                batch_count += 1
                start_ts = time.time()

                # Convert data batches to correct type
                input_batch = self._convert_data_batch(input_batch)
                output_batch = self._convert_data_batch(output_batch)

                # Set model to training mode and move data to compute device
                self.model.train()
                input_batch = input_batch.to(self.device)
                output_batch = output_batch.to(self.device)

                self.optimizer.zero_grad()
                predicted_outputs = self.model(input_batch)

                loss = loss_fn(input=predicted_outputs, target=output_batch)
                loss.backward()

                self.optimizer.step()

                # FIXME: Should scheduler be run once per batch (current) or once per epoch?
                self.scheduler.step()

                time_meter.update(time.time() - start_ts)

                if self.iter_count % params["print_info_interval"] == 0:
                    fmt_str = "Iter [{:d}/{:s}]  Loss: {:.4f}  lr: {:.6f}  Time/Batch: {:.4f}"
                    if batch_count == 0:
                        time_per_batch = time_meter.avg
                    else:
                        time_per_batch = time_meter.avg / batch_count
                    if self.iterations_per_epoch is None:
                        total_iters = "??"
                    else:
                        total_iters = str(
                            self.iterations_per_epoch * params["epochs"])
                    print_str = fmt_str.format(self.iter_count,
                                               total_iters,
                                               loss.item(),
                                               self.scheduler.get_lr()[0],
                                               time_per_batch)

                    print(print_str)
                    # writer.add_scalar('loss/train_loss', loss.item(), self.epoch + 1)
                    time_meter.reset()

                # if self.iter_count % len(training_data) == 0:
                #     self.epoch += 1
                #     break

            self.epoch += 1
            if self.epoch % params["validate_interval"] == 0:
                eval_metrics = self.evaluate(validation_data, params)
                self._logger.info("Iter %d Loss: %.4f" %
                                  (self.epoch, eval_metrics["loss"]))
                print("Eval metrics: %s" % str(eval_metrics))

                for key, value in viewitems(eval_metrics):
                    # Check if eval history has already been initialized to empty lists.
                    # Each time evaluation is run, store values in appropriate list
                    if eval_history.get(key, None) is None:
                        eval_history[key] = []
                    eval_history[key].append(value)

            if self.iterations_per_epoch is None:
                self.iterations_per_epoch = self.iter_count

        return eval_history

    def evaluate(self, test_data, params):
        # loss_fn = ptsemseg.loss.get_loss_function(cfg)
        loss_fn = ptsemseg.loss.get_loss_function({"training": {"loss": None}})

        # Setup Metrics
        running_metrics_val = ptsemseg.metrics.runningScore(
            self.model_parameters["n_classes"])
        val_loss_meter = ptsemseg.metrics.averageMeter()

        self.model.eval()
        with torch.no_grad():
            for i_val, (val_input_batch, val_output_batch) in tqdm.tqdm(enumerate(test_data)):
                # Convert data batches to correct type
                val_input_batch = self._convert_data_batch(val_input_batch)
                val_output_batch = self._convert_data_batch(val_output_batch)

                val_input_batch = val_input_batch.to(self.device)
                val_output_batch = val_output_batch.to(self.device)

                pred_val_outputs = self.model(val_input_batch)
                val_loss = loss_fn(input=pred_val_outputs,
                                   target=val_output_batch)

                pred = pred_val_outputs.data.max(1)[1].cpu().numpy()
                gt = val_output_batch.data.cpu().numpy()

                running_metrics_val.update(gt, pred)
                val_loss_meter.update(val_loss.item())

        # Add results to dictionary and return
        eval_metrics = {"loss": val_loss_meter.avg}

        score, class_iou = running_metrics_val.get_scores()
        for k, v in viewitems(score):
            eval_metrics[k] = v

        for k, v in viewitems(class_iou):
            eval_metrics[k] = v

        return eval_metrics

    def predict(self, input_data):
        """ Performs a forward pass on the provided image

        Returns:
            ndarray segmentation map
            list ndarray probability maps for each class
        """
        # FIXME: Is there a way to use the model for inference on a single data point instead of a batch?
        #       Can always convert single point into a batch of length 1 as done below, but seems unnecessary
        # Set network to evaluate mode and run the input data through
        self.model.eval()
        input_data_batch = torch.unsqueeze(input_data, 0)
        input_data_batch = input_data_batch.to(self.device)
        output = self.model(input_data_batch)
        return output[0]

    @classmethod
    def find_model_file(cls, model_dir, model_name=None):
        """Search for a trained LEC model in the given directory using the specified model_name.
        If no model_name provided or no file with model_name exists, use default model names.
        Typically used when model file name was not explicitly defined in metadata."""
        # Check for files matching provided model name. Assumes model file starts with <model_name> and ends with ".pkl"
        if model_name is not None:
            for model_filename in glob.glob(os.path.join(model_dir, model_name) + "*.pkl"):
                model_filename = os.path.abspath(model_filename)
                if os.path.isfile(model_filename):
                    return model_filename

        # If no files matching <model_name> found, check list of default model names
        for model_filename in cls.default_model_filenames:
            if os.path.isfile(os.path.join(model_dir, model_filename)):
                return os.path.join(model_dir, model_filename)

        return None

    @staticmethod
    def _convert_data_batch(data_batch):
        # # DataLoader formats each batch of data as a List of data points.
        # # For PyTorch, each data point in the List should be of type Tensor.
        # # Verify the data is formatted as expected, then use PyTorch "stack"
        # # function to turn this list of tensors into one large tensor.
        # if len(data_batch) > 0:
        #     if not isinstance(data_batch[0], torch.Tensor):
        #         raise TypeError("PyTorch-SemSeg adapter received data batch containing points of incorrect type. "
        #                         "Expected 'torch.Tensor', but got '%s'" % type(data_batch[0]))
        #
        # return torch.stack(data_batch)

        return data_batch
