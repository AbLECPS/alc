from __future__ import print_function
import pytest

import argparse
import json
import os

import random
import numpy as np

from alc_utils import config as alc_config
from alc_utils import common as alc_common

from alc_utils import ml_library_adapters
from alc_utils.ml_library_adapters import keras_library_adapter
from alc_utils.ml_library_adapters import pytorch_semseg_adapter

from torch.utils.data import TensorDataset
import torch


@pytest.fixture("module")
def keras_dave2_adapter():
    keras_adapter = ml_library_adapters.load_library_adapter("keras")
    keras_adapter.load_model(os.path.expandvars(
        "$ALC_HOME/alc_utils/network_models/dave2"))
    return keras_adapter


@pytest.fixture("module")
def pytorch_semseg_adapter():
    pytorch_adapter = ml_library_adapters.load_library_adapter(
        "pytorch_semseg")
    pytorch_adapter.load_model(lec_model_dir='')
    return pytorch_adapter


@pytest.fixture("module")
def keras_params():
    parameters = alc_config.training_defaults.var_dict_lower
    parameters["EPOCHS"] = 1
    parameters["epochs"] = 1
    parameters["BATCH_SIZE"] = 1
    parameters["DATASET_NAME"] = "mock-test"
    return parameters


@pytest.fixture("module")
def dave2_data_formatter(keras_dave2_adapter):
    formatter_params = {"input_shape": keras_dave2_adapter.get_input_shape()}
    return alc_common.load_formatter(os.path.expandvars("$ALC_HOME/alc_utils/network_models/dave2"), **formatter_params)


@pytest.fixture("module")
def semseg_data_formatter(pytorch_semseg_adapter):
    formatter_params = {
        "input_shape": pytorch_semseg_adapter.get_input_shape()}
    return alc_common.load_formatter(os.path.expandvars("$ALC_HOME/alc_utils/network_models/segnet"), **formatter_params)


@pytest.fixture()
def create_dave2_dataset(monkeypatch):
    def create_dummy_d2_dataset(data_uris, data_formatter, dataset_name, **dataset_params):
        inputs = []
        targets = []
        for i in range(2):
            inps = np.ones((66, 200, 3))
            inputs.append(inps)
            tgts = np.array([i])
            targets.append(tgts)

        inputs = np.array(inputs)
        targets = np.array(targets)
        inputs = torch.from_numpy(inputs)
        targets = torch.from_numpy(targets)
        return TensorDataset(inputs, targets)

    monkeypatch.setattr('alc_utils.common.load_dataset',
                        create_dummy_d2_dataset)


@pytest.fixture()
def create_semseg_dataset(monkeypatch):
    def create_dummy_ss_dataset(data_uris, data_formatter, dataset_name, **dataset_params):
        inputs = []
        targets = []
        for i in range(2):
            inps = np.zeros((100, 512, 3))
            inputs.append(inps)
            tgts = np.ones((100, 512, 3))
            targets.append(tgts)

        inputs = np.array(inputs)
        targets = np.array(targets)
        inputs = torch.from_numpy(inputs)
        targets = torch.from_numpy(targets)
        return TensorDataset(inputs, targets)

    monkeypatch.setattr('alc_utils.common.load_dataset',
                        create_dummy_ss_dataset)


@pytest.fixture('module')
def network_metadata():
    return {"rng_seed": 10,
            "training_data_fraction": 0.7,
            "dataset_storage_metadata": [
                {
                    "hash": "d73f781d33b0b23f6dfa426ccade5b736a3426bc",
                    "description": "No Upload",
                    "exptParams": {
                        "upload_path_prefix": "iver2_gooddata",
                        "pipe_angle_max": 0.011,
                        "enable_disturbance": False,
                        "testing": 1,
                        "vehicle_roll": 0,
                        "bury_len_max": 0.02,
                        "vehicle_latitude": 38.95320298,
                        "min_fls_samples": 5,
                        "headless": True,
                        "rl_model_dir": "/alc_workspace/jupyter/alc_ALC_IVER/RLTrainingLEC1/1568130616987/1568130616987/RLModel",
                        "num_pipes": 20,
                        "bury_len_min": 0.01,
                        "fls_clustering_neighborhood": 0.01,
                        "fls_from_file": False,
                        "unpause_timeout": 15,
                        "termination_topic": "/alc/stopsim",
                        "disturbance_filename": "",
                        "vehicle_pitch": 0,
                        "origin_latitude": 38.971203,
                        "random_seed": 42,
                        "num_episodes": 1,
                        "origin_altitude": 0,
                        "pipe_posx": 30,
                        "pipe_posy": 2,
                        "vehicle_altitude": -45,
                        "pipe_len_max": 30.1,
                        "pipe_len_min": 30,
                        "pipe_pos_side": 1,
                        "enable_obstacles": False,
                        "vehicle_longitude": -76.398,
                        "gui": False,
                        "vehicle_yaw": 0,
                        "init_speed": 1.0,
                        "origin_longitude": -76.398464,
                        "record": True,
                        "upload_results": False,
                        "timeout": 2500,
                        "pipe_angle_min": 0.01
                    },
                    "upload_prefix": None,
                    "result_url": "charlie_ALC_IVER_Demo/DataGen_Straight/1591716337078/1591716337078/config-0/result.ipynb",
                    "directory": "jupyter/charlie_ALC_IVER_Demo/DataGen_Straight/1591716337078/1591716337078/config-0"
                },
                {
                    "hash": "bb69f0048fd6cf79e90df807a5931d4c835be39f",
                    "description": "No Upload",
                    "exptParams": {
                        "upload_path_prefix": "iver2_gooddata",
                        "pipe_angle_max": -0.2,
                        "enable_disturbance": False,
                        "testing": 1,
                        "vehicle_roll": 0,
                        "bury_len_max": 0.2,
                        "vehicle_latitude": 38.95320298,
                        "min_fls_samples": 5,
                        "headless": True,
                        "rl_model_dir": "/alc_workspace/jupyter/alc_ALC_IVER/RLTrainingLEC1/1568130616987/1568130616987/RLModel",
                        "num_pipes": 20,
                        "bury_len_min": 0.1,
                        "fls_clustering_neighborhood": 0.01,
                        "fls_from_file": False,
                        "unpause_timeout": 15,
                        "termination_topic": "/alc/stopsim",
                        "disturbance_filename": "",
                        "vehicle_pitch": 0,
                        "origin_latitude": 38.971203,
                        "random_seed": 27168,
                        "num_episodes": 1,
                        "origin_altitude": 0,
                        "pipe_posx": 30,
                        "pipe_posy": 2,
                        "vehicle_altitude": -45,
                        "pipe_len_max": 30.1,
                        "pipe_len_min": 30,
                        "pipe_pos_side": -1,
                        "enable_obstacles": False,
                        "vehicle_longitude": -76.398,
                        "gui": False,
                        "vehicle_yaw": 0,
                        "init_speed": 1.0,
                        "origin_longitude": -76.398464,
                        "record": True,
                        "upload_results": False,
                        "timeout": 2500,
                        "pipe_angle_min": -0.201
                    },
                    "upload_prefix": None,
                    "result_url": "charlie_ALC_IVER_Demo/DataGen_Left_Bend/1591676103286/1591676103286/config-0/result.ipynb",
                    "directory": "jupyter/charlie_ALC_IVER_Demo/DataGen_Left_Bend/1591676103286/1591676103286/config-0"
                },
                {
                    "hash": "73af0b0dbf8a4969a0f53a35d13b874d643a62fc",
                    "description": "No Upload",
                    "exptParams": {
                        "upload_path_prefix": "iver2_gooddata",
                        "pipe_angle_max": 0.201,
                        "enable_disturbance": False,
                        "testing": 1,
                        "vehicle_roll": 0,
                        "bury_len_max": 0.2,
                        "vehicle_latitude": 38.95320298,
                        "min_fls_samples": 5,
                        "headless": True,
                        "rl_model_dir": "/alc_workspace/jupyter/alc_ALC_IVER/RLTrainingLEC1/1568130616987/1568130616987/RLModel",
                        "num_pipes": 20,
                        "bury_len_min": 0.1,
                        "fls_clustering_neighborhood": 0.01,
                        "fls_from_file": False,
                        "unpause_timeout": 15,
                        "termination_topic": "/alc/stopsim",
                        "disturbance_filename": "",
                        "upload_results": False,
                        "origin_latitude": 38.971203,
                        "random_seed": 27168,
                        "num_episodes": 1,
                        "pipe_posx": 30,
                        "origin_altitude": 0,
                        "pipe_posy": 2,
                        "vehicle_altitude": -45,
                        "pipe_len_max": 30.1,
                        "pipe_len_min": 30,
                        "pipe_pos_side": -1,
                        "enable_obstacles": False,
                        "vehicle_longitude": -76.398,
                        "gui": False,
                        "vehicle_yaw": 0,
                        "init_speed": 1.0,
                        "origin_longitude": -76.398464,
                        "record": True,
                        "vehicle_pitch": 0,
                        "timeout": 2500,
                        "pipe_angle_min": 0.2
                    },
                    "upload_prefix": None,
                    "result_url": "charlie_ALC_IVER_Demo/DataGen_Right_Bend/1591676814143/1591676814143/config-0/result.ipynb",
                    "directory": "jupyter/charlie_ALC_IVER_Demo/DataGen_Right_Bend/1591676814143/1591676814143/config-0"
                }
            ],
            "parent_model_metadata": {
                "state": "trained"
            },
            "input_shape": [
                100,
                512
            ],
            "validation_dataset_uris": None,
            "state": "trained",
            "testing_dataset_uris": None,
            "data_formatter_relative_path": "data_formatter.py",
            "training_method": "supervised",
            "model_relative_path": "/alc_workspace/jupyter/charlie_ALC_IVER_Demo/LEC-2/1591717560506/TrainingResult_2020_06_09_15_46_04/model.pkl",
            "training_parameters": {
                "shuffle": False,
                "verbose": True,
                "validate_interval": 1,
                "train_assurance_monitor": False,
                "useful_data_fraction": 1,
                "epochs": 5,
                "assurance_monitor_type": "CUSTOM",
                "data_split_mode": 2,
                "use_generators": False,
                "use_vae": True,
                "optimizer": "CUSTOM",
                "rng_seed": 10,
                "var_dict": {
                    "METRICS": [
                        "accuracy"
                    ],
                    "OPTIMIZER": "adam",
                    "VERBOSE": True,
                    "CALLBACKS": [],
                    "VALIDATE_INTERVAL": 1,
                    "PRINT_INFO_INTERVAL": 10,
                    "TRAIN_ASSURANCE_MONITOR": False,
                    "BATCH_SIZE": 64,
                    "LOSS": "mse",
                    "EPOCHS": 5,
                    "ASSURANCE_MONITOR_TYPE": "knn",
                    "DATA_SPLIT_MODE": 2,
                    "TRAINING_DATA_FRACTION": 1.0,
                    "USE_GENERATORS": False,
                    "USEFUL_DATA_FRACTION": 1.0,
                    "UPLOAD_RESULTS": False,
                    "SHUFFLE": False,
                    "DATA_BATCH_MODE": 2,
                    "RNG_SEED": 10,
                    "DATASET_NAME": "rosbag"
                },
                "window_size": 5,
                "print_info_interval": 10,
                "ml_library": "CUSTOM",
                "path_prefix": "CUSTOM",
                "data_batch_mode": 2,
                "upload_results": False,
                "lec_assurance_monitor": False,
                "epsilon": 0.75,
                "batch_size": 16,
                "metrics": [
                    "accuracy"
                ],
                "callbacks": [],
                "runsltrainingsetup": 1,
                "loss": "CUSTOM",
                "training_data_fraction": 0.7,
                "upload": False,
                "dataset_name": "CUSTOM"
            }
            }
