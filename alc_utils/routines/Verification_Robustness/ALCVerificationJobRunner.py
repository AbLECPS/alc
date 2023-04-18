#!/usr/bin/env python
# This contains python code for executing ALC jobs
# Based on the jobtype, it invokes the appropriate functions in alc_utils.
from __future__ import print_function

import os
import json
import sys
from alc_utils import common as alc_common

import tempfile
from pathlib import Path
import os
import re
from urllib.parse import urlunsplit, urljoin
import zipfile
import itertools
import imageio
import numpy
import urllib.request
import matlab.engine
from ActivityInterpreterBase import ActivityInterpreterBase


class Keys:
    current_choice_key_name = "current_choice"
    hash_key_name = "hash"
    inputs_key_name = "inputs"
    parameters_key_name = "parameters"
    outputs_key_name = "outputs"
    attributes_key_name = "attributes"


class Attributes:
    asset_attribute_name = "asset"
    constraint_attribute_name = "constraint"
    choice_list_parameter_name = "ChoiceList"
    current_choice_attribute_name = "CurentChoice"
    default_value_attribute_name = "defaultValue"
    max_attribute_name = "max"
    min_attribute_name = "min"
    name_attribute_name = "name"
    required_attribute_name = "required"
    value_attribute_name = "value"


alc_port = 8000
url_hostname = "192.168.1.83"


class ActivityInterpreter(ActivityInterpreterBase):

    alc_home_env_var_name = "ALC_HOME"

    training_data_dir_name = "TrainingData"
    verification_data_dir_name = "VerificationData"

    data_set_key_name = "DataSet"
    name_key_name = "name"
    method_key_name = "method"
    parameters_key_name = "parameters"
    params_table_key_name = "ParamsTable"

    input_lec_key_name = "inputLEC"
    input_training_data_key_name = "input_training_data"
    input_verification_data_key_name = "input_verification_data"

    brightening_attack_type_name = "brightening"
    darkening_attack_type_name = "darkening"
    random_noise_attack_type_name = "random_noise"

    image_path_key = "image_path"
    category_name_key = "category_name"
    category_number_key = "category_number"
    result_key = "result"

    attack_map = {
        brightening_attack_type_name: "perturbBrightening",
        darkening_attack_type_name: "perturbDarkening",
        random_noise_attack_type_name: "perturbRandomNoise"
    }

    def __init__(self, folder_path):
        ActivityInterpreterBase.__init__(self, folder_path)
        self.input_path = folder_path
        self.temp_dir = folder_path
        self.lec_file_name = None
        self.lec_file_hash = None
        self.lec_file_path = None
        self.training_data_zip_file_name = None
        self.training_data_zip_file_hash = None
        self.training_data_zip_file_path = None
        self.training_data_dir_path = None
        self.training_dataset_class = None
        self.verification_data_zip_file_name = None
        self.verification_data_zip_file_hash = None
        self.verification_data_zip_file_path = None
        self.verification_data_dir_path = None
        self.verification_dataset_class = None

    def setup(self):

        import argparse
        import json
        import os

        json_file = os.path.join(
            self.input_path, 'launch_activity_output.json')
        with Path(json_file).open("r") as json_fp:
            input_json_map = json.load(json_fp)

        self.input_map = input_json_map

        attributes = self.input_map[Keys.attributes_key_name]

        self.current_choice = attributes.get(Keys.current_choice_key_name)

        inputs = self.input_map[Keys.inputs_key_name]

        self.temp_dir = self.input_path
        temp_dir_path = Path(self.temp_dir.name)

        # GET INPUT LEC FILE INFO
        input_lec_asset = inputs.get(self.input_lec_key_name).get(
            Attributes.asset_attribute_name)
        self.lec_file_name = input_lec_asset.get(self.name_key_name)
        self.lec_file_hash = input_lec_asset.get(Keys.hash_key_name)
        self.lec_file_path = Path(temp_dir_path, self.lec_file_name).absolute()

        # GET TRAINING DATASET ZIP FILE INFO
        input_training_data_map = inputs.get(self.input_training_data_key_name)
        input_training_data_asset = input_training_data_map.get(
            Attributes.asset_attribute_name)
        self.training_data_zip_file_name = input_training_data_asset.get(
            self.name_key_name)
        self.training_data_zip_file_hash = input_training_data_asset.get(
            Keys.hash_key_name)
        self.training_data_zip_file_path = Path(
            temp_dir_path, self.training_data_zip_file_name).absolute()
        self.training_data_dir_path = Path(
            temp_dir_path, self.training_data_dir_name).absolute()

        input_training_data_parameters = input_training_data_map.get(
            self.parameters_key_name)
        training_data_params_table = input_training_data_parameters.get(
            self.params_table_key_name)
        training_data_dataset_code = training_data_params_table.get(
            self.data_set_key_name)
        self.training_dataset_class = self.get_dataset_class(
            training_data_dataset_code)

        # GET TEST DATASET ZIP FILE INFO
        input_verification_data_map = inputs.get(
            self.input_verification_data_key_name)
        input_verification_data_asset = input_verification_data_map.get(
            Attributes.asset_attribute_name)
        self.verification_data_zip_file_name = input_verification_data_asset.get(
            self.name_key_name)
        self.verification_data_zip_file_hash = input_verification_data_asset.get(
            Keys.hash_key_name)
        self.verification_data_zip_file_path = Path(
            temp_dir_path, self.verification_data_zip_file_name).absolute()
        self.verification_data_dir_path = Path(
            temp_dir_path, self.verification_data_dir_name)

        input_verification_data_parameters = input_verification_data_map.get(
            self.parameters_key_name)
        verification_data_params_table = input_verification_data_parameters.get(
            self.params_table_key_name)
        verification_data_dataset_code = verification_data_params_table.get(
            self.data_set_key_name)

        self.verification_dataset_class = self.get_dataset_class(
            verification_data_dataset_code)

        self.setup_download()

    def setup_download(self):
        #
        # DOWNLOAD FILES
        #
        # LEC
        url = urlunsplit(['http', "{0}:{1}".format(
            url_hostname, alc_port), "/rest/blob/download/", None, None])

        lec_file_url = urljoin(
            urljoin(url, self.lec_file_hash + "/"), self.lec_file_name)
        urllib.request.urlretrieve(lec_file_url, str(self.lec_file_path))

        # TRAINING DATA
        training_data_zip_file_url = \
            urljoin(urljoin(url, self.training_data_zip_file_hash + "/"),
                    self.training_data_zip_file_name)
        urllib.request.urlretrieve(training_data_zip_file_url, str(
            self.training_data_zip_file_path))

        self.training_data_dir_path.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(str(self.training_data_zip_file_path)) as training_data_zip:
            training_data_zip.extractall(str(self.training_data_dir_path))

        # VERIFICATION DATA
        verification_data_zip_file_url = \
            urljoin(urljoin(url, self.verification_data_zip_file_hash +
                            "/"), self.verification_data_zip_file_name)
        urllib.request.urlretrieve(verification_data_zip_file_url, str(
            self.verification_data_zip_file_path))

        self.verification_data_dir_path.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(str(self.verification_data_zip_file_path)) as verification_data_zip:
            verification_data_zip.extractall(
                str(self.verification_data_dir_path))

    def execute(self):
        # execute based on job type
        ret = {}

        from alc_utils.routines import nodeEnv
        from alc_utils.routines import verDep

        depDict = verDep.dep_dict

        from alc_utils import execution_runner
        eParams = json.loads(params)
        depDict["base_dir"] = self.input_path

        print("*********** STARTING EXPERIMENT IN DOCKER *************")
        configfilename = os.path.join(depDict['base_dir'], 'config.json')
        with open(configfilename, 'w') as outfile:
            json.dump(depDict, outfile)

        runner = execution_runner.ExecutionRunner(configfilename)
        result, resultdir = runner.run()

        import alc_utils
        if (result == 0):
            ret["exptParams"] = self.input_map
            ret["directory"] = depDict["base_dir"]
            jupyterworkdir = alc_utils.config.JUPYTER_WORK_DIR
            result_file = os.path.join(depDict['base_dir'], 'robustness.ipynb')
            result_url = result_file[len(jupyterworkdir)+1:]
            ret["result_url"] = 'ipython/notebooks/'+result_url

        return ret

    # method invoked to run the jobs
    def run(folder_name):
        pass
