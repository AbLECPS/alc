#!/usr/bin/env python
# This contains python code for executing ALC jobs
# Based on the jobtype, it invokes the appropriate functions in alc_utils.
from __future__ import print_function

import os
import json
import sys
import tempfile
from pathlib import Path
import os
import re
import itertools
from alc_utils.LaunchActivity.ActivityInterpreterBase import ActivityInterpreterBase
from alc_utils.LaunchActivity.KeysAndAttributes import Keys, Attributes


class ActivityInterpreter(ActivityInterpreterBase):

    alc_home_env_var_name = "ALC_HOME"

    training_data_dir_name = "TrainingData"
    verification_data_dir_name = "VerificationData"
    notebook_dir_name = "Notebook"

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

    @staticmethod
    def get_dataset_class(dataset_class_string):
        match = re.search(r"class\s+(\w+)", dataset_class_string)
        class_name = match.group(1)
        new_class_name = class_name
        ix = 0
        while new_class_name in globals():
            new_class_name = class_name + str(ix)
            ix += 1
        dataset_class_string = re.sub(
            r"class\s+{0}".format(class_name), "class {0}".format(new_class_name), dataset_class_string, count=1
        )
        exec(dataset_class_string, globals())
        return globals().get(new_class_name)

    def setup(self):

        import argparse
        import json
        import os

        json_file = os.path.join(
            self.input_path, 'launch_activity_output.json')
        with Path(json_file).open("r") as json_fp:
            input_json_map = json.load(json_fp)

        self.input_map = input_json_map

        self.attributes = self.input_map[Keys.attributes_key_name]
        self.current_choice = self.attributes.get(Keys.current_choice_key_name)

        inputs = self.input_map[Keys.inputs_key_name]

        # GET INPUT LEC FILE INFO
        input_lec_map = inputs.get(self.input_lec_key_name)
        self.lec_file_path = Path(input_lec_map.get(
            Attributes.asset_attribute_name)).absolute()
        self.lec_file_node_path = input_lec_map.get(Keys.node_path_key_name)
        self.lec_file_node_named_path = input_lec_map.get(
            Keys.node_named_path_key_name)

        # GET TRAINING DATASET ZIP FILE INFO
        input_training_data_map = inputs.get(self.input_training_data_key_name)
        self.input_training_data_zip_file_path = Path(
            input_training_data_map.get(Attributes.asset_attribute_name)
        ).absolute()
        self.input_training_data_node_path = input_training_data_map.get(
            Keys.node_path_key_name)
        self.input_training_data_node_named_path = input_training_data_map.get(
            Keys.node_named_path_key_name)

        # GET "DataSet" CODE FOR TRAINING DATASET
        input_training_data_parameters = input_training_data_map.get(
            self.parameters_key_name)
        training_data_params_table = input_training_data_parameters.get(
            self.params_table_key_name)
        training_data_dataset_code = training_data_params_table.get(
            self.data_set_key_name)
        self.input_training_dataset_class = self.get_dataset_class(
            training_data_dataset_code)

        # GET TEST DATASET ZIP FILE INFO
        input_verification_data_map = inputs.get(
            self.input_verification_data_key_name)
        self.input_verification_data_zip_file_path = Path(
            input_verification_data_map.get(Attributes.asset_attribute_name)
        ).absolute()
        self.input_verification_data_node_path = input_verification_data_map.get(
            Keys.node_path_key_name)
        self.input_verification_data_node_named_path = input_verification_data_map.get(
            Keys.node_named_path_key_name)

        # GET "DataSet" CODE FOR VERIFICATION DATASET
        input_verification_data_parameters = input_verification_data_map.get(
            self.parameters_key_name)
        verification_data_params_table = input_verification_data_parameters.get(
            self.params_table_key_name)
        verification_data_dataset_code = verification_data_params_table.get(
            self.data_set_key_name)
        self.input_verification_dataset_class = self.get_dataset_class(
            verification_data_dataset_code)

        # GET METHOD
        parameters = self.input_map[self.parameters_key_name]
        misc_parameters = parameters.get(self.params_table_key_name)
        self.method = misc_parameters.get(self.method_key_name)

        self.extra_parameter_map = parameters.get(self.current_choice)

    def execute(self):
        # execute based on job type
        ret = {}

        from alc_utils.routines import nodeEnv
        from alc_utils.routines.Verification_Robustness import Dep

        depDict = Dep.dep_dict

        from alc_utils import execution_runner
        #eParams = json.loads(params)
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
            result_file = os.path.join(
                depDict['base_dir'], self.notebook_dir_name, 'robustness.ipynb')
            result_url = result_file[len(jupyterworkdir)+1:]
            ret["result_url"] = 'ipython/notebooks/'+result_url

        return ret

    # method invoked to run the jobs
    def run(self):
        self.setup()
        self.execute()
