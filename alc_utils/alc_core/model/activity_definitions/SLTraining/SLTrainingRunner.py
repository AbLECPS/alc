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
from alc_utils.config import WORKING_DIRECTORY, JUPYTER_WORK_DIR




class SLTrainingRunner:

    alc_working_dir_env = "ALC_WORKING_DIR"
    jupyter_name        = "jupyter"
    notebook_dir        = "result"

    input_parentlec_key_name       = "ParentLECModel"
    input_training_data_key_name   = "TrainingData"
    input_test_data_key_name       = "TestData"
    input_validation_data_key_name = "ValidationData"
    metadata_directory_key_name    = "directory"

    model_context_key_name    = "Model"
    model_definition_key_name = "model_definition"
    data_processing_key_name  = "DataProcessing"
    data_formatter_key_name   = "formatter"
    data_loader_key_name      = "custom_loader"
    dataset_key_name          = "dataset_name"
    trainingparams_key_name   = "TrainingParams"
    data_processing_key_name  = "DataProcessing"

    data_loader_filename         = "data_loader.py"
    data_formatter_filename      = "data_formatter.py"
    lec_folder_name              = "LEC_Model"
    lec_definition_filename      = "LECModel.py"

    result_filename              = "result.json"   

    def __init__(self):
        self.input_path = '.'
        self.temp_dir = '.'

        self.inputs = {}

        self.parent_lec = None
        self.parent_lec_metadata = None
        
        self.training_data = []
        self.test_data     = []
        self.validation_data = []

        self.training_metadata = []
        self.test_metadata     = []
        self.validation_metadata = []

        self.formatter = None
        self.loader = None
        self.dataset_name = None
        self.lec_definition = None
        self.lec_definition_path = None
        self.lec_definition_folder = None
        self.data_formatter_path = None
        self.data_loader_path = None


        self.alc_working_dir = None
        self.mesgs = []
    
    def update_dir(self, dir):
        folder_path = dir
        pos = folder_path.find(self.jupyter_name)
        if (pos > -1):
            folder_path = folder_path[pos:]
        return os.path.join(self.alc_working_dir,folder_path)

    def get_input_data_dirs(self, key):
        dirs = []
        metadata = []

        result = self.inputs.get(key)
        if (not result):
            self.mesgs.append(" no input of {0} found ".format(key))
            return dirs, metadata
        
        metadata = result.get(Keys.input_set_name)
        if (not metadata):
            self.mesgs.append(" empty input set for {0} found ".format(key))
            return dirs, metadata

        
        ret_metadata = []
        for md in metadata:
            dir = md.get(self.metadata_directory_key_name)
            if (not dir):
                self.mesgs.append("Directory entry not found in metadata")
                continue
            updated_dir = self.update_dir(dir)
            dir_path = Path(updated_dir).absolute()
            if (not dir_path.exists()):
                self.mesgs.append('directory path {0} not found in input {1}'.format(updated_dir,key))
                continue
            md[self.metadata_directory_key_name] = str(dir_path)
            ret_metadata.append(md)
            dirs.append(str(dir_path))

        return dirs, metadata
    
    # def get_lec_definition(self):
    #     filename = ''
    #     lec_definition = ''
    #     context = self.input_map.get(Keys.context_key_name)
    #     if (context):
    #         model_context = context.get(self.model_context_key_name)
    #         if (model_context):
    #             model_content = model_context.get(Keys.content_key_name)
    #             if (model_content):
    #                 model_definition = model_content.get(self.model_definition_key_name)
    #                 if (model_definition):
    #                     lec_definition = model_definition.get(Attributes.definition_attribute_name)
    #                     filename = model_definition.get(Attributes.filename_attribute_name)
        
    #     return lec_definition, filename

    def get_lec_definition(self):
        lec_definition = ''
        parameters = self.input_map.get(Keys.parameters_key_name)
        if (parameters):
            model_params = parameters.get(self.model_context_key_name)
            if model_params:
                lec_definition = model_params.get(self.model_definition_key_name)
        return lec_definition

    
    def get_data_processing_parameters(self):
        dataset_name = ''
        data_formatter = ''
        data_loader = ''
        parameters = self.input_map.get(Keys.parameters_key_name)
        if parameters:
            data_processing_params = parameters.get(self.data_processing_key_name)
            if (data_processing_params):
                data_formatter = data_processing_params.get(self.data_formatter_key_name)
                data_loader    = data_processing_params.get(self.data_loader_key_name)
                dataset_name   = data_processing_params.get(self.dataset_key_name)
        
        return dataset_name, data_loader, data_formatter

    

    def get_training_parameters(self):
        training_params = {}
        parameters = self.input_map.get(Keys.parameters_key_name)
        if parameters:
            training_params = parameters.get(self.trainingparams_key_name)
        return training_params

    def create_folder(self, folder_path):
        x = Path(folder_path)
        if (x.exists()):
            return
        x.mkdir()
    
    def write_to_file(self, file_path,contents):
        f = open(file_path,'w')
        f.write(contents)
        f.close()

        
    def create_files(self):
        self.create_folder(self.lec_definition_folder)
        self.write_to_file(self.lec_definition_path, self.lec_definition)
        self.write_to_file(self.data_formatter_path, self.formatter)
        if (self.dataset_name == 'custom' and self.loader):
            self.write_to_file(self.data_loader_path, self.loader)
    
    def createTrainingOutputDirectory(self):
        import time
        x = time.strftime("%Y_%m_%d_%H_%M_%S", time.gmtime())
        folder_path = os.path.join(self.input_path,"TrainingResult_{0}".format(x))
        self.create_folder(folder_path)
        return folder_path



    def setup(self):

        import argparse
        import json
        import os

        
        

        json_file = os.path.join(self.input_path, 'launch_activity_output.json')
        with Path(json_file).open("r") as json_fp:
            input_json_map = json.load(json_fp)


        self.alc_working_dir = os.getenv(self.alc_working_dir_env)
        if not self.alc_working_dir:
            raise Exception("environment variable {0} not found".format(self.alc_working_dir_env))
        
        
        self.input_map = input_json_map
        self.inputs = self.input_map[Keys.inputs_key_name]


        self.training_data, self.training_metadata     = self.get_input_data_dirs(self.input_training_data_key_name)
        self.test_data, self.test_metadata             = self.get_input_data_dirs(self.input_test_data_key_name)
        self.validation_data, self.validation_metadata = self.get_input_data_dirs(self.input_validation_data_key_name)
        self.parent_lec, self.parent_lec_metadata      = self.get_input_data_dirs(self.input_parentlec_key_name)

        self.dataset_name, self.loader, self.formatter = self.get_data_processing_parameters()
        self.training_params = self.get_training_parameters()
        #self.lec_definition, filename = self.get_lec_definition()
        self.lec_definition = self.get_lec_definition()

        self.lec_definition_folder = os.path.join(self.input_path, self.lec_folder_name)
        self.lec_definition_path   = os.path.join(self.lec_definition_folder, self.lec_definition_filename)
        self.data_formatter_path   = os.path.join(self.lec_definition_folder, self.data_formatter_filename)
        self.data_loader_path      = os.path.join(self.lec_definition_folder, self.data_loader_filename)
        self.create_files()



    def execute(self):
        # execute based on job type
        ret = {}
        from alc_utils.routines import run_training
        from alc_utils.common import dict_convert_key_case, load_python_module
        from alc_utils.routines.setup import createResultNotebook2

        self.training_params['dataset_name'] = self.dataset_name
        param_dict = dict_convert_key_case(self.training_params, "lower")
        

        if self.dataset_name == "custom":
            if self.loader is None:
                raise IOError("Parameters specify a CUSTOM dataset, but no custom dataset code was found.")
            param_dict['dataset_name'] = self.data_loader_path

        

        if (not self.validation_metadata or len(self.validation_metadata) == 0):
            self.validation_metadata = None

        if (not self.test_metadata or len(self.test_metadata) == 0):
            self.test_metadata = None

        if self.training_data and self.lec_definition_path:
            output_dir = self.createTrainingOutputDirectory()

        if not self.training_data or not self.lec_definition or not output_dir:
            ret = {"status": "check training data, model"}
            return ret
        
        

        model_module = None
        if os.path.exists(self.lec_definition_path):
            model_module = load_python_module(self.lec_definition_path)
            optimizer = param_dict.get("optimizer")
            if optimizer and optimizer.upper() == 'CUSTOM':
                optimizer = model_module.get_optimizer(**param_dict)
                param_dict["optimizer"] = optimizer
            loss = param_dict.get("loss")
            if loss and loss.upper() == 'CUSTOM':
                loss = model_module.get_loss(**param_dict)
                param_dict["loss"] = loss
            metrics = param_dict.get("metrics")
            if metrics and isinstance(metrics, str)  and metrics.upper() == 'CUSTOM':
                metrics = model_module.get_metrics(**param_dict)
                param_dict["metrics"] = metrics
            callbacks = param_dict.get("callbacks")
            if callbacks and callbacks.upper() == 'CUSTOM':
                callbacks = model_module.get_callbacks(**param_dict)
                param_dict["callbacks"] = callbacks

        ret = run_training.run_training(param_dict, self.training_metadata, self.training_data, self.lec_definition_folder, output_dir,
                                        self.parent_lec, self.validation_metadata, self.test_metadata)
        ret= {}
        createResultNotebook2(output_dir)
        full_output_dir = os.path.abspath(output_dir)
        relative_folder_path = full_output_dir[ len(JUPYTER_WORK_DIR)+1:]
        ret['directory'] = full_output_dir
        ret['result_url'] = os.path.join('ipython','notebooks',relative_folder_path,'result.ipynb')
        if (not (ret.get('exptParams'))):
            ret['exptParams']= self.training_params
        return ret

if __name__ == '__main__':
    trainer = SLTrainingRunner()
    trainer.setup()
    result_output = trainer.execute()
    with Path(trainer.input_path, trainer.result_filename).open("w",encoding="utf-8") as json_fp:
        json_fp.write(json.dumps(result_output, indent=4, sort_keys=True,ensure_ascii=False))