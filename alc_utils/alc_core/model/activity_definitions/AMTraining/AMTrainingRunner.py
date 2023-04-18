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
import shutil



class AMTrainingRunner:

    alc_working_dir_env = "ALC_WORKING_DIR"
    jupyter_name        = "jupyter"
    notebook_dir        = "result"
    

    input_parentlec_key_name       = "lec_model"
    input_training_data_key_name   = "TrainingData"
    input_test_data_key_name       = "CalibrationData"
    input_validation_data_key_name = "ValidationData"
    metadata_directory_key_name    = "directory"

    model_context_key_name    = "AM_Model"
    model_parameter_key_name    = "Model"
    model_definition_key_name = "model_definition"
    lec_model_definition_key_name   = "lec_model_definition"
    input_shape_definition_key_name = "input_shape"
    data_processing_key_name  = "DataProcessing"
    data_formatter_key_name   = "formatter"
    data_loader_key_name      = "custom_loader"
    dataset_key_name          = "dataset_name"
    trainingparams_key_name   = "AMTrainingParams"
    data_processing_key_name  = "DataProcessing"

    am_type_key_name          = "assurance_monitor_type"

    data_loader_filename        = "data_loader.py"
    data_formatter_filename     = "data_formatter.py"
    am_folder_name              = "LEC_Model"
    am_definition_filename      = "am_net.py"
    lec_definition_filename     = "LECModel.py"



    result_filename              = "result.json"   

    def __init__(self):
        self.input_path = os.path.abspath('.')
        self.temp_dir = os.path.abspath('.')

        self.inputs = {}
        self.attributes = None

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
        self.am_definition = None
        self.am_definition_path = None
        self.am_definition_folder = None
        self.data_formatter_path = None
        self.data_loader_path = None
        self.current_choice = None


        self.alc_working_dir = None
        self.mesgs = []
        self.output_dir = None
    
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
    
    # def get_am_lec_definition(self):
    #     am_definition = ''
    #     lec_definition = ''
    #     context = self.input_map.get(Keys.context_key_name)
    #     if (context):
    #         model_context = context.get(self.model_context_key_name)
    #         if (model_context):
    #             model_content = model_context.get(Keys.content_key_name)
    #             if (model_content):
    #                 model_definition = model_content.get(self.model_definition_key_name)
    #                 if (model_definition):
    #                     am_definition = model_definition.get(Attributes.definition_attribute_name)
    #                 lec_model_definition = model_content.get(self.lec_model_definition_key_name)
    #                 if (lec_model_definition):
    #                     lec_definition = lec_model_definition.get(Attributes.definition_attribute_name)
    #                 input_shape_definition = model_content.get(self.input_shape_definition_key_name)
    #                 if (input_shape_definition):
    #                     input_shape = input_shape_definition.get(Attributes.definition_attribute_name)
    #                     input_shape = eval(input_shape)

    def get_am_lec_definition(self):
        am_definition = ''
        lec_definition = ''
        input_shape = None
        parameters = self.input_map.get(Keys.parameters_key_name)
        if (parameters):
            model_params = parameters.get(self.model_parameter_key_name)
            if model_params:
                am_definition = model_params.get(self.model_definition_key_name)
                lec_definition = model_params.get(self.lec_model_definition_key_name)
                input_shape = model_params.get(self.input_shape_definition_key_name)
        return am_definition, lec_definition, input_shape

    
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

    def getdirwithin(self,folder):
        print ('folder ',str(folder))
        # Check if RL file exists and return path if so
        rlfile = os.path.join(folder, 'RLAgent.py')
        if os.path.exists(rlfile):
            return folder

        kerasfile = os.path.join(folder, 'model.keras')
        if os.path.exists(kerasfile):
            return folder

        kerasfile = os.path.join(folder, 'model.h5')
        if os.path.exists(kerasfile):
            return folder

        kerasfile = os.path.join(folder, 'model_weights.h5')
        if os.path.exists(kerasfile):
            return folder
        
        modelfile = os.path.join(folder, 'model.pkl')
        if os.path.exists(modelfile):
            return folder

        amfile = os.path.join(folder, 'assurancemonitor.pkl')
        if os.path.exists(amfile):
            return folder

        contents = os.listdir(folder)

        for l in contents:
            if (l.find('.ipynb_checkpoints') >=0):
                continue

            fname = os.path.join(folder, l)
            if os.path.isdir(fname):
                return fname
        return ''

    def copyModelDir(self,model_dir, cur_dir, force=False):
        if (model_dir):
            model_folder = self.getdirwithin(model_dir)
            base_name = os.path.basename(model_folder)
            if base_name != 'RLModel':
                base_name = 'SLModel'
        else:
            base_name ='SLModel'
        destdir = os.path.join(cur_dir, base_name)
        if (base_name == 'SLModel'):
            if (not os.path.exists(destdir)):
                os.makedirs(destdir)
            if (not force):
                return destdir
        
        if (not model_dir):
            return destdir

        srcdir = model_folder
        if  (not srcdir or not os.path.exists(srcdir)):
            return destdir
        print('src dir : ' + srcdir)
        print('dst dir : ' + destdir)
        try:
            from distutils.dir_util import copy_tree
            copy_tree(srcdir, destdir)
        except OSError as exc:  # python >2.5
            if exc.errno == errno.ENOTDIR:
                shutil.copy(srcdir, destdir)
            else:
                raise
        return destdir
    
    def write_to_file(self, file_path,contents):
        if (not contents):
            return
        f = open(file_path,'w')
        f.write(contents)
        f.close()

        
    def create_files(self):
        self.create_folder(self.am_definition_folder)
        self.write_to_file(self.am_definition_path, self.am_definition)
        self.write_to_file(self.lec_definition_path, self.lec_definition)
        self.write_to_file(self.data_formatter_path, self.formatter)
        if (self.loader):
            self.write_to_file(self.data_loader_path, self.loader)
        
        cur_dir = self.input_path
        output_dir = self.copyModelDir(self.parent_lec, cur_dir, True)

        print('output dir ' + str(output_dir))
        
        
        if (os.path.exists(self.data_formatter_path)):
            data_formatter_dstpath = os.path.join(output_dir,'data_formatter.py')
            shutil.copyfile(self.data_formatter_path, data_formatter_dstpath)

        
        if (os.path.exists(self.am_definition_path)):
            am_defn_dstpath = os.path.join(output_dir,'am_net.py')
            shutil.copyfile(self.am_definition_path, am_defn_dstpath)
        
        if (os.path.exists(self.lec_definition_path)):
            lec_defn_dstpath = os.path.join(output_dir,'LECModel.py')
            shutil.copyfile(self.lec_definition_path, lec_defn_dstpath)
        
        return output_dir
    
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
        self.attributes = self.input_map[Keys.attributes_key_name]


        self.training_data, self.training_metadata     = self.get_input_data_dirs(self.input_training_data_key_name)
        self.test_data, self.test_metadata             = self.get_input_data_dirs(self.input_test_data_key_name)
        self.validation_data, self.validation_metadata = self.get_input_data_dirs(self.input_validation_data_key_name)
        self.parent_lec, self.parent_lec_metadata      = self.get_input_data_dirs(self.input_parentlec_key_name)
        if (self.parent_lec):
            self.parent_lec = self.parent_lec[0]
        if (self.parent_lec_metadata):
            self.parent_lec_metadata = self.parent_lec_metadata[0]

        self.dataset_name, self.loader, self.formatter = self.get_data_processing_parameters()
        self.training_params = self.get_training_parameters()
        self.am_definition, self.lec_definition, self.input_shape = self.get_am_lec_definition()

        self.current_choice = self.attributes[Keys.current_choice_key_name]
        self.training_params[self.am_type_key_name] = self.current_choice

        self.am_definition_folder = os.path.join(self.input_path, self.am_folder_name)
        self.am_definition_path   = os.path.join(self.am_definition_folder, self.am_definition_filename)
        self.lec_definition_path   = os.path.join(self.am_definition_folder, self.lec_definition_filename)
        self.data_formatter_path   = os.path.join(self.am_definition_folder, self.data_formatter_filename)
        self.data_loader_path      = os.path.join(self.am_definition_folder, self.data_loader_filename)
        self.output_dir = self.create_files()



    # def execute(self):
    #     # execute based on job type
    #     ret = {}
    #     #from alc_utils.routines import run_training
    #     from alc_utils.common import dict_convert_key_case, load_python_module
    #     from alc_utils.routines.setup import createResultNotebook2

    #     self.training_params['dataset_name'] = self.dataset_name
    #     param_dict = dict_convert_key_case(self.training_params, "lower")
        

    #     if self.dataset_name.lower() == "custom":
    #         if self.loader is None:
    #             raise IOError("Parameters specify a CUSTOM dataset, but no custom dataset code was found.")
    #         param_dict['dataset_name'] = self.data_loader_path

        
    #     print("{}".format(param_dict))
    #     if (self.validation_metadata and len(self.validation_metadata) == 0):
    #         self.validation_metadata = None

    #     if (self.test_metadata and  len(self.test_metadata) == 0):
    #         self.test_metadata = None

        
    #     output_dir = self.output_dir#self.am_definition_folder

    #     if not self.training_data:
    #         print(' no training data')
        
    #     if not self.am_definition :
    #         print(' no am definition')

    #     if not output_dir:
    #         print(' no output dir')

    #     if not self.training_data or not self.am_definition or not output_dir:
    #         ret = {"status": "check training data, model,output_dir"}
    #         return ret

    #     print('training data ', self.training_data)
        
        
    #     from alc_utils.routines import train_assurance_monitor
    #     #ret = train_assurance_monitor.run_assurance_monitor_training(param_dict, 
    #     #                                                            self.training_data, 
    #     #                                                            self.am_definition_folder,
    #     #                                                            output_dir, 
    #     #                                                            am_data_formatter_path=self.data_formatter_path,
    #     #                                                            validation_data_dirs=self.validation_data,
    #     #                                                            testing_data_dirs = self.test_data)
    #     ret = {}
    #     createResultNotebook2(output_dir)
    #     full_output_dir = os.path.abspath(output_dir)
    #     relative_folder_path = full_output_dir[ len(JUPYTER_WORK_DIR)+1:]
    #     ret['directory'] = full_output_dir
    #     ret['result_url'] = os.path.join('ipython','notebooks',relative_folder_path,'result.ipynb')
    #     if (not (ret.get('exptParams'))):
    #         ret['exptParams']= self.training_params
    #     return ret

    def execute(self):
        # execute based on job type
        from alc_utils.routines.setup import createResultNotebook2
        ret = {}
        
        from alc_utils.common import dict_convert_key_case
        param_dict = dict_convert_key_case(self.training_params, "lower")
        param_dict['dataset_name'] = self.data_loader_path
        
        if (self.input_shape):
            param_dict[self.input_shape_definition_key_name] = tuple(self.input_shape)
        
        #param_dict[self.am_type_key_name] = self.am_type

        #if (self.parent_model):
        #    param_dict[self.training_parent_model_key_name] = self.parent_model

        
        
        

        from alc_utils.routines import train_ood_detectors
        result_output = train_ood_detectors.run_ood_detector_training(param_dict, 
                                                                    self.training_data, 
                                                                    self.am_definition_folder,
                                                                    self.output_dir, 
                                                                    am_data_formatter_path=self.data_formatter_path,
                                                                    validation_data_dirs=self.validation_data,
                                                                    testing_data_dirs = self.test_data)
        
        
    
        ret = {}
        output_dir = self.output_dir
        createResultNotebook2(output_dir)
        full_output_dir = os.path.abspath(output_dir)
        relative_folder_path = full_output_dir[ len(JUPYTER_WORK_DIR)+1:]
        ret['directory'] = str(full_output_dir)
        ret['result_url'] = os.path.join('ipython','notebooks',relative_folder_path,'result.ipynb')
        if (not (ret.get('exptParams'))):
            ret['exptParams']= self.training_params

        with Path(self.input_path, self.result_filename).open("w",encoding="utf-8") as json_fp:
            json_fp.write(json.dumps(ret, indent=4, sort_keys=True,ensure_ascii=False))

        return ret
        
    

if __name__ == '__main__':
    trainer = AMTrainingRunner()
    trainer.setup()
    result_output = trainer.execute()
    print(result_output)
    
    