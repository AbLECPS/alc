#!/usr/bin/env python
# This contains python code for executing ALC jobs
# Based on the jobtype, it invokes the appropriate functions in alc_utils.
from __future__ import print_function

import json
import sys
import shutil
from pathlib import Path
import yaml
from alc_utils.LaunchActivity.ActivityInterpreterBase import ActivityInterpreterBase
from alc_utils.LaunchActivity.KeysAndAttributes import Keys
from alc_utils.config import WORKING_DIRECTORY


class ActivityInterpreter(ActivityInterpreterBase):

    alc_working_dir_env           = "ALC_WORKING_DIR"
    jupyter_name                  = "jupyter"
    notebook_dir                  = "results"
    notebook_filename             = "result.ipynb"
    notebook_template_filename    = "resultnb"
    postprocess_script_filename   = 'postprocess.py'         

    input_lec2_model_key_name      = "lec2"
    input_lec3_model_key_name      = "lec3"
    input_lecdd_model_key_name     = "lec_dd"
    
    
    
    
    metadata_directory_key_name    = "directory"

    postprocessing_key_name        = "PostProcessScript"
    postprocessing_param_name      = "PostProcess"

    execution_param_key            = "Execution"
    pipe_param_key                 = "Pipe"
    degradation_param_key          = "Degradation"
    obstacle_detection_param_key   = "Obstacles"
    continency_manager_param_key   = "Autonomy"
    
    mission_param_key              = "Mission"
    initial_state_param_key        = "InitialState"
    rtreach_param_key              = "RTReach"
    waypoint_param_key             = "Waypoint"
    degradation_detection_param_key = "FDIR"
    

    timeout_param_key              = "timeout"

    lec2_deployment_key            = "lec_model_dir"
    lecdd_deployment_key           = "fdir_path"
    lecdd_model_param_key_name     = "fdir_params"

    snapshot_am_choice_key = "snapshot_am_choice"
    trained_best_key     = "trained_best"
    override_threshold_key = "override_threshold"
    snapshot_am_threshold_key = "snapshot_am_threshold"
    snapshot_am_threshold_param_key = "am_threshold"
    combination_am_choice_key = "comb_am_choice"
    combination_am_user_choice_param_key = "user_choice"
    combination_am_threshold_key = "combined_am_threshold"
    combination_am_threshold_param_key ="am_s_threshold"
    combination_am_windowsize_key = "comb_am_window_size"
    combination_am_windowsize_param_key = "window_size"
    combination_am_type_key ="comb_type"
    combination_am_merge_key ="comb_fun_merge"
    combination_am_cdf_key ="comb_fun_cdf"
    combination_am_merge_type_key ="merge"
    combination_am_cdf_type_key ="cdf"
    combination_am_function_param_key = "comb_function"

    launch_cmd_key                 = "cmd"
    launch_param_key               = "params"
    launch_execute_key             = "execute"
    launch_id_key                  = "id"
    sim_launch_cmd_key             = "roslaunch vandy_bluerov start_bluerov_simulation.launch"
    random_cmd                     = '$RANDOM'
    random_seed_key                = 'random_seed'
    random_seed_key_alt            = 'random_val'
    disturbance_filename_key       = 'disturbance_filename'
    sim_scenario_runner_name       = 'sim_scenario_runner'
    sim_lec_runner_name            = 'sim_lec_runner'
    sim_param_filename             = 'parameters.yml'
    exec_config_filename           = 'config.json'
    config_foldername              = 'config-'
    config_id                      = 0

    result_error_key               = 'errors'
    result_url_key                 = 'result_url'
    result_expt_params_key         = 'exptParams'

    deployment_base_dir_key        = 'base_dir'
    activity_home_dir_key          = 'activity_home'

    parameter_list_keys = [
        execution_param_key,
        obstacle_detection_param_key,
        degradation_param_key,
        pipe_param_key,
        continency_manager_param_key,
        mission_param_key,
        rtreach_param_key,
        waypoint_param_key,
        degradation_detection_param_key
    ]
    # sim_parameter_keys = [
    #     execution_param_key,
    #     obstacle_detection_param_key,
    #     degradation_param_key,
    #     pipe_param_key,
    #     continency_manager_param_key,
    #     mission_param_key,
    #     rtreach_param_key,
    #     waypoint_param_key,
    #     degradation_detection_param_key
    # ]
    sim_parameter_keys = [
        execution_param_key,
        continency_manager_param_key,
        mission_param_key
    ]


    def __init__(self, input_dir_path):
        ActivityInterpreterBase.__init__(self, input_dir_path)
        self.temp_dir = input_dir_path

        self.inputs = {}
        self.attributes = None
        self.lec2_model = None
        self.lec3_model = None
        self.lecdd_model = None
        self.parameters = {}
        self.post_processing_script = None
        self.alc_working_dir = Path(WORKING_DIRECTORY)
        self.mesgs = []
        self.error_mesgs = []
        self.sim_parameters  = {}
        self.exec_folder_path = ''
        self.relative_folder_path = ''

    def update_dir(self, local_dir):
        folder_path = local_dir
        folder_path_parts_tuple = folder_path.parts
        if self.jupyter_name in folder_path_parts_tuple:
            pos = folder_path_parts_tuple.index(self.jupyter_name)
            folder_path = Path(*folder_path_parts_tuple[pos:])
        return Path(self.alc_working_dir, folder_path)

    def get_input_data_dirs(self, key):
        dirs = []
        metadata = []

        result = self.inputs.get(key)
        if not result:
            self.mesgs.append(" no input of {0} found ".format(key))
            return dirs, metadata
        
        metadata = result.get(Keys.input_set_name)
        if not metadata:
            self.mesgs.append(" empty input set for {0} found ".format(key))
            return dirs, metadata

        ret_metadata = []
        for md in metadata:
            local_dir = Path(md.get(self.metadata_directory_key_name))
            if not local_dir:
                self.mesgs.append("Directory entry not found in metadata")
                continue
            updated_dir = self.update_dir(local_dir)
            dir_path = updated_dir.absolute()
            if not dir_path.exists():
                self.mesgs.append('directory path {0} not found in input {1}'.format(updated_dir, key))
                continue
            md[self.metadata_directory_key_name] = str(dir_path)
            ret_metadata.append(md)
            dirs.append(dir_path)

        return dirs, metadata

    def get_parameters(self):
        parameters = self.input_map.get(Keys.parameters_key_name)
        return parameters

    def get_post_processing_script(self):
        parameters = self.input_map.get(Keys.parameters_key_name)
        if parameters:
            params = parameters.get(self.postprocessing_key_name)
            if params:
                script  = params.get(self.postprocessing_param_name)
                return script
        return None
    
    def build_lec_dd_params(self):
        ret = {}
        if (self.degradation_detection_param_key not in self.parameters.keys()):
            return json.dumps(ret)

        user_choice_snapshot = self.parameters[self.degradation_detection_param_key].get(self.snapshot_am_choice_key,self.trained_best_key)
        snapshot_threshold   = self.parameters[self.degradation_detection_param_key].get(self.snapshot_am_threshold_key,None)
        if (user_choice_snapshot == self.override_threshold_key and  snapshot_threshold):
            ret[self.snapshot_am_threshold_param_key] = snapshot_threshold
        
        user_choice_combined = self.parameters[self.degradation_detection_param_key].get(self.combination_am_choice_key,self.trained_best_key)
        
        ret[self.combination_am_user_choice_param_key] = user_choice_combined
        
        val   = self.parameters[self.degradation_detection_param_key].get(self.combination_am_threshold_key,None)
        if (val):
            ret[self.combination_am_threshold_param_key] = val
        
        val   = self.parameters[self.degradation_detection_param_key].get(self.combination_am_windowsize_key,None)
        if (val):
            ret[self.combination_am_windowsize_param_key] = val
        
        val   = self.parameters[self.degradation_detection_param_key].get(self.combination_am_type_key,None)
        val1   = self.parameters[self.degradation_detection_param_key].get(self.combination_am_merge_key,None)
        val2   = self.parameters[self.degradation_detection_param_key].get(self.combination_am_cdf_key,None)

        if (val == self.combination_am_merge_type_key):
            ret[self.combination_am_function_param_key] = val1
        if (val == self.combination_am_cdf_type_key):
            ret[self.combination_am_function_param_key] = val2

        
        return json.dumps(ret)

        

    
    def assemble_sim_parameters(self):
        print('1')
        print(self.parameter_list_keys)
        sim_keys = [x.lower() for x in self.parameter_list_keys]
        print(sim_keys)
        sim_parameters = {}
        for pkeys in self.parameters.keys():
            if pkeys.lower() not in sim_keys:
                print ('missing '+pkeys.lower())
                continue
            print('available '+ pkeys.lower())
            for param_keys in self.parameters[pkeys].keys():
                sim_parameters[param_keys] = self.parameters[pkeys][param_keys]
        if self.lec2_model:
            sim_parameters[self.lec2_deployment_key] = self.lec2_model
        if self.lecdd_model:
            sim_parameters[self.lecdd_deployment_key] = self.lecdd_model
            sim_parameters[self.lecdd_model_param_key_name] = self.build_lec_dd_params()
            #set sim_parameters for "lecdd_model_param_key_name"
        
        if sim_parameters.get(self.disturbance_filename_key, None) is None:
            sim_parameters[self.disturbance_filename_key] = ''
        
        if sim_parameters.get(self.random_seed_key, None) is None:
            if sim_parameters.get(self.random_seed_key_alt, None):
                sim_parameters[self.random_seed_key] = sim_parameters[self.random_seed_key_alt]
            else:
                sim_parameters[self.random_seed_key] = self.random_cmd
        
        #sim_parameters["small_bagfile"] = False
        
        # self.sim_lec2_parameters = {
        #     self.launch_execute_key: {
        #         self.launch_cmd_key: self.sim_launch_cmd_key,
        #         self.launch_param_key: sim_lec2_parameters
        #     },
        #     self.launch_id_key: self.sim_scenario_runner_name
        # }

        self.sim_parameters = sim_parameters

        full_param_file_name = Path(self.exec_folder_path, self.sim_param_filename)
        
        with full_param_file_name.open('w') as yaml_file:
            yaml.safe_dump(self.sim_parameters, yaml_file, default_flow_style=False)

    
    
    def create_post_process_script(self):
        if not self.post_processing_script:
            return

        filename = Path(self.exec_folder_path, self.postprocess_script_filename)
        with filename.open('w') as fp:
            fp.write(self.post_processing_script)

    def create_deployment_file(self):
        activity_folder = str(Path(__file__).absolute().parent)
        sys.path.append(activity_folder)
        import Dep
        dep_dict = Dep.dep_dict
        dep_dict[self.deployment_base_dir_key] = str(self.relative_folder_path)
        dep_dict[self.activity_home_dir_key] = str(activity_folder)
        execution_params = self.parameters.get(self.execution_param_key)
        if execution_params:
            timeout = execution_params.get(self.timeout_param_key, None)
            if timeout:
                dep_dict[self.timeout_param_key] = timeout

        file_path = Path(self.exec_folder_path, self.exec_config_filename)
        with file_path.open('w') as fp:
            json.dump(dep_dict, fp, indent=4, sort_keys=True)
        
        return file_path

    def create_execution_files(self):
        self.exec_folder_path = Path(self.input_dir_path, self.config_foldername + str(self.config_id))
        if not self.exec_folder_path.is_dir():
            self.exec_folder_path.mkdir(parents=True)
        
        result_folder_path = Path(self.exec_folder_path, self.notebook_dir)
        if not result_folder_path.is_dir():
            result_folder_path.mkdir(parents=True)
        
        self.relative_folder_path = Path(*self.exec_folder_path.parts[len(self.alc_working_dir.parts):])
        self.assemble_sim_parameters()
        self.create_post_process_script()
        return self.create_deployment_file()
    
    def create_result_notebook(self):
        filepath = Path(__file__).absolute()
        filedir  = filepath.parent
        srcpath = Path(filedir, self.notebook_template_filename)
        dstpath = Path(self.exec_folder_path, self.notebook_filename)
        shutil.copy(str(srcpath), str(dstpath))
        pos = dstpath.parts.index(self.jupyter_name)
        localdstpath = Path(*dstpath.parts[pos+1:])
        return localdstpath

    def check_parameters(self):
        mesg = []
        ret = True
        #param_keys = map(lambda x: x.lower(), self.parameters.keys())
        param_keys = [x.lower() for x in self.parameters.keys()]
        print (self.sim_parameter_keys)
        print(param_keys)
        for pkeys in self.sim_parameter_keys: #self.parameter_list_keys:
            if pkeys.lower() in param_keys:
                continue
            mesg.append('Missing parameter list for {0} '.format(pkeys))
            ret = False

        return ret, mesg

    def check(self):
        if not self.lec2_model:
            print("*********** no user specified lec2. using default *************")
        if not self.lec3_model:
            print("*********** no user specified lec3. using default *************")
        if not self.lecdd_model:
            print("*********** no user specified lecdd. using default *************")
        if not self.post_processing_script:
            print("*********** no post processing script *************************")
        ret, pmesgs = self.check_parameters()
        if not ret:
            self.error_mesgs.extend(pmesgs)
            print("Issues with parameters:\n  ", '\n'.join(pmesgs))
        return ret

    def setup(self):
        if not self.alc_working_dir:
            raise Exception("environment variable {0} not found".format(self.alc_working_dir_env))

        self.inputs = self.input_map[Keys.inputs_key_name]
        self.attributes = self.input_map[Keys.attributes_key_name]
        self.lec2_model, lec2_metadata = self.get_input_data_dirs(self.input_lec2_model_key_name)
        if len(self.lec2_model):
            self.lec2_model = str(self.lec2_model[0])
        
        self.lec3_model, lec3_metadata = self.get_input_data_dirs(self.input_lec3_model_key_name)
        if len(self.lec3_model):
            self.lec3_model = str(self.lec3_model[0])
        

        self.lecdd_model, lecdd_metadata = self.get_input_data_dirs(self.input_lecdd_model_key_name)
        if len(self.lecdd_model):
            self.lecdd_model = str(self.lecdd_model[0])

        self.parameters     = self.get_parameters()
        self.post_processing_script = self.get_post_processing_script()

    def execute(self):
        from alc_utils import execution_runner
        exec_config_file_path = self.create_execution_files()
        
        runner = execution_runner.ExecutionRunner(str(exec_config_file_path))
        
        result, resultdir = runner.run()
        
        if result == 0:
            from alc_utils.file_uploader import FileUploader
            file_loader = FileUploader()
            params = self.sim_parameters#[self.launch_execute_key][self.launch_param_key]
            log_results = file_loader.upload_with_params(str(self.exec_folder_path), params)
            log_results[self.result_url_key] = str(self.create_result_notebook())
            log_results[self.result_expt_params_key] = params
            return log_results

        return result

    # method invoked to run the jobs
    def run(self):
        self.setup()
        
        if not self.check() and self.error_mesgs:
            print('check messages {0}'.format(self.error_mesgs))
            ret = {
                self.result_error_key: self.error_mesgs
            }
            return ret
        
        return self.execute()
