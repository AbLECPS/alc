#!/usr/bin/env /usr/bin/python3.6
"""
This is where the implementation of the plugin code goes.
The LaunchVerification-class is imported from both run_plugin.py and run_debug.py
"""
import json
import time
import sys
import traceback
import logging
from pathlib import Path
from webgme_bindings import PluginBase
from alc_utils.slurm_executor import Keys

sys.path.append(str(Path(__file__).absolute().parent))
import RobustnessKeys
import SlurmSetup


def get_logger(clazz):
    logger_name = clazz.__module__ + "." + clazz.__name__
    logger = logging.Logger(logger_name)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger


class LaunchVerification(PluginBase):

    logger = None

    root_node_name = "ROOT"

    valid_meta_type_name_set = {
        "ALCMeta.VerificationSetup",
        "ALCMeta.ValidationSetup",
        "ALCMeta.SystemIDSetup"
    }

    parameter_name_set = {
        RobustnessKeys.attack_parameter_key,
        RobustnessKeys.method_parameter_key
    }

    alcmeta_eval_data_meta_name = "ALCMeta.EvalData"
    alcmeta_lec_model_meta_name = "ALCMeta.LEC_Model"
    alcmeta_verification_model_meta_name = "ALCMeta.Verification_Model"
    alcmeta_parameter_meta_name = "ALCMeta.parameter"

    alcmeta_eval_data_set_name = "Data"
    alcmeta_lec_model_dataset_name = "Dataset"
    alcmeta_set_member_attribute_name = "datainfo"
    data_info_map_directory_key = "directory"
    alcmeta_lec_model_pointer_name = "ModelDataLink"

    alcmeta_result_name = "ALCMeta.Result"
    alcmeta_result_data_alt_name = "DEEPFORGE.pipeline.Data"
    alcmeta_result_data_name = "{0}.{1}".format("ALCMeta", alcmeta_result_data_alt_name)

    network_directory_name = "Network"

    def __init__(self, *args, **kwargs):
        self.slurm_params = {
            Keys.webgme_port_key: kwargs.pop(Keys.webgme_port_key, 8888)
        }
        super(LaunchVerification, self).__init__(*args, **kwargs)
        self.test = False
        if "test" in kwargs:
            self.test = kwargs.get("test")

        self.config = self.get_current_config()

        self.setupJupyterNB = self.config['setupJupyterNB']

        self.active_node_meta_type = None
        self.active_node_meta_type_name = None

        self.timeout_param = 0

    def check_active_node_meta(self):
        return self.active_node_meta_type_name in LaunchVerification.valid_meta_type_name_set

    def get_child_nodes(self, parent_node):

        child_node_list = self.core.load_sub_tree(parent_node)
        child_node_map = {}
        for child_node in child_node_list:
            if not child_node:
                continue

            child_node_meta_type = self.core.get_meta_type(child_node)
            child_node_meta_type_name = self.core.get_fully_qualified_name(child_node_meta_type)
            child_node_list = child_node_map.get(child_node_meta_type_name, [])
            child_node_list.append(child_node)
            child_node_map[child_node_meta_type_name] = child_node_list

        return child_node_map

    def get_model_node_path(self, node, retval=Path()):
        if node is None:
            return retval

        node_name = self.core.get_fully_qualified_name(node)
        if node_name == LaunchVerification.root_node_name:
            return Path('/', retval)

        return self.get_model_node_path(
            self.core.get_parent(node), Path(node_name, retval)
        )
    
    def create_result_node(self, result_node_list, time_val, exec_name, result_dir):
        if not result_node_list or len(result_node_list) == 0:
            raise RuntimeError("Model needs to have one result node")

        result_node = result_node_list[0]
        meta = self.META.get(LaunchVerification.alcmeta_result_data_name, None)
        if meta is None:
            meta = self.META.get(LaunchVerification.alcmeta_result_data_alt_name)
        result_data_node = self.core.create_child(result_node, meta)
        node_name = 'result-'+exec_name
        self.core.set_attribute(result_data_node, 'name', node_name)
        self.core.set_attribute(result_data_node, 'createdAt', time_val*1000)
        self.core.set_attribute(result_data_node, 'resultDir', str(result_dir))
        return self.core.get_path(result_data_node)

    def main(self):
        try:
            #
            # template_parameter_map CONTAINS MAPPINGS TO FILL OUT JUPYTER NOTEBOOK
            #
            template_parameter_map = {}

            #
            # TEMPLATE PARAMETERS:  GET PROJECT NAME AND OWNER
            #
            project_info = self.project.get_project_info()
            project_name = project_info.get(Keys.project_name_key)
            project_owner = project_info.get(Keys.project_owner_key)

            template_parameter_map[RobustnessKeys.template_project_name_key] = project_name
            template_parameter_map[RobustnessKeys.template_owner_name_key] = project_owner

            #
            # VERIFICATION_SETUP NODE SHOULD BE ACTIVE NODE
            #
            self.active_node_meta_type = self.core.get_meta_type(self.active_node)
            self.active_node_meta_type_name = self.core.get_fully_qualified_name(self.active_node_meta_type)

            if not self.check_active_node_meta():
                raise RuntimeError("Model needs to be one of {0}".format(LaunchVerification.valid_meta_type_name_set))

            #
            # TEMPLATE PARAMETERS: VERIFICATION NODE PATH AND ID
            #

            template_parameter_map[RobustnessKeys.template_verification_node_path_key] = \
                str(self.get_model_node_path(self.active_node))
            template_parameter_map[RobustnessKeys.template_verification_node_id_key] = \
                self.core.get_path(self.active_node)

            #
            #  GET CHILD VERIFICATION_MODEL NODE OF VERIFICATION_SETUP NODE
            #
            verification_setup_child_node_map = self.get_child_nodes(self.active_node)
            verification_model_node_list = verification_setup_child_node_map.get(
                LaunchVerification.alcmeta_verification_model_meta_name, []
            )
            
            if len(verification_model_node_list) == 0:
                raise RuntimeError(
                    "No object found of meta-type \"{0}\" in \"{1}\" meta-type object.".format(
                        LaunchVerification.alcmeta_verification_model_meta_name,
                        self.alcmeta_verification_model_meta_name
                    )
                )

            if len(verification_model_node_list) > 1:
                LaunchVerification.logger.warning(
                    "More than one object of meta-type \"{0}\" found in \"{1}\" meta-type object.  "
                    "Using the first one.".format(
                        LaunchVerification.alcmeta_verification_model_meta_name,
                        self.alcmeta_verification_model_meta_name
                    )
                )

            verification_model_node = verification_model_node_list[0]

            #
            # GET CHILD PARAMETER NODES OF VERIFICATION_MODEL NODE
            #
            verification_model_child_node_map = self.get_child_nodes(verification_model_node)

            parameter_node_list = verification_model_child_node_map.get(
                LaunchVerification.alcmeta_parameter_meta_name, []
            )
            if len(parameter_node_list) == 0:
                raise RuntimeError(
                    "\"{0}\" meta-type object must have child \"{1}\" meta-type object.".format(
                        LaunchVerification.alcmeta_parameter_meta_name,
                        LaunchVerification.alcmeta_verification_model_meta_name
                    )
                )

            # READ PARAMETERS INTO DICT
            parameter_map = {}
            for parameter_node in parameter_node_list:
                parameter_name = self.core.get_attribute(parameter_node, "name")
                parameter_value = self.core.get_attribute(parameter_node, "value")
                # READ SLURM parameters
                if parameter_name.lower() == "slurm":
                    self.slurm_params.update(parameter_value)
                    continue
                # READ timout parameter
                if parameter_name.lower() == "timeout":
                    self.timeout_param = float(parameter_value)
                    continue
                
                parameter_map[parameter_name.lower()] = \
                    parameter_value.lower() if isinstance(parameter_value, str) else parameter_value

            # MAKE SURE ATTACK-PARAMETER SPECIFIED
            if RobustnessKeys.attack_parameter_key not in parameter_map:
                raise RuntimeError("\"{0}\" object must have \"{1}\" parameter.".format(
                    LaunchVerification.alcmeta_verification_model_meta_name,
                    RobustnessKeys.attack_parameter_key
                ))

            attack_type = parameter_map.get(RobustnessKeys.attack_parameter_key)
            if attack_type not in RobustnessKeys.attack_map.keys():
                raise RuntimeError("Invalid attack type \"{0}\":  must be one of \"{1}\".".format(
                    attack_type, RobustnessKeys.attack_map.keys()
                ))

            #
            # GET NAMES OF ALL REQUIRED PARAMETERS
            #
            
            extra_parameter_set = RobustnessKeys.attack_map \
                .get(attack_type) \
                .get(RobustnessKeys.required_parameters_key)

            required_parameter_name_set = LaunchVerification.parameter_name_set.union(
                extra_parameter_set
            )

            #
            # DELETE UNNECESSARY PARAMETERS
            #
            parameter_map_key_set = set(parameter_map.keys())
            unnecessary_parameter_name_set = parameter_map_key_set.difference(required_parameter_name_set)
            for key in unnecessary_parameter_name_set:
                del parameter_map[key]

            #
            # MAKE SURE ALL REQUIRED PARAMETERS ARE PRESENT
            #
            parameter_map_key_set = set(parameter_map.keys())
            missing_parameter_set = required_parameter_name_set.difference(parameter_map_key_set)

            if len(missing_parameter_set) != 0:
                raise RuntimeError("Missing parameters \"{0}\"for attack type \"{1}\".".format(
                    missing_parameter_set, attack_type
                ))

            #
            # GET PERCEPTION LEC MODEL DIRECTORY
            #

            # GET LEC MODEL NODE
            lec_model_node_list = verification_model_child_node_map.get(LaunchVerification.alcmeta_lec_model_meta_name)
            if len(lec_model_node_list) == 0:
                raise RuntimeError("At least one \"{0}\" object must be in \"{1}\" model".format(
                    LaunchVerification.alcmeta_lec_model_meta_name,
                    LaunchVerification.alcmeta_verification_model_meta_name
                ))
            if len(lec_model_node_list) > 1:
                LaunchVerification.logger.warning(
                    "More than one \"{0}\" object found in \"{1}\" model.  Using the first".format(
                        LaunchVerification.alcmeta_lec_model_meta_name,
                        LaunchVerification.alcmeta_verification_model_meta_name
                    )
                )

            lec_model_node = lec_model_node_list[0]

            #
            # TEMPLATE PARAMETERS:  LEC MODEL NODE PATH AND ID
            #
            template_parameter_map[RobustnessKeys.template_lec_node_reference_path_key] = \
                str(self.get_model_node_path(lec_model_node))
            template_parameter_map[RobustnessKeys.template_lec_node_reference_id_key] = \
                self.core.get_path(lec_model_node)

            #
            # GET NODE OF LEC POINTED TO BY LED MODEL NODE
            #
            lec_path = self.core.get_pointer_path(lec_model_node, LaunchVerification.alcmeta_lec_model_pointer_name)
            lec_node = self.core.load_by_path(self.root_node, lec_path)

            #
            # TEMPLATE PARAMETERS:  LEC NODE PATH AND ID
            #
            template_parameter_map[RobustnessKeys.template_lec_node_id_key] = lec_path
            template_parameter_map[RobustnessKeys.template_lec_node_path_key] = str(self.get_model_node_path(lec_node))

            # GET LEC MODEL DIRECTORY
            lec_info_map_str = self.core.get_attribute(
                lec_node, LaunchVerification.alcmeta_set_member_attribute_name
            )
            lec_info_map = json.loads(lec_info_map_str)
            lec_directory = lec_info_map.get(LaunchVerification.data_info_map_directory_key)
            lec_directory_path = Path(lec_directory)

            # GET THE LEC FILE
            network_directory_path = Path(lec_directory_path, LaunchVerification.network_directory_name)
            mat_file_list = sorted(network_directory_path.glob("*.mat"))

            if len(mat_file_list) == 0:
                raise RuntimeError(
                    "lec directory \"{0}\" must contain at least one mat-file"
                    " (that contains a neural network).".format(network_directory_path)
                )
            mat_file = mat_file_list[0].absolute()

            #
            # TEMPLATE PARAMETERS:  LEC DIRECTORY PATH, LEC NETWORK DIRECTORY PATH, LEC MAT-FILE NAME
            #
            template_parameter_map[RobustnessKeys.template_lec_parent_directory_path_key] = \
                str(lec_directory_path)
            template_parameter_map[RobustnessKeys.template_lec_directory_path_key] = str(network_directory_path)
            template_parameter_map[RobustnessKeys.template_lec_file_name_key] = str(mat_file.name)

            if len(mat_file_list) > 1:
                LaunchVerification.logger.warning(
                    "More than 1 mat-file found in \"{0}\" directory.  Using \"{1}\"".format(
                        network_directory_path, mat_file.name
                    )
                )

            #
            # GET TEST-DATA DIRECTORIES
            #

            # GET EVAL_DATA NODE
            if LaunchVerification.alcmeta_eval_data_meta_name not in verification_model_child_node_map:
                raise RuntimeError("\"{0}\" object must be in \"{1}\" model".format(
                    LaunchVerification.alcmeta_eval_data_meta_name,
                    LaunchVerification.alcmeta_verification_model_meta_name
                ))

            eval_data_node_list = verification_model_child_node_map.get(LaunchVerification.alcmeta_eval_data_meta_name)
            if len(eval_data_node_list) == 0:
                raise RuntimeError("At least one \"{0}\" object must be in \"{1}\" model".format(
                    LaunchVerification.alcmeta_eval_data_meta_name,
                    LaunchVerification.alcmeta_verification_model_meta_name
                ))
            if len(eval_data_node_list) > 1:
                LaunchVerification.logger.warning(
                    "More than one \"{0}\" object found in \"{1}\" model.  Using the first".format(
                        LaunchVerification.alcmeta_eval_data_meta_name,
                        LaunchVerification.alcmeta_verification_model_meta_name
                    )
                )

            eval_data_node = eval_data_node_list[0]

            #
            # TEMPLATE PARAMETERS:  EVALDATA NODE PATH AND ID
            #
            template_parameter_map[RobustnessKeys.template_eval_data_node_path_key] = \
                str(self.get_model_node_path(eval_data_node))
            template_parameter_map[RobustnessKeys.template_eval_data_node_id_key] = \
                self.core.get_path(eval_data_node)

            # GET "LaunchVerification.alcmeta_eval_data_set_name" SET MEMBER VALUE
            # -- LIST OF DATACOLLECTION RESULT NODE PATHS
            set_member_list = self.core.get_member_paths(eval_data_node, LaunchVerification.alcmeta_eval_data_set_name)
            if len(set_member_list) == 0:
                raise RuntimeError("\"{0}\" object must contain \"{1}\" set with at least 1 item.".format(
                    LaunchVerification.alcmeta_eval_data_meta_name, LaunchVerification.alcmeta_eval_data_set_name
                ))

            # GET (STRING) PATHS OF DIRECTORIES CONTAINING TEST DATA
            test_data_directory_list = []
            for set_member in set_member_list:
                set_member_node = self.core.load_by_path(self.root_node, set_member)
                data_info_map_str = self.core.get_attribute(
                    set_member_node, LaunchVerification.alcmeta_set_member_attribute_name
                )
                data_info_map = json.loads(data_info_map_str)
                test_data_directory_list.append(
                    data_info_map.get(LaunchVerification.data_info_map_directory_key)
                )

            if len(test_data_directory_list) == 0:
                raise RuntimeError(
                    "\"{0}\" object must contain at least one directory with category-named"
                    " subdirectories and test images of a given category under the corresponding category-named"
                    " directory."
                )

            #
            # TEMPLATE PARAMETERS:  TEST-DATA-DIRECTORY-LIST AND PARAMETER MAP
            #
            template_parameter_map[RobustnessKeys.template_test_data_directory_list_key] = test_data_directory_list
            template_parameter_map[RobustnessKeys.template_parameter_map_key] = parameter_map

            #
            # GET DATASET SCRIPT FOR EXTRACTING TRAINING/TESTING DATA IMAGES, CATEGORY NAMES, CATEGORY VALUES
            #
            lec_dataset_script_text = self.core.get_attribute(
                lec_model_node, LaunchVerification.alcmeta_lec_model_dataset_name
            )
            template_parameter_map[RobustnessKeys.template_dataset_key] = lec_dataset_script_text

            project_jupyter_notebook_directory_path = Path(
                lec_directory_path, RobustnessKeys.notebooks_directory_name
            )
            seconds_since_epoch = int(time.time())
            specific_notebook_directory_path = Path(project_jupyter_notebook_directory_path, str(seconds_since_epoch))
            specific_notebook_directory_path.mkdir(parents=True)

            template_parameter_map[RobustnessKeys.template_specific_notebook_directory_key] = \
                str(specific_notebook_directory_path)

            template_parameter_file = Path(
                specific_notebook_directory_path, RobustnessKeys.template_parameter_file_name
            )
            with template_parameter_file.open("w", encoding="utf-8") as template_parameter_file_fp:
                json.dump(
                    template_parameter_map, template_parameter_file_fp, indent=4, sort_keys=True, ensure_ascii=False
                )

            # CREATE RESULT NODE
            exec_name = "{0}-{1}".format(self.config['name'], seconds_since_epoch)
            result_node_path = self.create_result_node(
                verification_setup_child_node_map.get(LaunchVerification.alcmeta_result_name, None),
                seconds_since_epoch,
                exec_name,
                specific_notebook_directory_path
            )
            #
            # EXECUTE SLURM HERE
            #
            slurm_job_params = SlurmSetup.setup_job(
                self.project.get_project_info(),
                exec_name,
                result_node_path,
                self.slurm_params,
                self.timeout_param,
                specific_notebook_directory_path,
                RobustnessKeys.template_parameter_file_name
            )

            LaunchVerification.logger.info("Deploying job to slurm cluster with parameters: %s" % str(slurm_job_params))

            self.util.save(self.root_node, self.commit_hash, 'master', 'Launch Verification Finished')
            self.result_set_success(True)

        except Exception as err:
            msg = str(err)
            LaunchVerification.logger.info("exception {0}".format(msg))
            traceback_msg = traceback.format_exc()
            LaunchVerification.logger.info(traceback_msg)
            sys_exec_info_msg = sys.exc_info()[2]
            LaunchVerification.logger.info(sys_exec_info_msg)
            self.create_message(self.active_node, msg, 'error')
            self.create_message(self.active_node, traceback_msg, 'error')
            self.create_message(self.active_node, str(sys_exec_info_msg), 'error')
            self.result_set_error('LaunchVerification Plugin: Error encountered.  Check result details.')
            self.result_set_success(False)
            exit()


LaunchVerification.logger = get_logger(LaunchVerification)
