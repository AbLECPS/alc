"""
This is where the implementation of the plugin code goes.
The LaunchActivity-class is imported from both run_plugin.py and run_debug.py
"""
import sys
import traceback
import logging
import json
import re
import time
import os
from pathlib import Path
from webgme_bindings import PluginBase
from urllib.parse import urlunsplit, urljoin
import urllib.request
sys.path.append(str(Path(__file__).absolute().parent))
from . import SlurmSetup
from KeysAndAttributes import Keys, Attributes, References
from ros_gen import SystemLaunchGen
from future.utils import iteritems
from alc_utils.slurm_executor import WebGMEKeys, SlurmParameterKeys
from alc_utils.setup_repo import RepoSetup
import socket

# Setup a logger
logger = logging.getLogger('LaunchActivity')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)  # By default it logs to stderr..
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

url_hostname = "localhost"

input_meta_type_name        = "ALC_EP_Meta.Input"
context_meta_type_name      = "ALC_EP_Meta.Context"
content_meta_type_name      = "ALC_EP_Meta.content"
assembly_meta_type_name     = "ALC_EP_Meta.AssemblyModel"
system_meta_type_name       = "ALC_EP_Meta.SystemModel"
lec_meta_type_name          = "ALC_EP_Meta.LEC_Model"
block_meta_type_name        = "ALC_EP_Meta.Block"
parameter_meta_type_name    = "ALC_EP_Meta.Parameter"
params_table_meta_type_name = "ALC_EP_Meta.ParamsTable"
params_meta_type_name       = "ALC_EP_Meta.Params"


number_re = re.compile(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]\d+)?')

var_re = re.compile(r'(?:\$\$)*(\$\w+|\${\w+})')

alc_working_dir_env_var_name = "ALC_WORKING_DIR"
jupyter_dir_name = "jupyter"

root_node_name = "ROOT"

node_path_key_name = "nodePath"

config_deployment_folder_key_name = 'gen_folder'
config_deploy_job_key_name = 'deploy_job'
alc_meta_type_name  = "ALC"
exec_dir_name = ".exec"

setup_folder_attribute_name = "Setup_Folder"

launch_activity_campaign_attribute_name = "Campaign"

alc_repo_name = "alc_core"

class LaunchActivity(PluginBase):

    project_name_key = "name"
    project_owner_key = "owner"

    def __init__(
            self,
            webgme, commit_hash, BRANCH_NAME, ACTIVE_NODE_PATH, ACTIVE_SELECTION_PATHS, NAMESPACE,
            config=None, **kwargs
    ):
        PluginBase.__init__(self, webgme, commit_hash, BRANCH_NAME, ACTIVE_NODE_PATH, ACTIVE_SELECTION_PATHS, NAMESPACE)

        webgme_port = kwargs.pop(WebGMEKeys.webgme_port_key, 8888)
        self.slurm_params = {
            WebGMEKeys.webgme_port_key: webgme_port
        }

        self.timeout_param = 0
        self.metadata_url = urlunsplit(
            ['http', "{0}:{1}".format(url_hostname, webgme_port), "/rest/blob/metadata/", None, None]
        )
        self.download_url = urlunsplit(
            ['http', "{0}:{1}".format(url_hostname, webgme_port), "/rest/blob/download/", None, None]
        )

        project_info = self.project.get_project_info()
        self.project_owner = project_info.get(Keys.owner_key_name)
        self.project_name = project_info.get(Keys.name_key_name)

        self.temp_dir = None
        self.time_stamp = None
        if not config:
            self.config = self.get_current_config()
        else:
            self.config = config

        if SlurmParameterKeys.job_name_key not in self.config or self.config[SlurmParameterKeys.job_name_key] is None \
                or self.config[SlurmParameterKeys.job_name_key] == '':
            self.config[SlurmParameterKeys.job_name_key] = 'launch_activity'
        
        self.alc_node = None
        self.repo = ''
        self.branch = ''
        self.tag = ''
        self.repo_root = ''
        self.repo_home = ''
        self.launchedByWF = False
        self.setup_workflow_campaign = False
        self.camp_definition = None
        self.ALC_SSH_PORT='22'
        self.ALC_SSH_HOST=socket.gethostbyname(socket.gethostname())


    def create_temp_dir(self, node_name):

        # FIXME: Creation of a folder inside the docker container "/tmp" is not allowed for this exercise.
        # This is because the folder and the json file and the downloaded files are needed outside of the
        # webgme docker to run the activity
        alc_working_dir_name = os.environ.get(alc_working_dir_env_var_name, None)
        if alc_working_dir_name:
            project_dir_name = "{0}_{1}".format(self.project_owner, self.project_name)
            activity_name = node_name
            base_temp_dir = Path(alc_working_dir_name, jupyter_dir_name, project_dir_name, activity_name)
            self.time_stamp = str(int(time.time()))
            temp_dir = Path(base_temp_dir, self.time_stamp)
            while temp_dir.exists():
                time.sleep(1)
                self.time_stamp = str(int(time.time()))
                temp_dir = Path(base_temp_dir, self.time_stamp)

            self.temp_dir = temp_dir

            if not self.temp_dir.exists():
                self.temp_dir.mkdir(parents=True)

            logger.info('temp dir {0}'.format(str(self.temp_dir)))
            if self.temp_dir.exists():
                logger.info('created temp dir {0}'.format(str(self.temp_dir)))
            else:
                logger.info('did not create temp dir {0}'.format(str(self.temp_dir)))

        else:
            raise Exception("Environment Variable: \"{0}\" is not found.".format(alc_working_dir_env_var_name))

    def get_child_nodes(self, parent_node):

        child_node_list = self.core.load_children(parent_node)
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
    
    def get_descendant_nodes(self, parent_node, meta_type_name):

        child_node_list = self.core.load_sub_tree(parent_node)
        node_list = []

        for child_node in child_node_list:
            if not child_node:
                continue

            child_node_meta_type = self.core.get_meta_type(child_node)
            child_node_meta_type_name = self.core.get_fully_qualified_name(child_node_meta_type)
            if child_node_meta_type_name != meta_type_name:
                continue
            node_list.append(child_node)

        return node_list

    def execute_constraint(self, node, parameter_name, value):
        constraint = self.core.get_attribute(node, Attributes.constraint_attribute_name)
        if bool(constraint):
            local_map = {}
            exec(constraint, globals(), local_map)
            function_name = list(local_map.keys())[0]
            constraint_satisfied = local_map[function_name](value)
            if not constraint_satisfied:
                raise Exception("Value \"{0}\" of parameter \"{1}\" did not meet constraint \"{2}\"".format(
                    value, parameter_name, constraint
                ))

    @staticmethod
    def value_specified(value):
        if value == '':
            return False
        return True

    def get_parameters(self, parent_node, node_name, current_choice):

        child_node_map = self.get_child_nodes(parent_node)
        parameter_map = self.get_parameters_from_list(
            child_node_map.get(params_table_meta_type_name, {}), node_name, current_choice
        )

        for parameter in child_node_map.get(parameter_meta_type_name, []):

            choice_set = self.core.get_attribute(parameter, Attributes.choice_list_attribute_name)
            choice_set = set(map(lambda x: x.strip(), choice_set.split("\n"))) if bool(choice_set) else set()

            if not bool(choice_set) or current_choice.intersection(choice_set) != set():
                name = self.core.get_attribute(parameter, Attributes.name_attribute_name)
                full_name = "{0}.{1}".format(node_name, name)

                value = self.core.get_attribute(parameter, Attributes.value_attribute_name)
                type_val = self.core.get_attribute(parameter, Attributes.type_attribute_name)

                if type_val != Keys.asset_type_name and not self.value_specified(value):
                    value = self.core.get_attribute(parameter, Attributes.default_value_attribute_name)

                if self.value_specified(value):
                    if type_val == Keys.boolean_type_name:
                        value = (value.upper() in ["TRUE", "1"])

                    if type_val in [Keys.array_type_name, Keys.dict_type_name]:
                        logger.info(' type_val {0} value {1}'.format(type_val, value))
                        value = eval(value)

                    if (type_val in [Keys.float_type_name, Keys.integer_type_name]) and (number_re.fullmatch(value)):
                        if type_val == Keys.float_type_name:
                            value = float(value)
                        elif type_val == Keys.integer_type_name:
                            value = int(value)

                        min_value = self.core.get_attribute(parameter, Attributes.min_attribute_name)
                        if bool(min_value):
                            min_value = float(min_value)
                            if value < min_value:
                                raise Exception("Value for parameter \"{0}\" is less than minimum of {1}".format(
                                    full_name, min_value
                                ))

                        max_value = self.core.get_attribute(parameter, Attributes.max_attribute_name)
                        if bool(max_value) and max_value != "-1":
                            max_value = float(max_value)
                            if value > max_value:
                                raise Exception("Value for parameter \"{0}\" is greater than maximum of {1}".format(
                                    full_name, max_value
                                ))

                        self.execute_constraint(parameter, full_name, value)

                elif type_val == Keys.asset_type_name:
                    value = str(self.get_asset_attribute(parameter))

                if self.value_specified(value):
                    parameter_map[name] = value
                elif self.core.get_attribute(parameter, Attributes.required_attribute_name):
                    parameter_map[name] = ""
                    logger.info('required parameter {0} not specified '.format(name))

                    # raise Exception("Value for parameter \"{0}\" must be specified".format(full_name))

        return parameter_map

    def get_parameters_from_list(self, parameter_table_list, parameter_name, current_choice):

        parameter_map = {}

        for parameter_table in parameter_table_list:

            choice_set = self.core.get_attribute(parameter_table, Attributes.choice_list_attribute_name)
            choice_set = set(map(lambda x: x.strip(), choice_set.split("\n"))) if bool(choice_set) else set()

            #if not bool(choice_set) or current_choice in choice_set:
            if not bool(choice_set) or current_choice.intersection(choice_set) != set():
                name = self.core.get_attribute(parameter_table, Attributes.name_attribute_name)
                value = self.get_parameters(parameter_table, ".".join([parameter_name, name]), current_choice)
                if bool(value):
                    parameter_map[name] = value

        return parameter_map

    def get_contents(self, parent_node, current_choice):

        child_node_map = self.get_child_nodes(parent_node)
        content_map = {}

        for content in child_node_map.get(content_meta_type_name, []):

            choice_set = self.core.get_attribute(content, Attributes.choice_list_attribute_name)
            choice_set = set(map(lambda x: x.strip(), choice_set.split("\n"))) if bool(choice_set) else set()

            #if not bool(choice_set) or current_choice in choice_set:
            if not bool(choice_set) or current_choice.intersection(choice_set) != set():
                name = self.core.get_attribute(content, Attributes.name_attribute_name)

                definition = self.core.get_attribute(content, Attributes.definition_attribute_name)
                filename = self.core.get_attribute(content, Attributes.filename_attribute_name)

                # if bool(definition):
                content_map[name] = {
                    Keys.node_path_key_name: self.core.get_path(content),
                    Keys.node_named_path_key_name: str(self.get_model_node_named_path(content)),
                    Attributes.definition_attribute_name: definition,
                    Attributes.filename_attribute_name: filename
                }

        return content_map

    def get_assembly_parameter_lec_info(self, assembly_node):
        child_node_map = self.get_child_nodes(assembly_node)
        parameter_map = {}
        lecinfo_map = {}

        for system in child_node_map.get(system_meta_type_name, []):
            block_child_node_list = self.get_descendant_nodes(system, block_meta_type_name)
            for block in block_child_node_list:
                isimpl = self.core.get_attribute(block, Attributes.implementation_attribute_name)
                if not isimpl:
                    continue
                isactive = self.core.get_attribute(block, Attributes.isactive_attribute_name)
                if not isactive:
                    continue
                block_child_node_map = self.get_child_nodes(block)
                for param in block_child_node_map.get(params_meta_type_name, []):
                    try:
                        param_definition = self.core.get_attribute(param, Attributes.params_definition_attribute_name)
                        jsonval = json.loads(param_definition, strict=False)
                        for k in jsonval.keys():
                            parameter_map[k] = jsonval[k]
                    except Exception as err:
                        msg = str(err)
                        logger.info("system parameter exception {0}".format(msg))
                        pass
                for lec in block_child_node_map.get(lec_meta_type_name, []):
                    try:
                        depkey = self.core.get_attribute(lec, Attributes.deployment_key_attribute_name)
                        if not depkey:
                            continue
                        lecdata = self.core.get_pointer_path(lec, References.lec_reference_name)
                        if not lecdata:
                            continue
                        lecnode = self.core.load_by_path(self.root_node, lecdata)
                        if not lecnode:
                            continue
                        datainfostr = self.core.get_attribute(lecnode, Attributes.datainfo_attribute_name)
                        if datainfostr:
                            try:
                                jsonval = json.loads(datainfostr, strict=False)
                                lecinfo_map[depkey] = jsonval
                            except Exception as err:
                                msg = str(err)
                                logger.info("lec datainfo exception {0}".format(msg))
                                pass
                    except Exception as err:
                        msg = str(err)
                        logger.info("exception {0}".format(msg))
                        pass
        return parameter_map, lecinfo_map

    def get_assembly_info(self, assembly_node):

        assembly_map = {
            Keys.node_path_key_name: self.core.get_path(assembly_node),
            Keys.node_named_path_key_name: str(self.get_model_node_named_path(assembly_node))
        }

        try:
            assembly_map[Keys.container_key_name] = {}
            assembly_map[Keys.launchfile_key_name] = {}
            ros_launch_gen = SystemLaunchGen(self)
            logger.info("************************1")
            artifact_content, container_info = ros_launch_gen.gen_launch_file(assembly_node)
            logger.info("************************2")
            launch_file_info = {}
            for file_name, file_content in iteritems(artifact_content):
                launch_file_info["launch_files/%s" % file_name] = file_content
            assembly_map[Keys.container_key_name] = container_info
            assembly_map[Keys.launchfile_key_name] = launch_file_info
        except Exception as err:
            msg = str(err)
            logger.info("get assembly info exception {0}".format(msg))
            pass
        
        try:

            logger.info("************************3")

            parameter_info, lec_info = self.get_assembly_parameter_lec_info(assembly_node)

            logger.info("************************4")

            assembly_map[Keys.parameters_key_name]  = parameter_info
            assembly_map[Keys.lecinfo_key_name]  = lec_info

        except Exception as err:
            msg = str(err)
            logger.info("get assembly info exception {0}".format(msg))
            pass

        return assembly_map

    def get_assembly(self, parent_node):
        child_node_map = self.get_child_nodes(parent_node)
        assembly_map = {}
        for assembly in child_node_map.get(assembly_meta_type_name, []):
            name = self.core.get_attribute(assembly, Attributes.name_attribute_name)
            assembly_map[name] = self.get_assembly_info(assembly)
        return assembly_map

    def get_asset_attribute(self, node):
        local_hash = self.core.get_attribute(node, Attributes.asset_attribute_name)
        if not local_hash:
            logger.info('***************no asset***************')
            return None
        
        url = urljoin(self.metadata_url, local_hash)
        request = urllib.request.urlopen(url)
        data = json.loads(request.read())

        file_name = data.get(Keys.name_key_name)

        temp_file_path = Path(self.temp_dir, file_name)
        lec_file_url = urljoin(urljoin(self.download_url, local_hash + "/"), file_name)
        urllib.request.urlretrieve(lec_file_url, str(temp_file_path))

        return str(temp_file_path)

    def get_input_data_set(self, node):
        data = []
        setmembers = self.core.get_member_paths(node, Keys.input_set_name)
        for s in setmembers:
            refnode = self.core.load_by_path(self.root_node, s)
            content = self.core.get_attribute(refnode, Attributes.datainfo_attribute_name)
            if content:
                try:
                    jsonval = json.loads(content, strict=False)
                    data.append(jsonval)
                except:
                    pass
        return data

    def get_inputs(self, inputs_list, current_choice):
        inputs_map = {}
        for input_element in inputs_list:

            choice_set = self.core.get_attribute(input_element, Attributes.choice_list_attribute_name)
            choice_set = set(map(lambda x: x.strip(), choice_set.split("\n"))) if bool(choice_set) else set()

            #if bool(choice_set) and current_choice not in choice_set:
            if bool(choice_set) and current_choice.intersection(choice_set) == set():
                continue

            key_name = Attributes.value_attribute_name
            value = self.core.get_attribute(input_element, Attributes.value_attribute_name)
            if not bool(value):
                key_name = Attributes.asset_attribute_name
                value = self.get_asset_attribute(input_element)
                if not bool(value):
                    key_name = Keys.input_set_name
                    value = self.get_input_data_set(input_element)

            name = self.core.get_attribute(input_element, Attributes.name_attribute_name)
            value_map = {
                Keys.node_path_key_name: self.core.get_path(input_element),
                Keys.node_named_path_key_name: str(self.get_model_node_named_path(input_element))
            }

            if bool(value):
                value_map[key_name] = value

            parameter_map = self.get_parameters(input_element, name, current_choice)
            if bool(parameter_map):
                value_map[Keys.parameters_key_name] = parameter_map

            inputs_map[name] = value_map

        return inputs_map
    
    def get_contexts(self, contexts_list, current_choice):
        contexts_map = {}
        for context_element in contexts_list:

            choice_set = self.core.get_attribute(context_element, Attributes.choice_list_attribute_name)
            choice_set = set(map(lambda x: x.strip(), choice_set.split("\n"))) if bool(choice_set) else set()

            #if bool(choice_set) and current_choice not in choice_set:
            if bool(choice_set) and current_choice.intersection(choice_set) == set():
                continue

            name = self.core.get_attribute(context_element, Attributes.name_attribute_name)
            value_map = {
                Keys.node_path_key_name: self.core.get_path(context_element),
                Keys.node_named_path_key_name: str(self.get_model_node_named_path(context_element))
            }

            content_map = self.get_contents(context_element, current_choice)
            if bool(content_map):
                value_map[Keys.content_key_name] = content_map

            assembly_map = self.get_assembly(context_element)
            if bool(assembly_map):
                value_map[Keys.assembly_key_name] = assembly_map

            parameter_map = self.get_parameters(context_element, name, current_choice)
            if bool(parameter_map):
                value_map[Keys.parameters_key_name] = parameter_map

            contexts_map[name] = value_map

        return contexts_map

    def get_model_node_named_path(self, node, retval=Path()):
        if node is None:
            return retval

        node_name = self.core.get_fully_qualified_name(node)
        if node_name == root_node_name:
            return Path('/', retval)

        return self.get_model_node_named_path(
            self.core.get_parent(node), Path(node_name, retval)
        )
    
    @staticmethod
    def build_param_map(param_table, parent_list, ret):
        for param in param_table:
            if param not in ret:
                ret[param] = parent_list[:]
                ret[param].append(param)

    def build_parameter_map(self, parameter_map):
        ret = {}
        for param_table in parameter_map.keys():
            self.build_param_map(parameter_map[param_table], [param_table], ret)
        return ret

    def setup_and_build_repo (self, setup_and_build_repo):
        print('in setup and build repo')
        child_node_list = self.core.load_children(self.root_node)
        for child_node in child_node_list:
            if not child_node:
                continue

            child_node_meta_type = self.core.get_meta_type(child_node)
            child_node_meta_type_name = self.core.get_fully_qualified_name(child_node_meta_type)
            if child_node_meta_type_name.endswith(alc_meta_type_name):
                print("came here")
                self.alc_node = child_node
                self.repo = self.core.get_attribute(self.alc_node,'repo')
                self.branch = self.core.get_attribute(self.alc_node,'branch')
                self.tag = self.core.get_attribute(self.alc_node,'tag')
                break
        print('self.repo '+self.repo)

        
        alc_working_dir_name = os.environ.get(alc_working_dir_env_var_name, None)
        if alc_working_dir_name: 
            if self.repo:
                self.repo_root = Path(alc_working_dir_name, exec_dir_name, self.repo)
                self.repo_home = os.path.join(str(self.repo_root), self.repo)
                print('self.repo_home ' +self.repo_home)
                if (setup_and_build_repo or not self.repo_root.exists()):
                    if not self.repo_root.exists():
                        self.repo_root.mkdir(parents=True)
                    r =  RepoSetup()
                    r.clone_repo(self.repo_root, self.repo, self.branch, self.tag, logger)
                    r.build_repo(self.repo_root, self.repo, self.branch, self.tag, logger)
        else:
            raise Exception("Environment Variable: \"{0}\" is not found.".format(alc_working_dir_env_var_name))
        
    def update_setupfolder(self, setup_folder):
        alc_working_dir_name = os.environ.get(alc_working_dir_env_var_name, None)
        if alc_working_dir_name:
            if (setup_folder.startswith('$ALC_REPO_HOME')):
                repo_root = Path(alc_working_dir_name, exec_dir_name, alc_repo_name,alc_repo_name)
                ret = setup_folder.replace('$ALC_REPO_HOME',str(repo_root))
                return ret
            if (setup_folder.startswith('$REPO_HOME')):
                print('replacing repo home')
                ret = setup_folder.replace('$REPO_HOME',str(self.repo_home))
                print('self.repo_home ' +self.repo_home)
                print('replacing repo home ret = '+ret)
                return ret
        else:
            raise Exception("Environment Variable: \"{0}\" is not found.".format(alc_working_dir_env_var_name))
        
        return setup_folder

        
        


    def main(self):
        try:
            core = self.core
            active_node = self.active_node
            output_map = {
                Keys.attributes_key_name: {},
                Keys.inputs_key_name: {},
                Keys.outputs_key_name: {},
                Keys.parameters_key_name: {}
            }

            name = core.get_attribute(active_node, Attributes.name_attribute_name)

            self.setup_and_build_repo(self.config.get('setupAndBuildRepo',False))

            setup_folder = self.core.get_attribute(self.active_node,setup_folder_attribute_name)
            setup_folder = self.update_setupfolder(setup_folder)
            print('setup folder ='+str(setup_folder))

            if (not setup_folder):
                raise Exception("It appears that the activity has not been initialized. Please initialize the activity by running the ActivityInit Plugin.")
            if (not os.path.exists(setup_folder)):
                raise Exception("It appears that the activity-definition has not been setup. Activity-Definition setup folder does not exist. Please check.")


            gen_folder = self.config.get(config_deployment_folder_key_name, None)
            if not gen_folder:
                self.create_temp_dir(name)
            else:
                self.temp_dir = Path(gen_folder, name)
                if not self.temp_dir.exists():
                    self.temp_dir.mkdir(parents=True)
            
            deploy_job = self.config.get(config_deploy_job_key_name, True)

            if (not deploy_job and gen_folder):
                self.launchedByWF = True
            

            current_choice_input = core.get_attribute(active_node, Attributes.current_choice_attribute_name)
            current_choice = set(map(lambda x: x.strip(), current_choice_input.split(","))) if bool(current_choice_input) else set()


            # FIXME:
            # It is possible that the current choice can be  empty/ none.
            # This is acceptable when the choice list is empty.

            attributes_map = {
                Keys.project_name_key_name: self.project_name,
                Keys.owner_key_name: self.project_owner,
                Keys.current_choice_key_name: current_choice_input,
                Keys.temp_dir_key_name: str(self.temp_dir),
                Keys.node_path_key_name: self.core.get_path(active_node),
                Keys.node_named_path_key_name: str(self.get_model_node_named_path(active_node))
            }
            
            output_map[Keys.attributes_key_name] = attributes_map

            logger.info('ActiveNode at "{0}" has name {1}'.format(core.get_path(active_node), name))

            child_node_map = self.get_child_nodes(active_node)

            inputs_map = self.get_inputs(child_node_map.get(input_meta_type_name, []), current_choice)
            if bool(inputs_map):
                output_map[Keys.inputs_key_name] = inputs_map

            contexts_map = self.get_contexts(child_node_map.get(context_meta_type_name, []), current_choice)
            if bool(contexts_map):
                output_map[Keys.context_key_name] = contexts_map

            parameters_map = self.get_parameters_from_list(
                child_node_map.get(params_table_meta_type_name, []), "", current_choice
            )
            if bool(parameters_map):
                output_map[Keys.parameters_key_name] = parameters_map
                output_map[Keys.parameter_map_key_name] = self.build_parameter_map(parameters_map)

            
            
            base_node = self.core.get_base(active_node)
            job_type = self.core.get_attribute(active_node, Attributes.label_attribute_name)
            #job_type = self.core.get_attribute(base_node, Attributes.name_attribute_name)

            

            if (not self.launchedByWF):
                campaign_inputs = self.core.get_attribute(active_node, launch_activity_campaign_attribute_name)
                if (campaign_inputs):
                    self.setup_workflow_campaign = False
                    try:
                        self.camp_definition = json.loads(campaign_inputs, strict=False)
                        camp_keys = list(self.camp_definition.keys())
                        if (len(camp_keys)):
                            self.setup_workflow_campaign = True
                    except Exception as err:
                        self.camp_definition = None
                        msg = str(err)
                        logger.info("issues with parsing campaign definition-{0}".format(msg))
                        pass
            
            json_dump_folder = self.temp_dir

            if (self.setup_workflow_campaign):
                deploy_job = False
                logger.info("campaign definition-{0}".format(str(self.camp_definition)))
                json_dump_folder = Path(str(self.temp_dir), 'Prototype', name)
                if not json_dump_folder.exists():
                    json_dump_folder.mkdir(parents=True)

            
            with Path(json_dump_folder, "launch_activity_output.json").open("w") as json_fp:
                json.dump(output_map, json_fp, indent=4, sort_keys=True)

            SlurmSetup.setup_job(
                self,
                name,
                self.time_stamp,
                self.timeout_param,
                job_type,
                setup_folder,
                deploy_job,
                str(self.temp_dir),
                self.setup_workflow_campaign,
                self.camp_definition
            )
        except Exception as err:
            msg = str(err)
            logger.info("exception {0}".format(msg))
            traceback_msg = traceback.format_exc()
            logger.info(traceback_msg)
            sys_exec_info_msg = sys.exc_info()[2]
            logger.info(sys_exec_info_msg)
            self.create_message(self.active_node, msg, 'error')
            self.create_message(self.active_node, traceback_msg, 'error')
            self.create_message(self.active_node, str(sys_exec_info_msg), 'error')
            self.result_set_error('LaunchActivity Plugin: Error encountered.  Check result details.')
            self.result_set_success(False)
            exit()
