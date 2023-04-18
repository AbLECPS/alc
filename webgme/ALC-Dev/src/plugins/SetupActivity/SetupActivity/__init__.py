"""
This is where the implementation of the plugin code goes.
The SetupActivity-class is imported from both run_plugin.py and run_debug.py
"""
import sys
import traceback
import logging
import json
import re
import time
import os
import stat
import subprocess
from pathlib import Path
from webgme_bindings import PluginBase
from urllib.parse import urlunsplit, urljoin
import urllib.request
sys.path.append(str(Path(__file__).absolute().parent))
from future.utils import iteritems
from KeysAndAttributes import Keys, Attributes, References
from alc_utils.setup_repo import RepoSetup

# Setup a logger
logger = logging.getLogger('SetupActivity')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)  # By default it logs to stderr..
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

alc_port = 8888
url_hostname = "localhost"

file_meta_type_name                = "ALC_EP_Meta.File"
activitydefinition_meta_type_name  = "ALC_EP_Meta.ActivityDefinition"
deployment_meta_type_name          = "ALC_EP_Meta.Deployment"
activityinterpreter_meta_type_name = "ALC_EP_Meta.ActivityInterpreter"
dockercontainer_meta_type_name     = "ALC_EP_Meta.DockerContainer"
alc_meta_type_name                 = "ALC"



setupcode_attribute_name           = "setupCode" 
interpretercode_attribute_name     = "Interpreter" 
executioncode_attribute_name       = "executionCode" 
resultcode_attribute_name          = "resultCode" 
constraintcode_attribute_name      = "constraintCode"
setupfolder_attribute_name         = "Setup_Folder"
defintion_attribute_name           = "Definition"

deploymentdefinition_attribute_name = "Definition"
includeroscore_attribute_name       = "IncludeRosCore"
timeout_attribute_name              = "timeout"
option_attribute_name               = "Option"
envvariable_attribute_name          = "env_variable"
portmap_attribute_name              = "port_map"
volumemap_attribute_name            = "volume_map"
command_attribute_name              = "command"

init_filename                      = "__init__.py"
setupcode_filename                 = "ActivityInterpreter.py"
executioncode_filename             = "ExecutionCode.py"
resultcode_filename                = "ResultCode.py"
constraintcode_filename            = "ConstraintCode.py"
deployment_filename                = "Dep.py"
definition_filename                = "Definition.json"

image_ref_name                     = "image"

image_name_attribute_name   = "ImageName" 
tag_attribute_name          = "Tag"

file_asset_attribute_name   = "fileAsset"
file_content_attribute_name   = "content"



number_re = re.compile(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]\d+)?')
var_re = re.compile(r'(?:\$\$)*(\$\w+|\${\w+})')

alc_working_dir_env_var_name = "ALC_WORKING_DIR"
jupyter_dir_name = "jupyter"
activity_definition_folder_root_name = "activity_definition"
model_folder_root_name               = 'model'
build_dir_name      = '.build'

root_node_name = "ROOT"
node_path_key_name = "nodePath"



class SetupActivity(PluginBase):
    def __init__(self, webgme, commit_hash, BRANCH_NAME, ACTIVE_NODE_PATH, ACTIVE_SELECTION_PATHS, NAMESPACE):
        PluginBase.__init__(self, webgme, commit_hash, BRANCH_NAME, ACTIVE_NODE_PATH, ACTIVE_SELECTION_PATHS, NAMESPACE)

        self.metadata_url = urlunsplit(
            ['http', "{0}:{1}".format(url_hostname, alc_port), "/rest/blob/metadata/", None, None]
        )
        self.download_url = urlunsplit(
            ['http', "{0}:{1}".format(url_hostname, alc_port), "/rest/blob/download/", None, None]
        )

        self.temp_dir = None
        self.config = self.get_current_config()
        project_info = self.project.get_project_info()
        self.project_owner = project_info.get(Keys.owner_key_name)
        self.project_name = project_info.get(Keys.name_key_name)
        self.userid = 'admin'
        self.alc_node = ''
        self.repo =''
        self.branch = ''
        self.tag = ''
        self.repo_root = ''
        
    
    def create_temp_dir(self, node_name):

        # FIXME: Creation of a folder inside the docker container "/tmp" is not allowed for this exercise.
        # This is because the folder and the json file and the downloaded files are needed outside of the
        # webgme docker to run the activity

        if (self.repo_root.exists()):
            activity_name = node_name
            temp_dir = Path(str(self.repo_root),model_folder_root_name, activity_definition_folder_root_name,  activity_name)
            if  not temp_dir.exists():
                temp_dir.mkdir(parents=True)
                self.temp_dir = temp_dir
                return

        alc_working_dir_name = os.environ.get(alc_working_dir_env_var_name, None)
        if alc_working_dir_name:
            project_dir_name = "{0}_{1}".format(self.project_owner, self.project_name)
            activity_name = node_name
            base_temp_dir = Path(alc_working_dir_name, jupyter_dir_name, activity_definition_folder_root_name,  project_dir_name, activity_name)
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
            logger.info('child_node_meta_type_name {0}'.format(str(child_node_meta_type_name)))
            child_node_list = child_node_map.get(child_node_meta_type_name, [])
            child_node_list.append(child_node)
            child_node_map[child_node_meta_type_name] = child_node_list

        return child_node_map
    
    
    def get_asset_attribute(self, node, asset_attribute_name, file_name):
        local_hash = self.core.get_attribute(node, asset_attribute_name)
        if not local_hash:
            logger.info('***************no asset***************')
            return None
        
        url = urljoin(self.metadata_url, local_hash)
        request = urllib.request.urlopen(url)
        data = json.loads(request.read())
        asset_file_name = data.get(Keys.name_key_name)

        temp_file_path = Path(self.temp_dir, file_name)
        lec_file_url = urljoin(urljoin(self.download_url, local_hash + "/"), asset_file_name)
        urllib.request.urlretrieve(lec_file_url, str(temp_file_path))

        return str(temp_file_path)

    def write_to_file(self,content, file_name, exec_mode = False):
        #if (not content):
        #    return

        temp_file_path = Path(self.temp_dir, file_name)
        if (filename.endswith('.json')):
            with temp_file_path.open("w") as json_fp:
                json.dump(content, json_fp, indent=4, sort_keys=True)
            return




        
        with temp_file_path.open("w") as f:
            f.write(content)
        if (exec_mode):
            st = os.stat(str(temp_file_path))
            os.chmod(str(temp_file_path), st.st_mode | stat.S_IEXEC)
    
    def get_files(self, files_list):
        for file_element in files_list:
            file_name = self.core.get_attribute(file_element, Attributes.name_attribute_name)
            value = self.core.get_attribute(file_element, file_content_attribute_name)
            if not bool(value):
                value = self.get_asset_attribute(file_element, file_asset_attribute_name, file_name)
            else:
                exec_mode = False
                if file_name.endswith('.sh') or file_name.endswith('bash'):
                    exec_mode = True
                self.write_to_file(value,file_name,exec_mode)
    
    def get_interpreter_info(self, interpreter_node):
        if not interpreter_node:
            logger.info('***************no interpreter_node***************')
        else:
            name = self.core.get_attribute(interpreter_node,Attributes.name_attribute_name)
            logger.info('***************interpreter_node {0} ***************'.format(name))

        child_node_map = self.get_child_nodes(interpreter_node)
        self.get_files(child_node_map.get(file_meta_type_name, []))
        setup_code = self.core.get_attribute(interpreter_node,setupcode_attribute_name)
        if (not setup_code):
            setup_code = self.core.get_attribute(interpreter_node,interpretercode_attribute_name)

        if (not setup_code):
            raise RuntimeError("Interpreter attribute is not set for Activity Interpreter")
  
        self.write_to_file('',init_filename)
        self.write_to_file(setup_code,setupcode_filename)

        activity_definition_json = self.core.get_attribute(interpreter_node,defintion_attribute_name)
        if (not activity_definition_json):
            raise RuntimeError("Definition attribute is not set for Activity Interpreter")
        self.write_to_file(activity_definition_json,definition_filename)

        
        # execution_code = self.core.get_attribute(interpreter_node,executioncode_attribute_name)
        # self.write_to_file(execution_code,executioncode_filename)
        # result_code = self.core.get_attribute(interpreter_node,resultcode_attribute_name)
        # self.write_to_file(result_code,resultcode_filename)
        # constraint_code = self.core.get_attribute(interpreter_node,constraintcode_attribute_name)
        # self.write_to_file(constraint_code,constraintcode_filename)


    def build_container_info(self, c):
        option = self.core.get_attribute(c,option_attribute_name)
        portmap = self.core.get_attribute(c,portmap_attribute_name)
        env_variables = self.core.get_attribute(c,envvariable_attribute_name)
        volume_map = self.core.get_attribute(c,volumemap_attribute_name)
        command  = self.core.get_attribute(c, command_attribute_name)
        name = self.core.get_attribute(c,Attributes.name_attribute_name)
        imageref = self.core.get_pointer_path(c, image_ref_name)
        if not imageref:
            return
        image = self.core.load_by_path(self.root_node, imageref)
        image_name = self.core.get_attribute(image, image_name_attribute_name)
        image_tag = self.core.get_attribute(image, tag_attribute_name)

        if (not option):
            option = {}

        option['volume']= volume_map
        option['port'] = portmap

        ret = {}
        ret['name']=name
        ret['environment']=env_variables
        ret['options'] = option
        ret['image'] = image_name+':'+image_tag
        ret['input_file']= 'launch_activity_output.json'
        ret['command'] = os.path.join(str(self.temp_dir), command)

        return ret


    def build_deployment_info(self,containers):
        containers = []
        for c in containers:
            c_value = self.build_container_info(c)
            containers.append(c_value)
        return containers

        
    
    def get_deployment_info(self,deployment_node):
        dep_dict = {}
        child_node_map = self.get_child_nodes(deployment_node)
        #definition = self.core.get_attribute(deployment_node, deploymentdefinition_attribute_name)
        include_rosnode = self.core.get_attribute(deployment_node, includeroscore_attribute_name)
        timeout = self.core.get_attribute(deployment_node, timeout_attribute_name)
        containers  = self.build_deployment_info(child_node_map.get(dockercontainer_meta_type_name, []))
        name = self.core.get_attribute(self.active_node,Attributes.name_attribute_name)
        if (not containers):
            return False
        dep_dict['name'] = name
        dep_dict['base_dir'] = '.'
        dep_dict['results_dir']= '.'
        dep_dict['ros_master_image'] = None
        if (include_rosnode):
            dep_dict['ros_master_image'] = 'ros:kinetic-ros-core'
        dep_dict['containers'] = containers
        dep_dict['timeout']    = timeout
        dep_dict_string = json.dumps(dep_dict, indent=4, sort_keys=True)
        dep_content = 'dep_dict = '+dep_dict_string
        self.write_to_file(dep_content, deployment_filename)
        return True
    
    def get_ALC_node (self):
        child_node_list = self.core.load_children(self.root_node)
        for child_node in child_node_list:
            if not child_node:
                continue
            child_node_meta_type = self.core.get_meta_type(child_node)
            child_node_meta_type_name = self.core.get_fully_qualified_name(child_node_meta_type)
            if child_node_meta_type_name.endswith(alc_meta_type_name):
                self.alc_node = child_node
                break
    
    def get_repo_info (self):
        if (self.alc_node == ''):
            self.get_ALC_node()
        if (self.alc_node == ''):
            raise RuntimeError("Unable to access ALC node")
        self.repo = self.core.get_attribute(self.alc_node,'repo')
        self.branch = self.core.get_attribute(self.alc_node,'branch')
        self.tag = self.core.get_attribute(self.alc_node,'tag')
        if (self.repo):
            self.setup_repo()

    def create_repo_root_dir(self):
        alc_working_dir_name = os.environ.get(alc_working_dir_env_var_name, None)
        if alc_working_dir_name:
            self.repo_root = Path(alc_working_dir_name, build_dir_name, self.projectName)
            if  not self.repo_root.exists():
                self.repo_root.mkdir(parents=True)
        else:
            raise RuntimeError("Environment Variable: \"{0}\" is not found.".format(alc_working_dir_env_var_name))

    def setup_repo(self):
        try:
            self.create_repo_root_dir()
            logger.info(' trying to setup repo at {0}'.format(str(self.repo_root)))
            if self.repo_root.exists():
                r =  RepoSetup()
                dst_folder_path = os.path.join(str(self.repo_root),self.repo)
                r.clone_repo(self.repo_root, self.repo, self.branch, self.tag, logger)
                self.repo_root = Path(dst_folder_path)
        except Exception as err:
            logger.error(' Error encountered while setting up the repo')
            msg = str(err)
            logger.info("exception {0}".format(msg))
            traceback_msg = traceback.format_exc()
            logger.info(traceback_msg)
            sys_exec_info_msg = sys.exc_info()[2]
            logger.info(sys_exec_info_msg)
            raise RuntimeError("Unable to setup repo")


    def add_to_repo(self):
        if (not self.repo):
            return
        try:
            if self.repo_root.exists():
                r =  RepoSetup()

                r.add_to_repo(self.repo_root,'commited files from setup activity', self.userid, logger)
        except Exception as err:
            logger.error(' Error encountered while commiting to the repo ')
            msg = str(err)
            logger.info("exception {0}".format(msg))
            traceback_msg = traceback.format_exc()
            logger.info(traceback_msg)
            sys_exec_info_msg = sys.exc_info()[2]
            logger.info(sys_exec_info_msg)
            raise RuntimeError("Unable to commit and push to  repo")





    def main(self):
        try:
            core = self.core
            root_node = self.root_node
            active_node = self.active_node
            active_node_meta_type = self.core.get_meta_type(active_node)
            name = core.get_attribute(active_node, 'name')
            active_node_meta_type_name = self.core.get_fully_qualified_name(active_node_meta_type)
            logger.info('active_node_meta_type_name {0}'.format(str(active_node_meta_type_name)))
            

            if not active_node_meta_type == self.META[activitydefinition_meta_type_name]:
                    raise RuntimeError("Active Node should be of type ActivityDefinition")

            self.get_repo_info()
            self.create_temp_dir(name)

            child_node_map = self.get_child_nodes(active_node)
            interpreter_node = child_node_map.get(activityinterpreter_meta_type_name, [])
            if (not interpreter_node):
                raise Exception("Problems in setting up activity... no interpreter node")


            self.get_interpreter_info(interpreter_node[0])

            
            self.add_to_repo()
            self.result_set_success(True)
                
            self.core.set_attribute(self.active_node,setupfolder_attribute_name,str(self.temp_dir))
            #ret = self.get_deployment_info(child_node_map.get(deployment_meta_type_name, []))
            #if (not ret):
            #    raise Exception("Problems in setting up activity... Deployment information ")
            
        
            commit_info = self.util.save(root_node, self.commit_hash, 'master', 'Python plugin updated the model')
            logger.info('committed :{0}'.format(commit_info))
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
            self.result_set_error('SetupActivity Plugin: Error encountered.  Check result details.')
            self.result_set_success(False)
            exit()
