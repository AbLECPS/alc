"""
This is where the implementation of the plugin code goes.
The ImportData-class is imported from both run_plugin.py and run_debug.py
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
from KeysAndAttributes import Keys, Attributes, References
from alc_utils.slurm_executor import WebGMEKeys
import alc_utils.alc_model_updater as model_updater

# Setup a logger
logger = logging.getLogger('ImportData')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)  # By default it logs to stderr..
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

url_hostname = "localhost"
number_re = re.compile(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]\d+)?')
var_re = re.compile(r'(?:\$\$)*(\$\w+|\${\w+})')

alc_working_dir_env_var_name = "ALC_WORKING_DIR"
jupyter_dir_name = "jupyter"
user_data = "user_data"
result_meta_type_names = ['ALCMeta.Result', 'ALC_EP_Meta.Result']

config_assetdata_name = 'assetdata'
config_pathtodocument_name = 'pathtodocument'
config_recordname_name = 'recordname'


class ImportData(PluginBase):
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

        
        self.metadata_url = urlunsplit(
            ['http', "{0}:{1}".format(url_hostname, webgme_port), "/rest/blob/metadata/", None, None]
        )
        self.download_url = urlunsplit(
            ['http', "{0}:{1}".format(url_hostname, webgme_port), "/rest/blob/download/", None, None]
        )

        project_info = self.project.get_project_info()
        self.project_owner = project_info.get(Keys.owner_key_name)
        self.project_name = project_info.get(Keys.name_key_name)

        self.folder_path = None
        self.config = self.get_current_config()
        self.asset_data = self.config.get(config_assetdata_name,None)
        self.path_to_document = self.config.get(config_pathtodocument_name,None)
        self.record_name = self.config.get(config_recordname_name, None)
    
    
    def check_result_node(self):

        child_node_list = self.core.load_children(self.active_node)
        for child_node in child_node_list:
            if not child_node:
                continue

            child_node_meta_type = self.core.get_meta_type(child_node)
            child_node_meta_type_name = self.core.get_fully_qualified_name(child_node_meta_type)
            if child_node_meta_type_name in result_meta_type_names:
                return True

        return False

    def create_folder(self):

        # FIXME: Creation of a folder inside the docker container "/tmp" is not allowed for this exercise.
        # This is because the folder and the json file and the downloaded files are needed outside of the
        # webgme docker to run the activity
        node_name = self.core.get_attribute(self.active_node, Attributes.name_attribute_name)
        record_name = self.record_name

        if not record_name:
            raise Exception("Record Name was not set.")

        alc_working_dir_name = os.environ.get(alc_working_dir_env_var_name, None)
        if alc_working_dir_name:
            project_dir_name = "{0}_{1}".format(self.project_owner, self.project_name)
            activity_name = node_name
            base_temp_dir = Path(alc_working_dir_name, jupyter_dir_name, project_dir_name, activity_name, user_data)
            if not base_temp_dir.exists():
                base_temp_dir.mkdir(parents=True)
            
            folder_dir = Path(base_temp_dir,record_name)
            counter = 0 
            while folder_dir.exists():
                counter +=1
                folder_dir = Path(base_temp_dir,record_name+'_'+str(counter))

            self.folder_path = folder_dir

            if not self.folder_path.exists():
                self.folder_path.mkdir(parents=True)
        else:
            raise Exception("Environment Variable: \"{0}\" is not found.".format(alc_working_dir_env_var_name))
    
    def download_asset(self):
        local_hash = self.asset_data
        url = urljoin(self.metadata_url, local_hash)
        request = urllib.request.urlopen(url)
        data = json.loads(request.read())
        file_name = data.get(Keys.name_key_name)
        temp_file_path = Path(self.folder_path, file_name)
        file_url = urljoin(urljoin(self.download_url, local_hash + "/"), file_name)
        urllib.request.urlretrieve(file_url, str(temp_file_path))
        return str(temp_file_path)

    def check_path_to_document(self):
        doc_path = self.path_to_document
        alc_working_dir_name = os.environ.get(alc_working_dir_env_var_name, None)
        if not alc_working_dir_name:
            raise Exception("Environment Variable: \"{0}\" is not found.".format(alc_working_dir_env_var_name))
        child = os.path.realpath(doc_path)
        parent = os.path.realpath(alc_working_dir_name)
        if os.path.exists(child):
            relative = os.path.relpath(child, start=parent)
            if relative.startswith(os.pardir):
                self.folder_path = child
                return True
        combined_path = os.path.join(parent,doc_path)
        if os.path.exists(combined_path):
            self.folder_path = combined_path
            return True
        combined_path = os.path.join(parent,'jupyter',doc_path)
        if os.path.exists(combined_path):
            self.folder_path = combined_path
            return True
        
        return False

    def get_resultnode_from_router(self):
        
        project_info     = self.project.get_project_info()
        project_owner    = project_info[WebGMEKeys.project_owner_key]
        project_name     = project_info[WebGMEKeys.project_name_key]
        node_name        = self.core.get_attribute(self.active_node, Attributes.name_attribute_name)
        active_node_path = self.core.get_path(self.active_node)
        execution_name   = self.record_name
        
        modifications = {
            "createdAt": int(round(time.time() * 1000)),
            "activity" : node_name,
            #"resultDir" : self.folder_path,
            "datainfo"  : json.dumps({'directory':str(self.folder_path)}),
            "jobstatus": "Finished"
        }
        
        sets = {}
        return model_updater.create_data_node(logger, project_owner, project_name, active_node_path, execution_name, modifications,sets)


    def main(self):
        try:
            print(self.config)
                
            if not self.check_result_node():
                raise Exception("Activity needs to contain exactly one result node. None found. Please add one before proceeding.")

            if (self.asset_data and self.path_to_document):
                raise Exception("Both artifact and path to the existing document are supplied. Only one should be set")

            if (self.asset_data):
                self.create_folder()
                self.download_asset()
            elif (self.path_to_document):
                if (not self.check_path_to_document()):
                    raise Exception("Specified path does not exist inside $ALC_WORKING_DIR")
            else:
                raise Exception("One of artifact or path to the existing document are supplied. None has been set.")
            result_node_path = self.get_resultnode_from_router()
            result_node = self.core.load_by_path(self.root_node, result_node_path)
            if (result_node):
                result_node_name = self.core.get_attribute(result_node, Attributes.name_attribute_name)
                if (result_node_name):
                    self.create_message(self.active_node, 'Created Result node of name - '+result_node_name, 'error')
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
            self.result_set_error('ImportData Plugin: Error encountered.  Check result details.')
            self.result_set_success(False)
            exit()

        

        