"""
This is where the implementation of the plugin code goes.
The DockerImageBuilder-class is imported from both run_plugin.py and run_debug.py
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

# Setup a logger
logger = logging.getLogger('DockerImageBuilder')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)  # By default it logs to stderr..
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
alc_port = 8888
url_hostname = "localhost"

file_meta_type_name         = "ALC_EP_Meta.File"
dockerimage_meta_type_name  = "ALC_EP_Meta.DockerImage"
image_name_attribute_name   = "ImageName" 
tag_attribute_name          = "Tag"
build_script_attribute_name = "buildScript"
dockerfile_attribute_name   = "Dockerfile"
file_asset_attribute_name   = "fileAsset"
file_content_attribute_name   = "content"

dockerfile_name   = "Dockerfile"
buildscript_name  = "build.sh"
logfile_name = "slurm_exec_log.txt"


docker_image_folder_root_name = "DockerLibrary"


number_re = re.compile(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]\d+)?')
var_re = re.compile(r'(?:\$\$)*(\$\w+|\${\w+})')

alc_working_dir_env_var_name = "ALC_WORKING_DIR"
jupyter_dir_name = "jupyter"

root_node_name = "ROOT"
node_path_key_name = "nodePath"
default_build_script_contents = "\
#!/bin/bash\n\
docker build -t $1:$2 ."


class DockerImageBuilder(PluginBase):

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
        self.image_name = None
        self.image_tag = "latest"
    
    def create_temp_dir(self, node_name):

        # FIXME: Creation of a folder inside the docker container "/tmp" is not allowed for this exercise.
        # This is because the folder and the json file and the downloaded files are needed outside of the
        # webgme docker to run the activity
        alc_working_dir_name = os.environ.get(alc_working_dir_env_var_name, None)
        if alc_working_dir_name:
            project_dir_name = "{0}_{1}".format(self.project_owner, self.project_name)
            activity_name = node_name
            base_temp_dir = Path(alc_working_dir_name, jupyter_dir_name, docker_image_folder_root_name,  project_dir_name, activity_name)
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
        temp_file_path = Path(self.temp_dir, file_name)
        with temp_file_path.open("w") as f:
            f.write(content)
        if (exec_mode):
            st = os.stat(str(temp_file_path))
            os.chmod(str(temp_file_path), st.st_mode |  0o555)
    
    def get_files(self, files_list):
        for file_element in files_list:
            file_name = self.core.get_attribute(file_element, Attributes.name_attribute_name)
            value = self.core.get_attribute(file_element, file_content_attribute_name)
            if not bool(value):
                value = self.get_asset_attribute(file_element, file_asset_attribute_name, file_name)
            else:
                self.write_to_file(value,file_name)

    def get_docker_info(self, active_node):
        dockerfile_contents = self.core.get_attribute(active_node,dockerfile_attribute_name)
        self.write_to_file(dockerfile_contents,dockerfile_name)

        build_script_contents = self.core.get_attribute(active_node,build_script_attribute_name)
        if (not build_script_contents):
            build_script_contents = default_build_script_contents
        self.write_to_file(build_script_contents,buildscript_name,True)

        self.image_name = self.core.get_attribute(active_node,image_name_attribute_name)
        self.image_tag = self.core.get_attribute(active_node,tag_attribute_name)
        if (self.image_tag == ''):
            self.image_tag = 'latest'
    
    def dump_log(self,std_out, std_err, returncode):
        log_output = []
        
        if (returncode !=0):
            log_output.append('Return code : '+str(returncode))
        
        if std_err:
            log_output.append('Error: ')
            log_output.append('--------')
            log_output.append(std_err.decode('utf-8'))

        log_output.append('Output: ')
        log_output.append('--------')
        log_output.append(std_out.decode('utf-8'))

        
        #logger.info(str(log_output))

        self.write_to_file('\n'.join(log_output),logfile_name)

        if (returncode !=0):
            raise Exception("Error encountered while building the docker. Check log message!")


    def build_docker(self):
        try:
            
            cmd = 'cd '+str(self.temp_dir) +';echo $PWD;ls -al;echo '+buildscript_name +';cat '+buildscript_name +'; ./'+buildscript_name + ' '+self.image_name + ' '+self.image_tag
            logger.info(cmd)
            process = subprocess.Popen(cmd, shell=True, \
                                       stdout=subprocess.PIPE, \
                                       stderr=subprocess.PIPE,\
                                       cwd = str(self.temp_dir))
            stdout, stderr = process.communicate()
            
        except Exception as err:
            msg = str(err)
            logger.info("exception while building docker {0}".format(msg))
            raise

        self.dump_log(stdout, stderr, process.returncode)

        if process.returncode != 0:
            raise RuntimeError("Non-zero return code while building the docker")
    
    def main(self):
        try:
            core = self.core
            root_node = self.root_node
            active_node = self.active_node
            name = core.get_attribute(active_node, 'name')
            active_node_meta_type = self.core.get_meta_type(active_node)

            if not active_node_meta_type == self.META[dockerimage_meta_type_name]:
                raise RuntimeError("Active Node should be of type DockerImage")

            self.create_temp_dir(name)

            child_node_map = self.get_child_nodes(active_node)
            self.get_files(child_node_map.get(file_meta_type_name, []))
            self.get_docker_info(active_node)
            self.build_docker()


            logger.info('ActiveNode at "{0}" has name {1}'.format(core.get_path(active_node), name))
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
            self.result_set_error('DockerImageBuilder Plugin: Error encountered.  Check result details.')
            self.result_set_success(False)
            exit()
