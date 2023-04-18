"""
This is where the implementation of the plugin code goes.
The ExportArtifact-class is imported from both run_plugin.py and run_debug.py
"""

from future.utils import iteritems
import sys
import socket
import logging
import traceback
from webgme_bindings import PluginBase
import time
import json
import os
import stat
import jinja2
import math
from alc_utils.slurm_executor import WebGMEKeys
import alc_utils.common as alc_common
from ros_gen import SystemLaunchGen
from alc_utils.setup_repo import RepoSetup
from pathlib import Path
import tarfile
import base64


# Setup a logger
logger = logging.getLogger('ExportArtifact')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)  # By default it logs to stderr..
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Directory for JINJA templates
_package_directory = os.path.dirname(__file__)
alc_working_dir_env_var_name = "ALC_WORKING_DIR"
alc_meta_type_name  = "ALC"
dump_dir_name = ".target"

class ExportArtifact(PluginBase):
    def __init__(self, *args, **kwargs):
        super(ExportArtifact, self).__init__(*args, **kwargs)
        self.paramValues = {}
        self.lecParamValues = {}
        self.containerInfo = {}
        self.exptParamSetup = {}
        self.lecParamSetup = {}
        self.lecCodeSetup = {}
        self.zipFileSetup = {}
        
        
        self.config = self.get_current_config()
        self.generateROSLaunch = True
        self.active_node_meta_type = ''
        self.zipFileContents = {}

        self.alc_node = None
        self.repo = ''
        self.branch = ''
        self.tag = ''
        self.repo_root = ''
        self.repo_home = ''

        self.target_launch_filename= ''
        self.target_launch_file = ''
        self.target_ros_master_ip= ''
        self.target_ros_master_port=''
        self.target_local_ros_master_port_mapping=''
        self.target_lec_deployment_key= {}
        self.target_repo_archive_path = ''
        self.target_lec_archive_path = ''
    
    

    def add_zip_file_info(self, foldername, filenames):
        if len(filenames) == 0:
            return
        keys = self.zipFileSetup.keys()
        if foldername not in keys:
            self.zipFileSetup[foldername] = []

        for f in filenames:
            if f not in self.zipFileSetup[foldername]:
                self.zipFileSetup[foldername].append(f)

    def add_zip_content(self, filepath, content):
        logger.info('adding content for '+filepath)
        self.zipFileContents[filepath] = content

    

    # def get_lec_data(self, node_id):

    #     node = self.core.load_by_path(self.root_node, node_id)
    #     data = self.core.get_attribute(node, 'data')

    #     datadict = ''
    #     datastr = ''
    #     if data:
    #         datastr = self.get_file(data)
    #     dataval = self.core.get_attribute(node, 'datainfo')
    #     if dataval:
    #         # dataval = re.sub("null", 'NaN',dataval)
    #         try:
    #             datadict = json.loads(dataval, strict=False)
    #             self.alldata[node_id] = 'LData'
    #             #self.core.add_member(self.resultnode, 'LData', node)
    #             self.resultnode_info['LData'].append(node_id)
    #         except Exception as e:
    #             datadict = ''
    #             logger.info('problem in parsing lec1 "{0}"'.format(e))
    #     elif datastr:
    #         logger.info(' datastr = ' + datastr)
    #         n = len(datastr)
    #         datastr1 = datastr[2: n - 6]
    #         try:
    #             # datastr1 = re.replace("null", 'NaN',dataval)
    #             datadict = json.loads(datastr1, strict=False)
    #             self.alldata[node_id] = 'LData'
    #         except Exception as e:
    #             datadict = ''
    #             logger.info('problem in parsing lec2 "{0}"'.format(e))
    #     return datadict

    # def build_lec_param(self):
    #     leckeys = self.lecParamValues.keys()
    #     self.lecParamSetup = {}
    #     for key in leckeys:
    #         self.lecParamSetup[key] = self.get_lec_data(self.lecParamValues[key])

    #     self.outputLECParam = json.dumps(self.lecParamSetup)

    

    # Function which calls the ROSLaunchGenerator plugin and adds the generated files to the zip file
    def add_ros_launch_files(self):
        self.logger.info("Invoking ROSLaunch plugin...")

        # Initialize ROS Launch Generator plugin extension and invoke on active_node
        ros_launch_gen = SystemLaunchGen(self)
        artifact_content, container_info = ros_launch_gen.gen_launch_file(self.active_node)
        self.target_launch_filename= ros_launch_gen.target_launch_filename
        self.target_launch_file = ros_launch_gen.target_launch_file
        self.target_ros_master_ip= ros_launch_gen.target_ros_master_ip
        self.target_ros_master_port=ros_launch_gen.target_ros_master_port
        self.target_local_ros_master_port_mapping=ros_launch_gen.local_ros_master_port_mapping
        self.target_lec_deployment_key= ros_launch_gen.lec_deployment_key
        self.containerInfo = container_info
        # Add each generated launch file to the zip file contents
        for file_name, file_content in iteritems(artifact_content):
            self.zipFileContents["launch_files/%s" % file_name] = file_content
            self.add_zip_file_info('launch_files', [file_name])
        
    def add_to_artifact(self,archivename, contents):
        artifact_content = {}
        if (len(contents.keys())==0):
            return 
        for name in contents.keys():
            filename = contents[name]
            data = ''
            with open(filename,"rb") as f:
                data = f.read()
                #data = base64.b64encode(data)
            artifact_content[name]=data

        self.add_artifact(archivename, artifact_content)

    def add_file_archive(self, filename, key, mode='rb'):
        with open(filename,mode) as f:
            data = f.read()
            #if (mode == 'r'):
                #data = base64.b64encode(data)
            self.add_file(key, data)

    def dump_target_artifacts(self, dump_artifacts):
        if not dump_artifacts:
            return False

        logger.info('in setup_and_build_repo')
        child_node_list = self.core.load_children(self.root_node)
        for child_node in child_node_list:
            if not child_node:
                continue
            child_node_meta_type = self.core.get_meta_type(child_node)
            child_node_meta_type_name = self.core.get_fully_qualified_name(child_node_meta_type)
            if child_node_meta_type_name.endswith(alc_meta_type_name):
                logger.info('in setup_and_build_repo - got alc_node')
                self.alc_node = child_node
                self.repo = self.core.get_attribute(self.alc_node,'repo')
                self.branch = self.core.get_attribute(self.alc_node,'branch')
                self.tag = self.core.get_attribute(self.alc_node,'tag')

                logger.info('in setup_and_build_repo - got repo info == '+self.repo)
                break

        alc_working_dir_name = os.environ.get(alc_working_dir_env_var_name, None)
        if not alc_working_dir_name:
            logger.info('environment variable {0} not defined '+alc_working_dir_env_var_name)
            return False

        if (self.repo):
            alc_working_dir_name = os.environ.get(alc_working_dir_env_var_name, None)
            if alc_working_dir_name:
                logger.info('in dump_target repo alc_working_dir ' +alc_working_dir_name)
                repo_root = Path(alc_working_dir_name, dump_dir_name, self.repo)
                tar_filepath = Path(alc_working_dir_name, dump_dir_name, 'source.tar')
                tarfilename = str(tar_filepath)
                r =  RepoSetup()
                result = r.archive_repo(tar_filepath,repo_root, self.repo, self.branch, self.tag, logger)
                if result:
                    self.target_repo_archive_path = tar_filepath
                    self.add_file_archive(self.target_repo_archive_path,'source.tar')


        #archive the folder
        lec_keys = self.target_lec_deployment_key.keys()
        
        if (len(lec_keys)):
            logger.info('lecs to archive')
            tar_filepath = Path(alc_working_dir_name, dump_dir_name, 'trainedartifacts_')
            tarfilename_ = str(tar_filepath)
            
            for lk in lec_keys:


                logger.info('lec key  {0}'.format(lk))
                logger.info('lec path  {0}'.format(self.target_lec_deployment_key[lk]))
                tarfilename = tarfilename_+lk+".tar.gz"
                localfilename = 'trainedartifacts_' +lk
                filepath = self.target_lec_deployment_key[lk]
                pos1 = filepath.find('LEC2')
                #if (pos1 > -1):
                #    continue

                pos = filepath.find('jupyter')
                if (pos == -1):
                    continue
                
                filepath = filepath[pos:]
                totalfilepath = os.path.join(alc_working_dir_name,filepath)

                artifactname = localfilename
                contents={}
                for root, dirs, files in os.walk(totalfilepath, topdown=False):
                    for name in files:
                        file_name = os.path.join(root, name)
                        pos = file_name.find('jupyter')
                        if (pos == -1):
                            continue
                        file_name_key = file_name[pos:]
                        newfilepath = os.path.join('alc_workspace',file_name_key)
                        contents[newfilepath] = file_name
                self.add_to_artifact(artifactname, contents)
        logger.info('finished archiving')
        return True



    def main(self):
        try:

            
            self.active_node_meta_type = self.core.get_meta_type(self.active_node)
            activenodename = self.core.get_attribute(self.active_node, 'name')
            
            # FIXME: Any other job types where this may be needed?
            if not (
                    self.active_node_meta_type == self.META["ALCMeta.ExperimentSetup"] or
                    self.active_node_meta_type == self.META["ALCMeta.RLTrainingSetup"]
                ):
                raise RuntimeError(
                    "ExportArchive can be run on ExperimentSetup or RLTrainingSetup model"
                )
            
            self.exptParamSetup['generated_ros_launch'] = True
            self.add_ros_launch_files()
            if not self.containerInfo:
                logger.error('ContainerInfo empty in deployment nodes. Please Check!')
                raise RuntimeError('ContainerInfo empty in deployment nodes. Please Check!')
            try:
                if (self.dump_target_artifacts(True)):
                    self.logger.info('dumped target info')
                    self.result_set_success(True)
                    return
            except Exception as e:
                logger.error('Error encountered while executing experiment. Please check!')
                logger.error('Error ' + str(e))
                raise RuntimeError('Error encountered while executing experiment. Please check! ' + str(e))

            logger.info('Skipped Export...')
            self.result_set_success(True)
        
        except Exception as err:
            self.send_notification(str(err))
            #raise err
            msg = str(err)
            logger.info("exception {0}".format(msg))
            traceback_msg = traceback.format_exc()
            logger.info(traceback_msg)
            sys_exec_info_msg = sys.exc_info()[2]
            logger.info(sys_exec_info_msg)
            self.create_message(self.active_node, msg, 'error')
            self.create_message(self.active_node, traceback_msg, 'error')
            self.create_message(self.active_node, str(sys_exec_info_msg), 'error')
            self.result_set_error('LaunchExpt Plugin: Error encountered.  Check result details.')
            self.result_set_success(False)
            exit()