"""
This is where the implementation of the plugin code goes.
The IDELaunch-class is imported from both run_plugin.py and run_debug.py
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
from alc_utils.check_docker_status import CheckDockerStatus
import stat
import git
from git import Repo
import shutil
import socket

from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from alc_utils.setup_repo import RepoSetup

template_dir_path = Path(Path(__file__).absolute().parent, "templates")
file_loader = FileSystemLoader(str(template_dir_path))
environment = Environment(loader=file_loader)
start_script_template = environment.get_template("start_dind.sh")
stop_script_template = environment.get_template("stop_dind.sh")

slurm_job_params_filename = "slurm_params.json"

# Setup a logger
logger = logging.getLogger('IDELaunch')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)  # By default it logs to stderr..
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

url_hostname = "localhost"
alc_working_dir_env_var_name = "ALC_WORKING_DIR"
docker_dir_name = ".docker"
user_dir_name = ".users"
docker_prefix ="codeserver"



#export ALC_DOCKERROOT=$ALC_WORKING_DIR/docker
#export REPO_ROOT=$ALC_DOCKERROOT/users/user${did}




class IDELaunch(PluginBase):

    project_name_key = "name"
    project_owner_key = "owner"

    def __init__(
            self,
            webgme, commit_hash, BRANCH_NAME, ACTIVE_NODE_PATH, ACTIVE_SELECTION_PATHS, NAMESPACE, USERID,
            config=None, **kwargs
    ):
        PluginBase.__init__(self, webgme, commit_hash, BRANCH_NAME, ACTIVE_NODE_PATH, ACTIVE_SELECTION_PATHS, NAMESPACE)
        self.userid = USERID

        webgme_port = kwargs.pop(WebGMEKeys.webgme_port_key, 8888)
        self.slurm_params = {
            WebGMEKeys.webgme_port_key: webgme_port
        }

        self.timeout_param = 0
        self.user_url = urlunsplit(
            ['http', "{0}:{1}".format(url_hostname, webgme_port), "/api/users", None, None]
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
            self.config[SlurmParameterKeys.job_name_key] = 'launch_ide'
        self.repo_info = ''
        self.branch_info= ''
        self.tag_info= ''
        self.user_dir=''
        self.ALC_SSH_PORT='22'
        self.ALC_SSH_HOST=socket.gethostbyname(socket.gethostname())


    def create_temp_dir(self):

        # FIXME: Creation of a folder inside the docker container "/tmp" is not allowed for this exercise.
        # This is because the folder and the json file and the downloaded files are needed outside of the
        # webgme docker to run the activity
        alc_working_dir_name = os.environ.get(alc_working_dir_env_var_name, None)
        if alc_working_dir_name:
            project_dir_name = "{0}_{1}".format(self.project_owner, self.project_name)
            base_temp_dir = Path(alc_working_dir_name, docker_dir_name, self.userid)
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

    def create_user_dir(self):
        alc_working_dir_name = os.environ.get(alc_working_dir_env_var_name, None)
        if alc_working_dir_name:
            user_folder = 'user{0}'.format(self.userid)
            self.user_dir = Path(alc_working_dir_name, docker_dir_name, user_dir_name, user_folder)
            if  not self.user_dir.exists():
                self.user_dir.mkdir(parents=True)
        else:
            raise Exception("Environment Variable: \"{0}\" is not found.".format(alc_working_dir_env_var_name))

    
    def clone_repo(self):
        try:
            self.create_user_dir()
            if self.user_dir.exists():
                if (self.repo_info == ''):
                    return True
                r =  RepoSetup()
                setup_status = r.clone_repo(self.user_dir, self.repo_info, self.branch_info, self.tag_info, logger, True)
                if (not setup_status):
                    self.create_message(self.active_node, 'The repository could not be pulled cleanly. Please check the IDE.', 'info')
                return setup_status
        except Exception as err:
            logger.error(' Error encountered while setting up the repo')
            msg = str(err)
            logger.info("exception {0}".format(msg))
            traceback_msg = traceback.format_exc()
            logger.info(traceback_msg)
            sys_exec_info_msg = sys.exc_info()[2]
            logger.info(sys_exec_info_msg)
        return False



    def remove_docker_dir(self):

        alc_working_dir_name = os.environ.get(alc_working_dir_env_var_name, None)
        if alc_working_dir_name:
            docker_dir = "docker{0}".format(self.userid)
            docker_dir_path = Path(alc_working_dir_name, docker_dir_name, docker_dir)
            if docker_dir_path.exists():
                try:
                    logger.info('removing docker folder {0}'.format(str(docker_dir_path)))
                    shutil.rmtree(str(docker_dir_path))#.unlink()
                    logger.info('removed docker folder {0}'.format(str(docker_dir_path)))
                except OSError as e:
                    logger.error("Error, unable to delete temporary docker folder : %s : %s" % (str(docker_dir_path), e.strerror))
            else:
                logger.info('docker dir path does not exist .... removing docker folder {0}'.format(str(docker_dir_path)))

                    
                

    def compile_url_message(self):
        ret = {}
        url_info = self.core.get_attribute(self.active_node,'url')
        if (url_info != ''):
            ret = json.loads(url_info)
        ret[self.userid] = self.userid
        y = json.dumps(ret)
        logger.info('compile_url_message --------- {0}'.format(y))
        self.core.set_attribute(self.active_node,'url',y)
        self.create_message(self.active_node, 'Switch to the IDE visualizer to launch the urls', 'info')
        return ret
    
    def remove_url_message(self):
        ret = {}
        url_info = self.core.get_attribute(self.active_node,'url')
        if (url_info != ''):
            ret = json.loads(url_info)
            if (self.userid in ret.keys()):
                del ret[self.userid]
        y = json.dumps(ret)
        logger.info('remove_url_message --------- {0}'.format(y))
        self.core.set_attribute(self.active_node,'url',y)
        self.create_message(self.active_node, 'Switch to the IDE visualizer to launch the urls', 'info')
        return ret



    def generate_termination_template(self):
        # Fill out top-level launch template
        main_template = stop_script_template
        code = main_template.render(did=self.userid)
        main_path = Path(str(self.temp_dir), 'run.sh')
        with main_path.open("w") as f:
            f.write(code)
        st = main_path.stat()
        main_path.chmod(st.st_mode | stat.S_IEXEC)
    
    def generate_launch_template(self):
        # Fill out top-level launch template
        main_template = start_script_template
        code = main_template.render(did=self.userid)
        main_path = Path(str(self.temp_dir), 'run.sh')
        with main_path.open("w") as f:
            f.write(code)
        st = main_path.stat()
        main_path.chmod(st.st_mode | stat.S_IEXEC)

    
    def terminate_docker(self,dockerModule):
        self.create_temp_dir()
        self.generate_termination_template()
        dockerModule.terminate_docker(logger)
        self.remove_docker_dir()
        self.remove_url_message()

    def submit_slurm_job(self):
        os.environ['ALC_SSH_PORT'] = self.ALC_SSH_PORT
        os.environ['ALC_SSH_HOST'] = self.ALC_SSH_HOST
        SlurmSetup.setup_job(
                self,
                "SetupIDE"+self.userid,
                self.time_stamp,
                self.timeout_param,
                "-VSCode",
                True,
                str(self.temp_dir)
            )

    def launch_docker(self):
        self.create_temp_dir()
        self.generate_launch_template()
        ##self.execute_docker_launch()
        self.submit_slurm_job()
        self.compile_url_message()

    
    def main(self):
        try:
            core = self.core
            active_node = self.active_node
            self.repo_info = self.core.get_attribute(self.active_node,'repo')
            self.branch_info = self.core.get_attribute(self.active_node,'branch')
            self.tag_info = self.core.get_attribute(self.active_node,'tag')
        
            docker_name = "{0}{1}".format(docker_prefix, self.userid)
            dockerModule = CheckDockerStatus(docker_name)
            status, statusstring = dockerModule.check_status(logger)
            if (status == 1 and self.config.get('Operation')=='Start'):
                message = self.compile_url_message()
                self.result_set_success(True)
                commit_info = self.util.save(self.root_node, self.commit_hash, 'master', 'LaunchIDE')
                exit()

            if ((status == 1 and self.config.get('Operation')=='Stop') or (status ==0)):
                self.terminate_docker(dockerModule)
                self.result_set_success(True)
                commit_info = self.util.save(self.root_node, self.commit_hash, 'master', 'LaunchIDE')
                exit()

            if (status == -1 and self.config.get('Operation')=='Start'):
                if (self.repo_info):
                    status = self.clone_repo()
                    if (not status):
                        self.create_message(self.active_node, 'The repository could not be pulled cleanly. Please check the IDE.', 'info')
                self.launch_docker()
                self.result_set_success(True)
                commit_info = self.util.save(self.root_node, self.commit_hash, 'master', 'LaunchIDE')
                exit()

            if (status == -1 and self.config.get('Operation')=='Stop'):
                self.create_message(self.active_node,"IDE is not running currently. Stop command ignored.")
                self.result_set_success(True)
                exit()
            
            self.result_set_error("unknown execution choice... not doing anything...")
            self.result_set_success(False)

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
            self.result_set_error('LaunchIDE Plugin: Error encountered.  Check result details.')
            self.result_set_success(False)
            exit()
