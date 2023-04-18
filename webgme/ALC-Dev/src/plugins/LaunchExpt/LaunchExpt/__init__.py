"""
This is where the implementation of the plugin code goes.
The LaunchExpt-class is imported from both run_plugin.py and run_debug.py
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
import alc_utils.slurm_executor as slurm_executor
from alc_utils.slurm_executor import WebGMEKeys
from alc_utils.update_job_status_daemon import Keys as UpdateKeys
import alc_utils.common as alc_common
from ros_gen import SystemLaunchGen
import alc_utils.alc_model_updater as model_updater
from alc_utils.setup_repo import RepoSetup
from pathlib import Path
import tarfile
import base64

# Setup a logger
logger = logging.getLogger('LaunchExpt')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)  # By default it logs to stderr..
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Useful macros
SECONDS_PER_MINUTE = 60.0
SLURM_GRACE_TIME_MIN = 5

# Directory for JINJA templates
_package_directory = os.path.dirname(__file__)
_template_directory = os.path.join(_package_directory, "templates")
alc_working_dir_env_var_name = "ALC_WORKING_DIR"
alc_meta_type_name  = "ALC"
exec_dir_name = ".exec"


#building of the sources can be part of the launch (./run.sh), before the execution runner is invoked.
#the alc_home can point to the repo_root or repo_home

#clone the repo in the plugin
#invoke the run.sh command passing the repo download folder to slurm as alc_home,
    #or pass both alc_home and repo_home to slurm and the execution runner sets the repo_home as alc_home. 
# execute


class LaunchExpt(PluginBase):

    campaign_count_key = "camp_count"
    command_for_srun_key = "command_for_srun"
    project_info_key = "project_info"
    project_name_key = "name"
    workflow_output_key = 'WF_Output_Path'

    bash_command = "bash"
    bash_script_name = "run.sh"

    def __init__(self, *args, **kwargs):
        self.slurm_params = {
            WebGMEKeys.webgme_port_key: kwargs.pop(WebGMEKeys.webgme_port_key, 8888)
        }
        super(LaunchExpt, self).__init__(*args, **kwargs)
        self.nodes = {}
        self.environments = {}
        self.missions = {}
        self.implementations = {}
        self.experimentSetup = {}
        self.campaign = {}

        self.paramValues = {}
        self.lecParamValues = {}
        self.modelValues = {}

        self.errorValues = []

        self.containerInfo = {}
        self.exptParamSetup = {}
        self.campaignParamSetup = {}
        self.lecParamSetup = {}
        self.lecCodeSetup = {}
        self.zipFileSetup = {}
        self.jobParams = {}
        self.allParams = {}

        self.outputExptParam = ''
        self.outputCampaignParam = ''
        self.outputLECParam = ''
        self.outputAllParams = ''

        self.config = self.get_current_config()
        self.setupJupyterNB = False
        self.generateROSLaunch = False
        self.execName = ''
        self.genTime = ''
        self.projectName = ''
        self.modelName = ''
        self.alldata = {}
        self.active_node_meta_type = ''
        self.zipFileContents = {}

        self.dtval = int(round(time.time() * 1000))
        self.dt = str(self.dtval)
        self.resultnode = ''
        self.resultnodepath = ''
        self.resultnodename = ''
        self.resultnode_info = {
            'TData':[],
            'VData':[],
            'EData':[],
            'LData':[],
            'PData':[],
            'name':'',
            'activity': '',
            'createdAt':'',
            'resultdir':''}
        self.top_script_path = ''
        self.campcount = 0

        self.evaldatasets = {}
        self.valdatasets = {}

        self.verification_model = {
            'Model': '',
            'InitSet': '',
            'LEC': [],
            'Params': [],
            'Transform': [],
            'ResultSet': [],
            'Ver_Flow': [],
            'Plant_Model': [],
            'Plant_Model_Content': [],
            'Configuration': [],
            'Configuration_Content': []
        }
        self.runVerificationSetup = 0
        self.runValidationSetup = 0
        self.runSystemIDSetup = 0
        self.runSLTrainingSetup = 0
        self.runAssuranceMonitorSetup = 0
        self.runEvaluationSetup = 0
        self.postprocesscriptFound = 0
        self.verification_filehashes = {}
        self.verification_code = ''
        self.sysid_code = ''
        self.run_init = '''
alc_wd = getenv('ALC_WORKING_DIR')
cur_dir = fullfile(alc_wd,'<<HOME_DIR>>')
setenv('temp_alc', cur_dir)
cd '/verivital/nnv/code/nnv'
startup_nnv
cur_dir = getenv('temp_alc')
cd(cur_dir)
'''

        # Initialize JINJA template engine
        self.template_loader = jinja2.FileSystemLoader(searchpath=_template_directory)
        self.template_env = jinja2.Environment(loader=self.template_loader)

        self.alc_node = None
        self.repo = ''
        self.branch = ''
        self.tag = ''
        self.repo_root = ''
        self.repo_home = ''
        self.ALC_SSH_PORT='22'
        self.ALC_SSH_HOST=socket.gethostbyname(socket.gethostname())

        self.target_launch_filename= ''
        self.target_launch_file = ''
        self.target_ros_master_ip= ''
        self.target_ros_master_port=''
        self.target_local_ros_master_port_mapping=''
        self.target_lec_deployment_key= {}
        self.target_repo_archive_path = ''
        self.target_lec_archive_path = ''

    def set_job_info(self):
        proj_info = self.project.get_project_info()
        pname = proj_info['_id']
        self.genTime = self.dt
        self.modelName = self.core.get_attribute(self.active_node, 'name')
        self.projectName = pname.replace('+', '_')
        self.jobParams = {
            'datetime': self.genTime,
            'model': self.modelName,
            'project': self.projectName
        }
        metanode = self.core.get_meta_type(self.active_node)
        metaname = self.core.get_attribute(metanode, 'name')
        self.jobParams['jobType'] = metaname.upper()

    # compiles all relevant parameters for any job
    # jobinfo - project details
    # expt info - collection of all parameters in the setup (without campaign)
    # campaign info - campaign parameters
    # lec info - lec related parameters
    # leccodeinfo - lec code
    # file info - file heirarchy in the zip file
    def build_all_params(self):
        self.allParams = {
            'jobInfo': self.jobParams,
            'exptInfo': self.exptParamSetup,
            'campaignInfo': self.campaignParamSetup,
            'lecInfo': self.lecParamSetup,
            'leccodeInfo': self.lecCodeSetup,
            'fileInfo': self.zipFileSetup,
            'containerInfo': self.containerInfo,
            'targetContainerName':self.target_launch_filename,
            'targetROSMasterIP':self.target_ros_master_ip,
            'targetROSMasterPort':self.target_ros_master_port,
            'localROSMaster': self.target_local_ros_master_port_mapping

        }
        self.outputAllParams = json.dumps(self.allParams)

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

    def get_plant_model(self):
        pass

    def add_implementation_parameters(self):
        keys = self.modelValues.keys()
        if 'Implementation' not in keys:
            return

        ikeys = self.modelValues['Implementation'].keys()
        for iname in ikeys:
            pkeys = self.modelValues['Implementation'][iname]['params'].keys()
            for pkey in pkeys:
                pval = self.modelValues['Implementation'][iname]['params'][pkey]
                self.add_params(pval)

    def add_expt_setup_parameters(self):
        keys = self.modelValues.keys()
        if 'ExperimentSetup' not in keys:
            return

        pkeys = self.modelValues['ExperimentSetup']['params'].keys()
        for pkey in pkeys:
            pval = self.modelValues['ExperimentSetup']['params'][pkey]
            self.add_params(pval)

    def add_environment_parameters(self):
        keys = self.modelValues.keys()
        if 'Environment' not in keys:
            return

        pkeys = self.modelValues['Environment']['params'].keys()
        for pkey in pkeys:
            pval = self.modelValues['Environment']['params'][pkey]
            self.add_params(pval)

    def add_mission_parameters(self):
        keys = self.modelValues.keys()
        if 'Mission' not in keys:
            return

        pkeys = self.modelValues['Mission']['params'].keys()
        for pkey in pkeys:
            pval = self.modelValues['Mission']['params'][pkey]
            self.add_params(pval)

    def add_campaign_parameters(self):
        keys = self.modelValues.keys()
        if 'ParamSweep' not in keys:
            return

        pkeys = self.modelValues['ParamSweep']['params'].keys()
        for pkey in pkeys:
            pval = self.modelValues['ParamSweep']['params'][pkey]
            self.add_campaign_params(pval)

    def add_params(self, pvalues):
        keys = pvalues.keys()
        for pname in keys:
            pvalue = pvalues[pname]
            if pname.lower() == "slurm":
                self.slurm_params.update(pvalue)
            elif pname.lower() == "generateroslaunch":
                self.generateROSLaunch = pvalue
            else:
                self.exptParamSetup[pname] = pvalue

    def add_campaign_params(self, pvalues):
        keys = pvalues.keys()
        for pname in keys:
            pvalue = pvalues[pname]
            self.campaignParamSetup[pname] = pvalue
    
    def get_campaign_count(self):
        keys = self.campaignParamSetup.keys()
        ret = 1 
        for pname in keys:
            pvaluelen = len(self.campaignParamSetup[pname])
            ret = ret*pvaluelen
        if ret == 0:
            ret = 1
        return ret

    def update_parameters_from_config(self):
        if self.config == '':
            return

        keys = self.config.keys()
        if 'ParamUpdates' not in keys:
            return

        paramupdates = self.config['ParamUpdates']

        keys = paramupdates.keys()
        okeys = self.campaignParamSetup.keys()
        updatedparams = []
        for k in keys:
            if k not in okeys:
                continue
            if isinstance(paramupdates[k], list):
                self.campaignParamSetup[k] = paramupdates[k]
                updatedparams.append(k)

        okeys = self.exptParamSetup.keys()
        for k in keys:
            if k in updatedparams:
                continue
            if k not in okeys:
                continue
            self.exptParamSetup[k] = paramupdates[k]
            updatedparams.append(k)

    def build_param_dictionary(self):
        self.add_implementation_parameters()
        self.add_expt_setup_parameters()
        self.add_environment_parameters()
        self.add_mission_parameters()
        self.add_campaign_parameters()
        self.update_parameters_from_config()
        self.outputExptParam = json.dumps(self.exptParamSetup)
        self.outputCampaignParam = json.dumps(self.campaignParamSetup)

    def get_lec_data(self, node_id):

        node = self.core.load_by_path(self.root_node, node_id)
        data = self.core.get_attribute(node, 'data')

        datadict = ''
        datastr = ''
        if data:
            datastr = self.get_file(data)
        dataval = self.core.get_attribute(node, 'datainfo')
        if dataval:
            # dataval = re.sub("null", 'NaN',dataval)
            try:
                datadict = json.loads(dataval, strict=False)
                self.alldata[node_id] = 'LData'
                #self.core.add_member(self.resultnode, 'LData', node)
                self.resultnode_info['LData'].append(node_id)
            except Exception as e:
                datadict = ''
                logger.info('problem in parsing lec1 "{0}"'.format(e))
        elif datastr:
            logger.info(' datastr = ' + datastr)
            n = len(datastr)
            datastr1 = datastr[2: n - 6]
            try:
                # datastr1 = re.replace("null", 'NaN',dataval)
                datadict = json.loads(datastr1, strict=False)
                self.alldata[node_id] = 'LData'
            except Exception as e:
                datadict = ''
                logger.info('problem in parsing lec2 "{0}"'.format(e))
        return datadict

    def build_lec_param(self):
        leckeys = self.lecParamValues.keys()
        self.lecParamSetup = {}
        for key in leckeys:
            self.lecParamSetup[key] = self.get_lec_data(self.lecParamValues[key])

        self.outputLECParam = json.dumps(self.lecParamSetup)

    def check_active_node_meta(self):
        meta_types = [self.META["ALCMeta.ExperimentSetup"], self.META["ALCMeta.SLTrainingSetUp"],
                      self.META['ALCMeta.AssuranceMonitorSetup'], self.META["ALCMeta.EvaluationSetup"],
                      self.META["ALCMeta.RLTrainingSetup"], self.META["ALCMeta.VerificationSetup"],
                      self.META["ALCMeta.ValidationSetup"], self.META["ALCMeta.SystemIDSetup"]]
        ret = False
        if self.active_node_meta_type in meta_types:
            ret = True

        if self.active_node_meta_type == self.META['ALCMeta.RLTrainingSetup']:
            self.exptParamSetup['runRLTrainingSetup'] = 1
        elif self.active_node_meta_type == self.META['ALCMeta.VerificationSetup']:
            self.exptParamSetup['runVerificationSetup'] = 1
            self.setupJupyterNB = 1
            self.runVerificationSetup = 1
        elif self.active_node_meta_type == self.META['ALCMeta.ValidationSetup']:
            self.exptParamSetup['runValidationSetup'] = 1
            self.setupJupyterNB = 1
            self.runValidationSetup = 1
        elif self.active_node_meta_type == self.META['ALCMeta.SystemIDSetup']:
            self.exptParamSetup['runSystemIDSetup'] = 1
            self.setupJupyterNB = 1
            self.runSystemIDSetup = 1
        elif self.active_node_meta_type == self.META['ALCMeta.SLTrainingSetUp']:
            self.exptParamSetup['runSLTrainingSetup'] = 1
            self.runSLTrainingSetup = 1
        elif self.active_node_meta_type == self.META['ALCMeta.AssuranceMonitorSetup']:
            self.exptParamSetup['runAssuranceMonitoring'] = 1
            self.runAssuranceMonitorSetup = 1
        elif self.active_node_meta_type == self.META['ALCMeta.EvaluationSetup']:
            self.exptParamSetup['runEvaluationSetup'] = 1
            self.runEvaluationSetup = 1
            self.setupJupyterNB = 1
            logger.info("**************************************************")
            logger.info("**************************************************")
            logger.info("**************************************************")
            logger.info("**************************************************")
            logger.info("**************************************************")

        return ret

    def check_param_parent(self, parentmeta):
        valid_meta_types = [self.META["ALCMeta.Environment"], self.META["ALCMeta.Mission"],
                            self.META["ALCMeta.ExperimentSetup"], self.META["ALCMeta.ParamSweep"],
                            self.META["ALCMeta.Block"], self.META["ALCMeta.SLTrainingSetUp"],
                            self.META['ALCMeta.AssuranceMonitorSetup'], self.META["ALCMeta.EvaluationSetup"]]
        if parentmeta in valid_meta_types:
            return True
        return False

    def build_verification_model_info(self):
        pass

    def build_validation_model_info(self):
        pass

    def build_system_id_info(self):
        pass

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
        

    def get_utils(self):
        ret = {}
        rc = self.core.load_children(self.root_node)
        alc_root = ''
        alc_workspace = ''
        alc_utilities = ''
        # get alc_root
        for c in rc:
            cmeta = self.core.get_meta_type(c)
            if cmeta == self.META["ALCMeta.ALC_ROOT"]:
                alc_root = c
                break
        if not alc_root:
            return ret
        # get alc_workspace
        rc = self.core.load_children(alc_root)
        if len(rc) == 1:
            cmeta = self.core.get_meta_type(rc[0])
            if cmeta == self.META["ALCMeta.Workspace"]:
                alc_workspace = rc[0]

        if not alc_workspace:
            return ret
        rc = self.core.load_children(alc_workspace)
        for c in rc:
            cmeta = self.core.get_meta_type(c)
            if cmeta == self.META["ALCMeta.DEEPFORGE.Utilities"]:
                alc_utilities = c
                break
        if not alc_utilities:
            return ret

        rc = self.core.load_children(alc_utilities)
        for c in rc:
            cmeta = self.core.get_meta_type(c)
            if cmeta == self.META["ALCMeta.DEEPFORGE.pipeline.Code"]:
                name = self.core.get_attribute(c, 'name')
                code = self.core.get_attribute(c, 'code')
                ret[name] = code

        return ret

    def generate_scripts(self, result_dir, relative_result_dir):
        # Fill out top-level launch template
        run_template = self.template_env.get_template("run_script.sh")
        from_plugin = run_template.render(
            relative_result_dir=relative_result_dir,
        )

        # Write "from_plugin" to file
        self.top_script_path = os.path.join(result_dir, 'run.sh')
        with open(self.top_script_path, 'w') as f:
            f.write(from_plugin)
        st = os.stat(self.top_script_path)
        os.chmod(self.top_script_path, st.st_mode | stat.S_IEXEC)

    def generate_main_file(self, param_file_path, result_dir, camp_count):
        # Fill out top-level launch template
        result_file_path = os.path.join(result_dir, 'result_metadata.json')
        main_template = self.template_env.get_template("python_main.py")
        code = main_template.render(
            param_file_path=param_file_path,
            result_file_path=result_file_path,
            setup_jupyter_nb=self.setupJupyterNB,
            camp_count=camp_count
        )

        # Write python main code to path
        main_file_path = os.path.join(result_dir, 'main.py')
        with open(main_file_path, "w") as f:
            f.write(code)

    def generate_files(self, campcount):
        alc_wkdir = os.getenv('ALC_WORK', '')
        logger.info('ALC_WORKING_DIR = ' + alc_wkdir)
        # x= os.environ
        # x1 = json.dumps(x,indent=4)
        # logger.info('ALL Environment Variables  = '+str(x1))
        if not alc_wkdir:
            raise RuntimeError('Environment variable ALC_WORK is unknown or not set')
        if not os.path.isdir(alc_wkdir):
            raise RuntimeError('ALC_WORK: ' + alc_wkdir + ' does not exist')
        jupyter_dir = os.path.join(alc_wkdir, 'jupyter')
        logger.info('jupyter_dir = ' + jupyter_dir)
        if not os.path.isdir(jupyter_dir):
            raise RuntimeError('jupyter directory : ' + jupyter_dir + ' does not exist in ALC_WORK')

        model_code_dir = os.path.join(jupyter_dir, self.jobParams['project'], self.jobParams['model'],
                                      self.jobParams['datetime'], 'ModelData')
        if not os.path.isdir(model_code_dir):
            os.makedirs(model_code_dir)

        logger.info('model_code_dir = ' + model_code_dir)
        keys = self.zipFileContents.keys()
        for k in keys:
            file_path = os.path.join(model_code_dir, k)
            logger.info('file_path = ' + file_path)
            dir_path = os.path.dirname(file_path)
            logger.info('dir_path = ' + dir_path)
            if not os.path.isdir(dir_path):
                os.makedirs(dir_path)
                logger.info('created dir_path = ' + dir_path)
            print(k)
            print(self.zipFileContents[k])

            f = open(file_path, "w")
            f.write(self.zipFileContents[k])
            f.close()

        param_file_path = os.path.join(model_code_dir, 'params.json')
        f = open(param_file_path, "w")
        f.write(self.outputAllParams)
        f.close()

        utils_code = self.get_utils()
        utils_code_dir = os.path.join(jupyter_dir, self.jobParams['project'], self.jobParams['model'],
                                      self.jobParams['datetime'], 'utils')
        if not os.path.isdir(utils_code_dir):
            os.makedirs(utils_code_dir)

        file_path = os.path.join(utils_code_dir, '__init__.py')
        open(file_path, "a").close()

        keys = utils_code.keys()
        for k in keys:
            fname = k
            if not fname.endswith('.py'):
                fname += '.py'

            file_path = os.path.join(utils_code_dir, fname)
            f = open(file_path, "w")
            f.write(utils_code[k])
            f.close()

        camp_dir = 'config-'+str(campcount)

        dir_base = os.path.join(
            self.jobParams['project'],
            self.jobParams['model'],
            self.jobParams['datetime']
        )

        result_dir_base = os.path.join(dir_base, self.jobParams['datetime'], camp_dir)
        relative_result_dir = os.path.join('jupyter',result_dir_base)
        result_dir = os.path.join(jupyter_dir, result_dir_base)
        
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir)

        self.resultnode_info['resultdir']=result_dir
        if (self.setupJupyterNB):
            url_dir = os.path.join(dir_base,self.jobParams['datetime'],'main.ipynb')
            if (self.runEvaluationSetup):
                url_dir = os.path.join(dir_base,'main.ipynb')
            self.resultnode_info['datainfo'] = json.dumps({'url':url_dir})

        result_node = self.get_result_node(campcount)
        self.generate_main_file(param_file_path, result_dir, campcount)
        if not self.setupJupyterNB:
            self.generate_scripts(result_dir, relative_result_dir)
        return result_dir, result_node
    
    # def get_result_node(self, current_count):
    #     if self.campcount == 1 or self.setupJupyterNB:
    #         return self.resultnode
        
    #     result_parent = self.core.get_parent(self.resultnode)
    #     if current_count == 0:
    #         cn = self.resultnode
    #     else:
    #         cn = self.core.copy_node(self.resultnode, result_parent)
    #     result_name = self.core.get_attribute(self.resultnode, 'name')
    #     createtime = self.core.get_attribute(self.resultnode, 'createdAt')
    #     result_name_idx = result_name.rindex('-')
    #     if result_name_idx > -1:
    #         result_name = result_name[:result_name_idx]
    #     self.core.set_attribute(cn, 'name', result_name + '-'+str(current_count))
    #     self.core.set_attribute(cn, 'createdAt', createtime + current_count)
    #     return cn

    def get_result_node(self, current_count):
        if self.campcount == 1 or self.setupJupyterNB:
            return self.get_resultnode_from_router(0)
        
        return self.get_resultnode_from_router(current_count)
    
    def get_resultnode_from_router(self, current_count):
        execution_name = self.resultnode_info.get("name", 'data')
        if (current_count > 0):
            execution_name = execution_name + '-'+str(current_count)

        project_info   = self.project.get_project_info()
        project_owner  = project_info[WebGMEKeys.project_owner_key]
        project_name   = project_info[WebGMEKeys.project_name_key]
        active_node_path = self.core.get_path(self.active_node)
        
        modifications = {
            "createdAt": self.resultnode_info["createdAt"]+current_count,
            "activity" : self.resultnode_info["activity"],
            "resultDir" : self.resultnode_info["resultdir"],
            "jobstatus": "Submitted"
        }
        if (self.setupJupyterNB):
            modifications["jobstatus"] = "Finished"
            modifications["datainfo"]  = self.resultnode_info["datainfo"]

        sets = {}
        set_entries = ['TData','VData','EData','LData','PData']
        for s in set_entries:
            s_vals = self.resultnode_info.get(s,[])
            if not s_vals:
                continue
            sets[s]=s_vals
        
        return model_updater.create_data_node(logger, project_owner, project_name, active_node_path, execution_name, modifications,sets)

        

    def execute_experiment(self):
        # execute the experiment
        os.system(self.top_script_path)
        # subprocess.Popen([self.top_script_path])
        # DETACHED_PROCESS = 0x00000008
        # proc = subprocess.Popen(
        #     [self.top_script_path]),
        #     shell=True,
        #     stdin=None,
        #     stdout=None,
        #     stderr=None,
        #     close_fds=True,
        #     creationflags = DETACHED_PROCESS
        # )
        # logger.info('launched proc '+str(proc))

    def is_verification_activity(self, activity_node):
        # logger.info('6')
        ver_types = [self.META["ALCMeta.VerificationSetup"], self.META["ALCMeta.ValidationSetup"],
                     self.META["ALCMeta.SystemIDSetup"]]
        if self.core.get_meta_type(activity_node) in ver_types:
            # logger.info('7')
            return True
        # logger.info('8')
        return False

    def create_jupyter_notebook(self, eval=False):
        alc_wkdir = os.getenv('ALC_WORK', '')
        logger.info('ALC_WORKING_DIR = ' + alc_wkdir)
        if not alc_wkdir:
            raise RuntimeError('Environment variable ALC_WORK is unknown or not set')
        if not os.path.isdir(alc_wkdir):
            raise RuntimeError('ALC_WORK: ' + alc_wkdir + ' does not exist')
        jupyter_dir = os.path.join(alc_wkdir, 'jupyter')
        
        base_folder = os.path.join(
            jupyter_dir,
            self.jobParams['project'],
            self.jobParams['model'],
            self.jobParams['datetime']
        )

        if (not eval):
            result_folder = os.path.join(base_folder, self.jobParams['datetime'])
            main_template = self.template_env.get_template("main.ipynb")
            code = main_template.render(
                result_folder=result_folder,
                camp_count=self.campcount,
                post_process=self.postprocesscriptFound
            )
        else:
            result_folder = base_folder

            file_folder = os.path.join(base_folder, 'ModelData')
            
            filename = os.path.join(file_folder, 'eval_data.json')
            with open(filename, 'w') as f:
                json.dump(self.evaldatasets, f)

            filename = os.path.join(file_folder, 'val_data.json')
            with open(filename, 'w') as f:
                json.dump(self.valdatasets, f)
        
            filename = os.path.join(file_folder, 'lec_data.json')
            with open(filename, 'w') as f:
                json.dump(self.lecParamSetup, f)
            
            main_template = self.template_env.get_template("evaluation.ipynb")
            code = main_template.render(
                model_data_folder=file_folder,
                eval_data=self.evaldatasets,
                val_data=self.valdatasets,
                lec_data=self.lecParamSetup,
                post_process=self.postprocesscriptFound
            )

        # Write python main code to path
        main_file_path = os.path.join(result_folder, 'main.ipynb')
        with open(main_file_path, "w") as f:
            f.write(code)

        project_info     = self.project.get_project_info()
        project_owner    = project_info[WebGMEKeys.project_owner_key]
        project_name     = project_info[WebGMEKeys.project_name_key]
        active_node_path = self.core.get_path(self.active_node)
        execution_name   = self.resultnode_info['name']
       
        prefix = 'ipython/notebooks/'
        activity_node = self.active_node
        if self.is_verification_activity(activity_node):
            prefix = 'matlab/notebooks/'

        relative_result_path = os.path.join(
            self.jobParams['project'],
            self.jobParams['model'],
            self.jobParams['datetime'])
        if (not self.runEvaluationSetup):
            relative_result_path = os.path.join(relative_result_path, self.jobParams['datetime'])
        relative_result_path = os.path.join(relative_result_path,'main.ipynb')

        url_str = prefix + relative_result_path
        pos = {'x': 100,'y': 100}
        model_updater.create_jupyter_node(
                        logger,
                        project_owner,
                        project_name,
                        active_node_path,
                        execution_name,
                        url_str,
                        pos)



    def setup_and_build_repo (self, build_repo):
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

        if (self.repo == ''):
            logger.info('in setup_and_build_repo - no repo')
            return

        logger.info('in setup_and_build_repo repo == '+ self.repo)

        alc_working_dir_name = os.environ.get(alc_working_dir_env_var_name, None)
        if alc_working_dir_name:
            logger.info('in setup_and_build_repo repo alc_working_dir ' +alc_working_dir_name)
            self.repo_root = Path(alc_working_dir_name, exec_dir_name, self.repo)
            logger.info('in setup_and_build_repo repo self.repo_root ' +str(self.repo_root))
            if (build_repo):
                r =  RepoSetup()
                logger.info('in setup_and_build_repo repo trying to clone')
                r.clone_repo(self.repo_root, self.repo, self.branch, self.tag, logger)
                logger.info('in setup_and_build_repo repo trying to build')
                r.build_repo(str(self.repo_root), self.repo, logger)
                self.repo_home = os.path.join(str(self.repo_root), self.repo)
                logger.info('after building repo -> self.repo_home = '+self.repo_home)
            else:
                self.repo_home = os.path.join(str(self.repo_root), self.repo)
                if (not os.path.exists(self.repo_home)):
                    self.repo_home = ''
                logger.info('without building repo self.repo_home = '+self.repo_home)

        logger.info('finished setup_and_build_repo repo self.repo_home = '+self.repo_home)


    def main(self):
        
        try:

            logger.info('1 ')

            self.active_node_meta_type = self.core.get_meta_type(self.active_node)
            self.setupJupyterNB = self.config['setupJupyterNB']
            activenodename = self.core.get_attribute(self.active_node, 'name')
            self.lecCodeSetup = {}
            if self.config['name'] == '':
                self.config['name'] = activenodename

            self.execName = self.config['name'] + '-' + self.dt
            self.resultnode = ''
            self.resultnodepath = ''
            self.resultnodename = 'result-' + self.execName
            self.resultnode_info['name'] = self.resultnodename
            self.resultnode_info['createdAt'] = self.dtval
            self.resultnode_info['activity'] = self.core.get_fully_qualified_name(self.active_node)

            


            rlcode = {}
            scenariosets = []
            plantdatarefs = []
            traindatarefs = []
            evaldatarefs = []
            valdatarefs = []
            evaldatasets = {}
            valdatasets = {}

            if not self.check_active_node_meta():
                raise RuntimeError(
                    "Execute Expt can be run on ExperimentSetup or TrainingSetup or AssuranceMonitorSetup or "
                    "EvaluationSetup or VerificationSetup"
                )

            # logger.info('######## DEBUG LOCATION 2 ########')
            # set job informtion
            self.set_job_info()

            node_list = self.core.load_sub_tree(self.active_node)
            nodepath = self.core.get_path(self.active_node)
            self.nodes[nodepath] = self.active_node

            if self.active_node_meta_type == self.META['ALCMeta.ExperimentSetup']:
                self.experimentSetup[nodepath] = self.active_node

            if self.active_node_meta_type == self.META['ALCMeta.Campaign']:
                self.campaign[nodepath] = self.active_node
                fullname = self.core.get_fully_qualified_name(self.active_node)
                self.modelValues['Campaign'] = {}
                self.modelValues['Campaign']['name'] = fullname
                self.modelValues['Campaign']['params'] = {}

            # Loop through all elements in the subtree
            # get the relevant code parameters
            # initialize the dictionary elements for parameters, campaign parameters
            # logger.info(str(len(node_list)))
            for i in range(0, len(node_list)):
                logger.info(str(i))
                if not node_list[i]:
                    continue
                nodepath = self.core.get_path(node_list[i])
                logger.info('nodepath ' + nodepath)
                if nodepath == '/H/E/3/n/4/gcI':
                    continue

                self.nodes[nodepath] = node_list[i]
                name = self.core.get_attribute(node_list[i], 'name')
                logger.info('nodename ' + name)
                if not name or name == 'undefined':
                    continue

                fullname = self.core.get_fully_qualified_name(node_list[i])
                metanode = self.core.get_meta_type(node_list[i])
                metaname = self.core.get_attribute(metanode, 'name')
                logger.info('metaname ' + metaname)
                logger.info('fullname ' + fullname)

                # create a pipeline data object for the experiment
                # this temporary node will hold the zip asset that will be created
                #if metanode == self.META['ALCMeta.Result']:
                    # logger.info('2')
                    #rparent = self.core.get_parent(node_list[i])
                    #if rparent == self.active_node:
                        #self.resultnode = self.core.create_child(
                        #    node_list[i], self.META["ALCMeta.DEEPFORGE.pipeline.Data"]
                        #)
                        #self.core.set_attribute(self.resultnode, 'name', self.resultnodename)
                        #self.core.set_attribute(self.resultnode, 'createdAt', self.dtval)
                        #self.core.set_attribute(
                        #    self.resultnode, 'activity', self.core.get_fully_qualified_name(self.active_node)
                        #)
                        #self.resultnodepath = self.core.get_path(self.resultnode)
                    #continue

                # simulation parameters for - experiments, verification, RLTrainingSetup
                # looks like verification will not need simulation
                if (
                        metanode == self.META["ALCMeta.ExperimentSetup"] or
                        metanode == self.META["ALCMeta.VerificationSetup"] or
                        metanode == self.META["ALCMeta.RLTrainingSetup"] or
                        metanode == self.META["ALCMeta.VerificationSetup"] or
                        metanode == self.META["ALCMeta.ValidationSetup"] or
                        metanode == self.META["ALCMeta.SystemIDSetup"] or
                        metanode == self.META["ALCMeta.SLTrainingSetUp"] or
                        metanode == self.META["ALCMeta.AssuranceMonitorSetup"] or
                        metanode == self.META["ALCMeta.EvaluationSetup"]
                ):
                    logger.info('expt setup')
                    self.experimentSetup[nodepath] = node_list[i]
                    self.modelValues['ExperimentSetup'] = {}
                    self.modelValues['ExperimentSetup']['name'] = fullname
                    self.modelValues['ExperimentSetup']['params'] = {}

                # environment
                if metanode == self.META["ALCMeta.Environment"]:
                    logger.info('env')
                    self.environments[nodepath] = node_list[i]
                    self.modelValues['Environment'] = {}
                    self.modelValues['Environment']['name'] = fullname
                    self.modelValues['Environment']['params'] = {}
                    continue

                # mission
                if metanode == self.META["ALCMeta.Mission"]:
                    logger.info('Mission')
                    self.missions[nodepath] = node_list[i]
                    self.modelValues['Mission'] = {}
                    self.modelValues['Mission']['name'] = fullname
                    self.modelValues['Mission']['params'] = {}
                    logger.info('Mission completed')
                    continue

                # scenario set
                if metanode == self.META["ALCMeta.ScenarioSet"]:
                    setmembers = self.core.get_member_paths(node_list[i], 'Data')
                    for s in setmembers:
                        scenariosets.append(s)
                    continue

                # FIXME: the variable rlcode is being abused. it is no longer
                #  specific to RL, but is being used for all contexts.
                # RLAgent for RL Learning
                if metanode == self.META["ALCMeta.RLAgent"]:
                    rlcode['RLModel/RLAgent.py'] = self.core.get_attribute(node_list[i], 'Code')
                    rlcode['RLModel/RLEnvironment.py'] = self.core.get_attribute(node_list[i], 'RLEnvironment')

                    self.add_zip_content('RLModel/RLAgent.py', rlcode['RLModel/RLAgent.py'])
                    self.add_zip_content('RLModel/RLEnvironment.py', rlcode['RLModel/RLEnvironment.py'])
                    self.add_zip_file_info('RLModel', ['RLAgent.py', 'RLEnvironment.py'])
                    continue

                # Any code block (part of RLAgent or others)
                if metanode == self.META["ALCMeta.Code"]:
                    logger.info('********GOT CODE BLOCK a************')
                    # Determine what to do with this code block based on the META-type of its parent block
                    cparent = self.core.get_parent(node_list[i])
                    cparentmeta = self.core.get_meta_type(cparent)
                    # RL Learning
                    logger.info('********GOT CODE BLOCK b************')
                    if cparentmeta == self.META["ALCMeta.RLAgent"]:
                        filename = self.core.get_attribute(node_list[i], 'FileName')
                        definition = self.core.get_attribute(node_list[i], 'code')
                        fname = 'RLModel/' + filename
                        rlcode[fname] = definition
                        self.add_zip_content(fname, definition)
                        self.add_zip_file_info('RLModel', [filename])

                    # verification
                    logger.info('********GOT CODE BLOCK c************')
                    if cparentmeta == self.META["ALCMeta.Specification"]:
                        filename = self.core.get_attribute(node_list[i], 'FileName')
                        definition = self.core.get_attribute(node_list[i], 'code')
                        fname = 'Specification/' + filename
                        rlcode[fname] = definition
                        self.add_zip_content(fname, definition)
                        self.add_zip_file_info('Specification', [filename])

                    # verification
                    logger.info('********GOT CODE BLOCK d************')
                    try:
                        if cparentmeta == self.META["ALCMeta.Model"]:
                            filename = self.core.get_attribute(node_list[i], 'FileName')
                            definition = self.core.get_attribute(node_list[i], 'code')
                            fname = 'Model/' + filename
                            rlcode[fname] = definition
                            self.add_zip_content(fname, definition)
                            self.add_zip_file_info('Model', [filename])
                    except:
                        logger.info('********GOT CODE BLOCK d1************')


                    # all cases - experiment, rltraining, verification
                    logger.info('********GOT CODE BLOCK e************')
                    if cparentmeta == self.META["ALCMeta.PostProcess"]:
                        logger.info('********GOT CODE BLOCK ************')
                        filename = self.core.get_attribute(node_list[i], 'FileName')
                        definition = self.core.get_attribute(node_list[i], 'code')
                        fname = 'PostProcess/' + filename
                        rlcode[fname] = definition
                        self.add_zip_content(fname, definition)
                        self.add_zip_file_info('PostProcess', [filename])
                        self.postprocesscriptFound = 1
                        print('content '+definition)
                        logger.info('********GOT CODE BLOCK ************')
                    
                    if cparentmeta == self.META["ALCMeta.ROSInfo"]:
                        continue

                    continue

                # verification specification
                if metanode == self.META["ALCMeta.Specification"]:
                    cparent = self.core.get_parent(node_list[i])
                    cparentmeta = self.core.get_meta_type(cparent)
                    if cparentmeta == self.META["ALCMeta.VerificationSetup"]:
                        definition = self.core.get_attribute(node_list[i], 'Definition')
                        fname = 'Specification/Specification.cfg'
                        rlcode[fname] = definition
                        self.add_zip_content(fname, definition)
                        self.add_zip_file_info('Specification', ['Specification.cfg'])
                    continue

                # systemid
                if self.runSystemIDSetup == 1:
                    if metanode == self.META["ALCMeta.Params"]:
                        self.verification_model['Params'].append(node_list[i])

                # verification_model
                if self.runVerificationSetup == 1 or self.runValidationSetup == 1:
                    if (
                            metanode == self.META["ALCMeta.Verification_Model"] or
                            metanode == self.META["ALCMeta.Validation_Model"]
                    ):
                        self.verification_model['Model'] = node_list[i]

                    if metanode == self.META["ALCMeta.Transform"]:
                        self.verification_model['Transform'].append(node_list[i])

                    if metanode == self.META["ALCMeta.InitSet"]:
                        self.verification_model['InitSet'] = node_list[i]

                    if metanode == self.META["ALCMeta.LEC_Model"]:
                        self.verification_model['LEC'].append(node_list[i])
                        lecdata_ver = self.core.get_pointer_path(node_list[i], 'ModelDataLink')

                        if not lecdata_ver:
                            raise RuntimeError("Please set the LEC reference in LEC Model -   (%s)" % name)

                    if metanode == self.META["ALCMeta.Params"]:
                        self.verification_model['Params'].append(node_list[i])

                    if metanode == self.META["ALCMeta.Ver_Flow"]:
                        self.verification_model['Ver_Flow'].append(node_list[i])

                    if metanode == self.META["ALCMeta.Ver_Result"]:
                        self.verification_model['ResultSet'].append(node_list[i])

                    if metanode == self.META["ALCMeta.SpaceEx.Configuration"]:
                        self.verification_model['Configuration'].append(node_list[i])
                        self.verification_model['Configuration_Content'].append(
                            self.core.get_attribute(node_list[i], 'content'))

                    cparent = None
                    if (
                            metanode == self.META["ALCMeta.SpaceEx.BaseComponent"] or
                            metanode == self.META["ALCMeta.SpaceEx.NetworkComponent"]
                    ):
                        cparent = self.core.get_parent(node_list[i])
                        cparentmeta = self.core.get_meta_type(cparent)
                        if cparentmeta == self.META["ALCMeta.PlantModel"]:
                            if cparent not in self.verification_model['Plant_Model']:
                                self.verification_model['Plant_Model'].append(cparent)

                    if metanode == self.META["ALCMeta.PlantModel"]:
                        sys_id = self.core.get_pointer_path(node_list[i], 'SystemID')
                        if sys_id:
                            if cparent is not None:
                                self.verification_model['Plant_Model'].append(cparent)
                            plantdatarefs.append(sys_id)

                if metanode == self.META["ALCMeta.EvalData"]:
                    setmembers = self.core.get_member_paths(node_list[i], 'Data')
                    evaldatasets[name] = []
                    for s in setmembers:
                        evaldatarefs.append(s)
                        evaldatasets[name].append(s)
                        self.alldata[s] = 'EData'
                    continue

                if metanode == self.META["ALCMeta.TrainingData"]:
                    setmembers = self.core.get_member_paths(node_list[i], 'Data')
                    for s in setmembers:
                        traindatarefs.append(s)
                        self.alldata[s] = 'TData'
                    continue

                if metanode == self.META["ALCMeta.ValidationData"]:
                    setmembers = self.core.get_member_paths(node_list[i], 'Data')
                    valdatasets[name] = []
                    for s in setmembers:
                        valdatarefs.append(s)
                        valdatasets[name].append(s)
                        self.alldata[s] = 'VData'
                    continue

                if (
                        metanode == self.META["ALCMeta.LEC_Model"] and
                        (
                                self.runSLTrainingSetup == 1 or
                                self.runAssuranceMonitorSetup == 1
                        )
                ):
                    lecmodel = node_list[i]
                    leccode = self.core.get_attribute(lecmodel, "Definition")
                    lecdataformatter = self.core.get_attribute(lecmodel, "DataFormatter")
                    lecdataloader = self.core.get_attribute(lecmodel, "Dataset")
                    lectrainingcode = self.core.get_attribute(lecmodel, "TrainingSetup")
                    lecamcode = self.core.get_attribute(lecmodel, 'AMDefinition')
                    self.lecCodeSetup['leccode'] = leccode
                    self.lecCodeSetup['lecdataformatter'] = lecdataformatter
                    self.lecCodeSetup['lecdataloader'] = lecdataloader
                    self.lecCodeSetup['lectrainingsetup'] = lectrainingcode
                    self.lecCodeSetup['amdefinition'] = lecamcode

                    fnames = []

                    if leccode:
                        self.add_zip_content('LECModel/LECModel.py', leccode)
                        fnames.append('LECModel.py')

                    if lecdataformatter:
                        self.add_zip_content('LECModel/data_formatter.py', lecdataformatter)
                        fnames.append('data_formatter.py')

                    if lecdataloader and lecdataloader != '':
                        self.add_zip_content('LECModel/data_loader.py', lecdataloader)
                        fnames.append('data_loader.py')

                    if lectrainingcode:
                        self.add_zip_content('LECModel/training_code.py', lectrainingcode)
                        fnames.append('training_code.py')
                    
                    if lecamcode:
                        self.add_zip_content('LECModel/am_net.py', lecamcode)
                        fnames.append('am_net.py')

                    self.add_zip_file_info('LECModel', fnames)

                    lecdata = self.core.get_pointer_path(node_list[i], 'ModelDataLink')
                    if lecdata:
                        self.lecParamValues["TrainedLECModel"] = lecdata

                    if self.runAssuranceMonitorSetup and (not lecdata):
                        raise RuntimeError("LEC Model reference not set. Please check.")
                        # self.result.setSuccess(false)
                        # self.result.setError('LEC Model reference not set. Please check.')
                        # callback(null, self.result)

                    if leccode == '':
                        raise RuntimeError("LEC Model Definition is not set")

                    if lecdataformatter == '':
                        raise RuntimeError("LEC Model  dataformatter is not set")

                    continue

                # PostProcess script
                if metanode == self.META["ALCMeta.PostProcess"]:
                    rlcode['PostProcess/PostProcess.py'] = self.core.get_attribute(node_list[i], 'Code')
                    self.add_zip_content('PostProcess/PostProcess.py', rlcode['PostProcess/PostProcess.py'])
                    self.add_zip_file_info('PostProcess', ['PostProcess.py'])
                    self.postprocesscriptFound = 1
                    continue

                # Setup the dictionary elements for active implementation blocks
                if metanode == self.META["ALCMeta.Block"]:
                    isimpl = self.core.get_attribute(node_list[i], 'IsImplementation')
                    if not isimpl:
                        continue
                    # self.logger.info('impl')
                    isactive = self.core.get_attribute(node_list[i], 'IsActive')
                    if not isactive:
                        continue

                    self.implementations[nodepath] = node_list[i]
                    keym = self.modelValues.keys()
                    if 'Implementation' not in keym:
                        self.modelValues['Implementation'] = {}
                    self.modelValues['Implementation'][nodepath] = {}
                    self.modelValues['Implementation'][nodepath]['params'] = {}
                    self.modelValues['Implementation'][nodepath]['lecparams'] = {}
                    continue

                # Setup the dictionary elements for campaign and get the campaign code
                if metanode == self.META["ALCMeta.Campaign"]:
                    rlcode['Campaign/CampaignSetup.py'] = self.core.get_attribute(node_list[i], 'Code')
                    self.add_zip_content('Campaign/CampaignSetup.py', rlcode['Campaign/CampaignSetup.py'])
                    self.add_zip_file_info('Campaign', ['CampaignSetup.py'])
                    pvalue = self.core.get_attribute(node_list[i], 'Definition')
                    if pvalue != '':
                        try:
                            jsonval = json.loads(pvalue, strict=False)
                            self.add_campaign_params(jsonval)
                        except Exception:
                            pfname = self.core.get_fully_qualified_name(node_list[i])
                            estr = 'Unable to parse JSON input for Campaign parameter ' + pfname + ' value = ' + pvalue
                            self.errorValues.append(estr)
                            logger.error(estr)
                            logger.error(pvalue)

            logger.info('DEBUG: End of first loop through model elements')

            # Second loop through model elements
            for i in range(0, len(node_list)):
                if not node_list[i]:
                    continue

                nodepath = self.core.get_path(node_list[i])
                name = self.core.get_attribute(node_list[i], 'name')
                # fullname = self.core.get_fully_qualified_name(node_list[i])
                metanode = self.core.get_meta_type(node_list[i])
                # metaname = self.core.get_attribute(metanode, 'name')

                # Look at param elements
                if metanode == self.META["ALCMeta.Params"]:
                    parent = self.core.get_parent(node_list[i])
                    pmetanode = self.core.get_meta_type(parent)
                    pmetaname = self.core.get_attribute(pmetanode, 'name')
                    pnodepath = self.core.get_path(parent)
                    # look at param elements in the contexts listed  below
                    if self.check_param_parent(pmetanode):

                        # consider only active implementation blocks
                        if pmetanode == self.META["ALCMeta.Block"]:
                            isimpl = self.core.get_attribute(parent, 'IsImplementation')
                            if not isimpl:
                                continue
                            isactive = self.core.get_attribute(parent, 'IsActive')
                            if not isactive:
                                continue

                        pvalue = self.core.get_attribute(node_list[i], 'Definition')
                        # parse the parameter json and store values
                        try:
                            # pvalue = pvalue.replace(/\bNaN\b/g, "null")
                            jsonval = json.loads(pvalue, strict=False)
                            self.paramValues[nodepath] = jsonval
                            # print 'done parsing'
                            if pmetaname.find('Block') == -1:
                                # FIXME: should this check even be made???
                                if (
                                        pmetaname.find('VerificationSetup') > -1 or
                                        pmetaname.find('RLTrainingSetup') > -1 or
                                        pmetaname.find('SystemIDSetup') > -1 or
                                        pmetaname.find('ValidationSetup') > -1 or
                                        pmetaname.find('SLTrainingSetUp') > -1 or
                                        pmetaname.find('AssuranceMonitorSetup') > -1 or
                                        pmetaname.find('EvaluationSetup') > -1
                                ):
                                    pmetaname = 'ExperimentSetup'
                                # print 'came here1'
                                self.modelValues[pmetaname]['params'][nodepath] = jsonval
                                # print 'came here2'
                            else:
                                self.modelValues['Implementation'][pnodepath]['params'][nodepath] = jsonval

                        except Exception:
                            self.paramValues[nodepath] = ''
                            pfname = self.core.get_fully_qualified_name(node_list[i])
                            estr = 'Unable to parse JSON input for parameter ' + pfname + ' value = ' + pvalue
                            self.errorValues.append(estr)
                            logger.error(estr)
                            logger.error(pvalue)
                        continue

                # lec model
                if (
                        metanode == self.META["ALCMeta.LEC_Model"] and
                        self.runSLTrainingSetup == 0 and
                        self.runAssuranceMonitorSetup == 0
                ):
                    parent = self.core.get_parent(node_list[i])
                    pmetanode = self.core.get_meta_type(parent)
                    # pmetaname = self.core.get_attribute(pmetanode, 'name')
                    # pnodepath = self.core.get_path(parent)
                    # inside active implementation blocks
                    if pmetanode == self.META["ALCMeta.Block"]:
                        isimpl = self.core.get_attribute(parent, 'IsImplementation')
                        if not isimpl:
                            continue

                        isactive = self.core.get_attribute(parent, 'IsActive')
                        if not isactive:
                            continue
                        lecparamname = self.core.get_attribute(node_list[i], 'DeploymentKey')
                        lecdata = self.core.get_pointer_path(node_list[i], 'ModelDataLink')
                        if not lecparamname or not lecdata:
                            continue
                        # store reference information
                        self.lecParamValues[lecparamname] = lecdata
                    
                    if self.runEvaluationSetup == 1:
                        lecdata = self.core.get_pointer_path(node_list[i], 'ModelDataLink')
                        self.lecParamValues[name] = lecdata

            # Return if there are any errors
            logger.info('DEBUG: End of second loop through model elements')
            if len(self.errorValues) > 0:
                estr = '\r\n'.join(self.errorValues)
                # self.result.setSuccess(false)
                # self.result.setError(estr)
                raise RuntimeError("Errors in model: %s" % estr)

            self.build_param_dictionary()
            logger.info('******** DEBUG LOCATION 4 ********')

            if (
                    self.runVerificationSetup == 1 or
                    self.runValidationSetup == 1
            ):
                if len(evaldatarefs) != 1:
                    message = 'Expected ONE data set in EvalData. Please add exactly one data set to EvalData!'
                    logger.error(message)
                    raise RuntimeError(message)
                    # self.result.setSuccess(false)
                    # self.result.setError(
                    #     'Expected ONE data set in EvalData. Please add exactly one data set to EvalData model!'
                    # )
                    # callback(null, self.result)
                    # return
                if len(plantdatarefs) == 0:
                    logger.error('Please set the SystemID reference in Plant Model!')
                    raise RuntimeError('Please set the SystemID reference in Plant Model!')
                    # self.result.setSuccess(false)
                    # self.result.setError('Please set the SystemID reference in Plant Model!')
                    # callback(null, self.result)
                    # return

            if self.runVerificationSetup == 1:
                self.build_verification_model_info()

            if self.runValidationSetup == 1:
                # logger.info('******** DEBUG LOCATION 41 ********')
                self.build_validation_model_info()
                # logger.info('******** DEBUG LOCATION 42 ********')

            if self.runSystemIDSetup == 1:
                if len(traindatarefs) == 0:
                    message = 'No Training Data set for system identification. ' \
                              'Please add some data sets to training data model.!'
                    logger.error(message)
                    raise RuntimeError(message)
                    # self.result.setSuccess(false)
                    # self.result.setError(
                    #     'No Training Data set for system identification. '
                    #     'Please add some data sets to training data model.!'
                    # )
                    # callback(null, self.result)
                    # return
                self.build_system_id_info()

            if self.verification_code:
                self.add_zip_content('Verification/run.m', self.verification_code)
                self.add_zip_file_info('Verification', ['run.m'])
                self.add_zip_content('Verification/run_init.m', self.run_init)
                self.add_zip_file_info('Verification', ['run_init.m'])
            if self.sysid_code:
                self.add_zip_content('Verification/sys_id.m', self.sysid_code)
                self.add_zip_file_info('Verification', ['sys_id.m'])

            # build lec info based on the references
            self.build_lec_param()
            self.lecCodeSetup['lecmodelurl'] = self.outputLECParam
            self.lecCodeSetup['runAssuranceMonitoring'] = self.runAssuranceMonitorSetup

            # FIXME: Any other job types where this may be needed?
            if (
                    self.active_node_meta_type == self.META["ALCMeta.ExperimentSetup"] or
                    self.active_node_meta_type == self.META["ALCMeta.RLTrainingSetup"]
            ):
                if (
                        self.config["generateRosLaunch"] or
                        self.generateROSLaunch or
                        self.config.get("dumpTargetArtifacts",False)
                ):
                    logger.info("Calling addRosLaunchFiles...")
                    self.exptParamSetup['generated_ros_launch'] = True
                    self.add_ros_launch_files()
                    if not self.containerInfo:
                        logger.error('ContainerInfo empty in deployment nodes. Please Check!')
                        raise RuntimeError('ContainerInfo empty in deployment nodes. Please Check!')

            if len(self.verification_model['Plant_Model']) and len(plantdatarefs) == 0:
                if self.get_plant_model():
                    if len(self.verification_model['Plant_Model_Content']):
                        self.add_zip_content('Verification/sys.xml', self.verification_model['Plant_Model_Content'][0])
                        self.add_zip_file_info('Verification', ['sys.xml'])
                    if len(self.verification_model['Configuration_Content']):
                        self.add_zip_content(
                            'Verification/sys.cfg', self.verification_model['Configuration_Content'][0]
                        )
                        self.add_zip_file_info('Verification', ['sys.cfg'])
                else:
                    logger.error('Cannot generate SpaceEx output for the Plant Model. Please Check!')
                    raise RuntimeError('Cannot generate SpaceEx output for the Plant Model. Please Check!')
                    # self.result.setSuccess(false)
                    # self.result.setError('Cannot generate SpaceEx output for the Plant Model. Please Check!')
                    # callback(null, self.result)

            fnames = self.verification_filehashes.keys()
            for h in fnames:
                fname = h
                file_hash = self.verification_filehashes[h]
                if file_hash:
                    contents = self.core.get_file(file_hash)
                    self.add_zip_content('Verification/' + fname, contents)
            self.add_zip_file_info('Verification', fnames)

            alldatarefs = []
            alldatarefs.extend(traindatarefs)
            alldatarefs.extend(evaldatarefs)
            alldatarefs.extend(plantdatarefs)

            num = 0
            fnames = []
            for f in traindatarefs:
                refnode = self.core.load_by_path(self.root_node, f)
                #self.core.add_member(self.resultnode, 'TData', refnode)
                self.resultnode_info['TData'].append(f)
                content = self.core.get_attribute(refnode, 'datainfo')
                fname = 'TrainingData' + str(num) + '.pkl'
                fnames.append(fname)
                name = 'TrainingData/' + fname
                self.add_zip_content(name, content)
                num += 1
            self.add_zip_file_info('TrainingData', fnames)

            if self.runEvaluationSetup == 1:
                for f in evaldatasets.keys():
                    self.evaldatasets[f] = []
                    for s in evaldatasets[f]:
                        refnode = self.core.load_by_path(self.root_node, s)
                        content = self.core.get_attribute(refnode, 'datainfo')
                        try:
                            jsonval = json.loads(content, strict=False)
                            self.evaldatasets[f].append(jsonval)
                        except Exception:
                            pass
                for f in valdatasets.keys():
                    self.valdatasets[f] = []
                    for s in valdatasets[f]:
                        refnode = self.core.load_by_path(self.root_node, s)
                        content = self.core.get_attribute(refnode, 'datainfo')
                        try:
                            jsonval = json.loads(content, strict=False)
                            self.valdatasets[f].append(jsonval)
                        except Exception:
                            pass

            num = 0
            fnames = []
            for f in valdatarefs:
                refnode = self.core.load_by_path(self.root_node, f)
                #self.core.add_member(self.resultnode, 'VData', refnode)
                self.resultnode_info['VData'].append(f)
                content = self.core.get_attribute(refnode, 'datainfo')
                fname = 'ValidationData' + str(num) + '.pkl'
                fnames.append(fname)
                name = 'ValidationData/' + fname
                self.add_zip_content(name, content)
                num += 1
            self.add_zip_file_info('ValidationData', fnames)

            num = 0
            fnames = []
            for f in evaldatarefs:
                refnode = self.core.load_by_path(self.root_node, f)
                #self.core.add_member(self.resultnode, 'EData', refnode)
                self.resultnode_info['EData'].append(f)
                content = self.core.get_attribute(refnode, 'datainfo')
                fname = 'EvalData' + str(num) + '.pkl'
                fnames.append(fname)
                name = 'EvalData/' + fname
                self.add_zip_content(name, content)
                num += 1
            self.add_zip_file_info('EvalData', fnames)

            num = 0
            fnames = []
            for f in plantdatarefs:
                refnode = self.core.load_by_path(self.root_node, f)
                # self.core.add_member(tempnode, 'PData', refnode)
                self.resultnode_info['PData'].append(f)
                content = self.core.get_attribute(refnode, 'datainfo')
                fname = 'PlantData' + str(num) + '.pkl'
                fnames.append(fname)
                name = 'PlantData/' + fname
                self.add_zip_content(name, content)
                num += 1
            self.add_zip_file_info('PlantData', fnames)

            num = 0
            fnames = []
            for f in scenariosets:
                refnode = self.core.load_by_path(self.root_node, f)
                content = self.core.get_attribute(refnode, 'datainfo')
                fname = 'ScenarioData' + str(num) + '.pkl'
                fnames.append(fname)
                name = 'ScenarioData/' + fname
                self.add_zip_content(name, content)
                num += 1
            self.add_zip_file_info('ScenarioData', fnames)

            self.build_all_params()

            

            

            self.campcount = self.get_campaign_count()

            try:
                project_info = self.project.get_project_info()

                result_dirs  = []
                result_nodes = []
                slurm_jobs   = []

                self.setup_and_build_repo(self.config.get('setupAndBuildRepo',False))

                for counter in range(self.campcount):
                    result_dir, resultnode = self.generate_files(counter)
                    if self.setupJupyterNB:
                        continue
                    # Get any user configured SLURM settings (make sure all dict keys are lower case)
                    # Handle special case parameters if User has not set them explicitly
                    # Make sure job type is included in slurm params since this is used to determine execution defaults
                    slurm_job_params = alc_common.dict_convert_key_case(self.slurm_params, "lower")
                    slurm_job_params[WebGMEKeys.job_type_key] = self.jobParams['jobType']
                    os.environ['ALC_SSH_PORT'] = self.ALC_SSH_PORT
                    os.environ['ALC_SSH_HOST'] = self.ALC_SSH_HOST
                    if (self.repo_home):
                        os.environ[WebGMEKeys.repo_home_key] = self.repo_home
                    else:
                        os.environ[WebGMEKeys.repo_home_key] = ''
                    if slurm_job_params.get(WebGMEKeys.job_name_key, None) is None:
                        if self.campcount > 1:
                            slurm_job_params[WebGMEKeys.job_name_key] = self.execName+'-'+str(counter)
                        else:
                            slurm_job_params[WebGMEKeys.job_name_key] = self.execName
                    if slurm_job_params.get(WebGMEKeys.time_limit_key, None) is None:
                        if self.exptParamSetup.get("timeout", None) is not None:
                            # Expt timeout specified in seconds, but slurm expects minutes.
                            # Convert and add grace period.
                            slurm_job_params[WebGMEKeys.time_limit_key] = int(
                                math.ceil(self.exptParamSetup["timeout"] / SECONDS_PER_MINUTE) + SLURM_GRACE_TIME_MIN)

                    slurm_job_params[WebGMEKeys.project_owner_key] = project_info[WebGMEKeys.project_owner_key]
                    slurm_job_params[WebGMEKeys.project_name_key] = project_info[WebGMEKeys.project_name_key]
                    slurm_job_params[WebGMEKeys.result_dir_key] = str(result_dir)
                    #resultnodepath = self.core.get_path(resultnode)
                    resultnodepath = resultnode
                    slurm_job_params[UpdateKeys.data_node_path_key] = resultnodepath
                    slurm_job_params[WebGMEKeys.command_for_srun_key] = "{0} {1}".format(
                        self.bash_command, self.bash_script_name
                    )
                    if self.workflow_output_key in self.config:
                        slurm_job_params[self.workflow_output_key.lower()] = self.config.get(self.workflow_output_key)
                        slurm_job_params[self.campaign_count_key] = self.campcount
                    
                    slurm_jobs.append(slurm_job_params)
                    result_nodes.append(resultnode)
                    result_dirs.append(result_dir)
            
                if not self.setupJupyterNB:
                    counter = 0
                    for counter in range(self.campcount):
                        slurm_job_params = slurm_jobs[counter]
                        resultnode  = result_nodes[counter]
                        result_dir = result_dirs[counter]
                        # Deploy job to Slurm cluster
                        logger.info("Dealing with job number "+str(counter))
                        logger.info("Deploying job to slurm cluster with parameters: %s" % str(slurm_job_params))
                        slurm_executor.slurm_deploy_job(result_dir, job_params=slurm_job_params)
                        #self.core.set_attribute(resultnode, 'jobstatus', 'Submitted')

                # Handle case where user wants to run job interactively in jupyter notebook
                if self.setupJupyterNB:
                    if self.runEvaluationSetup != 1:
                        self.create_jupyter_notebook()
                    else:
                        self.create_jupyter_notebook(eval=True)
            except Exception as e:
                logger.error('Error encountered while executing experiment. Please check!')
                logger.error('Error ' + str(e))
                raise RuntimeError('Error encountered while executing experiment. Please check! ' + str(e))

            logger.info('done---------------------------------')

            #self.util.save(self.root_node, self.commit_hash, 'master', 'Launch Expt Finished')
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