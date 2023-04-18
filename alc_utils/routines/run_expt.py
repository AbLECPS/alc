# Methods to execute experiments
# The methods in this code are invoked from ALCJobRunner
# it uses the execution_runner and other utilities defined in alc_utils to complete the job.
# experiments in alc_iver using the execution runner are configured with deployment dictionary in iverDep.py
from __future__ import print_function

import os
import yaml
import shutil
import errno
import json
from alc_utils.routines import setup, config_generator
import alc_utils.file_uploader
import alc_utils.config as alc_config
import copy

# method invoked from ALCJobRunner for running experiments and their campaigns


def run(updatedExptParams, campaignParams, deployment_template_dict, project_name, container_info={}, campaign_idx=-1):
    configs = config_generator.generateConfigs(
        updatedExptParams, campaignParams)
    outputs = executeExperiments(configs, updatedExptParams, campaignParams,
                                 deployment_template_dict, project_name, container_info, campaign_idx)
    return outputs


# method invoked from ALCJobRunner for running RL experiments
def runRL(rlcodepath, datapath, updatedExptParams, campaignParams, deployment_template_dict, project_name, container_info={}, campaign_idx=-1):
    configs = config_generator.generateConfigs(
        updatedExptParams, campaignParams)
    outputs = executeRLExperiments(rlcodepath, datapath, configs, updatedExptParams, campaignParams,
                                   deployment_template_dict, project_name, container_info, campaign_idx)
    return outputs


def createExptBasePath():
    cwd = os.getcwd()
    bname = os.path.basename(cwd)
    fullpath = os.path.join(cwd, bname)
    if not os.path.isdir(os.path.join(fullpath)):
        os.makedirs(fullpath)
    x = os.path.abspath(alc_utils.config.WORKING_DIRECTORY)
    length = len(x)
    localdir = fullpath[length + 1:]

    return fullpath, localdir


# execute all experiments (it could be a single experiment or a set of experiments in a campaign
def executeExperiments(configs, expt_params, campaign_params, deployment_template_dict, project_name, container_info={}, campaign_idx=-1):
    (full_base_path, rel_base_path) = createExptBasePath()

    # Generate experiment directory (with all files necessary for execution)
    expt_dep_files = {}
    for i, config_key in enumerate(configs):
        if (campaign_idx > -1 and i != campaign_idx):
            continue
        config = configs[config_key]
        expt_dep_files[i] = createExptFolder(expt_params,
                                             full_base_path,
                                             rel_base_path,
                                             i,
                                             config,
                                             deployment_template_dict,
                                             project_name,
                                             container_info,
                                             campaign_idx)

    # Determine upload path prefix
    path_prefix = expt_params.get("path_prefix", None)
    path_prefix = expt_params.get("fs_path_prefix", path_prefix)
    exptBaseFolderName = os.path.basename(full_base_path)
    if path_prefix:
        pathPrefixFinal = os.path.join(path_prefix, exptBaseFolderName)
    else:
        pathPrefixFinal = exptBaseFolderName

    configkeys = configs.keys()
    uploadparams = expt_params
    uploadparams['fs_path_prefix'] = pathPrefixFinal

    # Run each experiment & upload results
    outputs = []
    for i in expt_dep_files.keys():
        if (campaign_idx > -1 and i != campaign_idx):
            continue
        exptresults = executeExperiment(expt_dep_files[i])

        # Upload results folder, as specified in expt_config
        local_expt_path = 'config-' + str(i)
        foldername = os.path.join(full_base_path, local_expt_path)
        x1 = alc_utils.file_uploader.FileUploader()
        log_results = x1.upload_with_params(foldername, uploadparams)
        print('log results ' + str(log_results))
        result_url = setup.createResultNotebook2(foldername)
        log_results['result_url'] = result_url
        einfo = configs[configkeys[i]]
        log_results['exptParams'] = einfo
        # if (error_code == 0):
        outputs.append(log_results)

    return outputs


def createExptFolder(exptParams, full_base_path, rel_base_path, i, config, deployment_template_dict, project_name, container_info={}, campaign_idx=-1):
    # Create experiment directory
    local_expt_path = 'config-' + str(i)
    full_expt_path = os.path.join(full_base_path, local_expt_path)
    if (not os.path.isdir(full_expt_path)):
        os.makedirs(full_expt_path)

    # Write parameters to file
    param_file_name = os.path.join(local_expt_path, 'parameters.yml')
    full_param_file_name = os.path.join(full_base_path, param_file_name)
    print('******************project name in createExptFolder*****************************' + str(project_name))
    config['project_name'] = project_name
    with open(full_param_file_name, 'w') as yaml_file:
        yaml.safe_dump(config, yaml_file, default_flow_style=False)

    if (container_info):
        cikeys = container_info.keys()
        updated_deployment_template_dict = copy.deepcopy(
            deployment_template_dict)
        updated_deployment_template_dict['containers'] = []
        for launch_name in cikeys:
            ldepinfo = container_info[launch_name]

            if (not ldepinfo):
                continue
            ldepinfo = json.loads(ldepinfo)

            print('ldepinfo ', ldepinfo)
            container_name = ldepinfo.get('name', None)
            if (not container_name):
                continue

            cwd = os.getcwd()
            launch_file_name = os.path.join(
                cwd, 'ModelData', 'launch_files', launch_name)
            config['launchfile'] = launch_file_name

            config_file_name = 'parameters_'+launch_name+'.yml'
            param_file_name = os.path.join(local_expt_path, config_file_name)
            full_param_file_name = os.path.join(
                full_base_path, param_file_name)
            with open(full_param_file_name, 'w') as yaml_file:
                yaml.safe_dump(config, yaml_file, default_flow_style=False)

            lencontainers = len(deployment_template_dict['containers'])
            for k in range(0, lencontainers):
                ck = deployment_template_dict['containers'][k]
                ckname = ck.get('name', None)
                if ((not ckname) or (ckname != container_name)):
                    continue
                deployment_template_dict['containers'][k]['input_file'] = config_file_name
                updated_deployment_template_dict['containers'].append(
                    deployment_template_dict['containers'][k])
                break
        if (len(updated_deployment_template_dict['containers'])):
            deployment_template_dict = updated_deployment_template_dict

    # Fill experiment deployment info template and write to file
    dep_file_name = os.path.join(local_expt_path, 'exe_config.json')
    full_dep_file_name = os.path.join(full_base_path, dep_file_name)
    base_dir = os.path.join(rel_base_path, local_expt_path)
    dep_dict = fill_deployment_template(
        base_dir, exptParams, deployment_template_dict)
    with open(full_dep_file_name, 'w') as fp:
        json.dump(dep_dict, fp)

    return full_dep_file_name


def fill_deployment_template(base_dir, expt_params, deployment_template_dict):
    dep_dict = deployment_template_dict.copy()

    # Update values in deployment dictionary as needed
    dep_dict["base_dir"] = base_dir
    if "timeout" in expt_params:
        dep_dict["timeout"] = expt_params["timeout"]

    return dep_dict


def executeExperiment(configfilename):
    return executeExperimentFromDocker(configfilename)


def executeExperimentFromDocker(configfilename):
    from alc_utils import execution_runner
    print("*********** STARTING EXPERIMENT IN DOCKER *************")
    runner = execution_runner.ExecutionRunner(configfilename)
    result, resultdir = runner.run()
    print('result in executeexpt ')
    print(' result ' + str(result))
    print("*********** EXPERIMENT COMPLETE IN DOCKER *************")
    return [(configfilename, resultdir)]


def copyRLFolder(rlcodepath, full_base_path):
    print('++++++++++++++++ COPYING RL MODEL ++++++++++++++++')

    srcdir = rlcodepath
    destdir = os.path.join(full_base_path, 'RLModel')

    print('src dir : ' + srcdir)
    print('dst dir : ' + destdir)
    try:
        shutil.copytree(srcdir, destdir)
    except OSError as exc:  # python >2.5
        if exc.errno == errno.ENOTDIR:
            shutil.copy(srcdir, destdir)
        else:
            raise

    return destdir


# execute all experiments (it could be a single experiment or a set of experiments in a campaign
def executeRLExperiments(rlcodepath, datapath, configs, expt_params, campaign_params, deployment_template_dict, project_name, container_info={}, campaign_idx=-1):
    print('+++++++++++in Execute RL Experiments ')
    if rlcodepath:
        print(' rl code path ' + rlcodepath)
    else:
        print(' rl code  path - not defined ')

    if datapath:
        print(' data path ' + datapath)
    else:
        print(' data path - not defined ')

    (full_base_path, rel_base_path) = createExptBasePath()

    # Set RL path appropriately
    updated_rl_path = ''
    if datapath:
        datapath_basename = os.path.basename(datapath)
        datamodelpath = os.path.join(datapath, 'RLModel')
        if os.path.exists(datamodelpath) and os.path.isdir(datamodelpath):
            updated_rl_path = copyRLFolder(datamodelpath, full_base_path)
        elif datapath_basename == 'RLModel':
            updated_rl_path = copyRLFolder(datapath, full_base_path)
    if updated_rl_path == '' and rlcodepath:
        updated_rl_path = copyRLFolder(rlcodepath, full_base_path)

    updatesToConfig = {}
    updatesToConfig['rl_model_dir'] = updated_rl_path
    if expt_params['testing'] == 0:
        deployment_template_dict['echo_logs'] = False

    # Generate experiment directory (with all files necessary for execution)
    expt_dep_files = {}
    for i, config_key in enumerate(configs):
        if (campaign_idx > -1 and i != campaign_idx):
            continue
        config = configs[config_key]
        expt_dep_files[i] = createExptFolder(expt_params,
                                             full_base_path,
                                             rel_base_path,
                                             i,
                                             config,
                                             deployment_template_dict,
                                             project_name,
                                             container_info,
                                             campaign_idx)

    # Determine upload path prefix
    path_prefix = expt_params.get("path_prefix", None)
    path_prefix = expt_params.get("fs_path_prefix", path_prefix)
    exptBaseFolderName = os.path.basename(full_base_path)
    if path_prefix:
        pathPrefixFinal = os.path.join(path_prefix, exptBaseFolderName)
    else:
        pathPrefixFinal = exptBaseFolderName

    num_episodes = expt_params.get("num_episodes", 1)
    configkeys = configs.keys()
    uploadparams = expt_params
    uploadparams['fs_path_prefix'] = pathPrefixFinal

    outputs = []
    for i in expt_dep_files.keys():
        if (campaign_idx > -1 and i != campaign_idx):
            continue
        exptresults = executeExperiment(expt_dep_files[i])
        local_expt_path = 'config-' + str(i)
        foldername = os.path.join(full_base_path, local_expt_path)
        if expt_params['testing'] == 1:
            # (error_code, log_results) = uploadFiles(foldername, uploadDescription, pathPrefixFinal)
            # print('error code' + str(error_code))
            x1 = alc_utils.file_uploader.FileUploader()
            log_results = x1.upload_with_params(foldername, uploadparams)
            print('log results ' + str(log_results))
            result_url = setup.createResultNotebook2(foldername)
            log_results['result_url'] = result_url
            einfo = configs[configkeys[i]]
            log_results['exptParams'] = einfo
            outputs.append(log_results)

    if expt_params['testing'] == 0:
        foldername = os.path.join(full_base_path, 'RLModel')
        x1 = alc_utils.file_uploader.FileUploader()
        log_results = x1.upload_with_params(foldername, uploadparams)
        # (error_code, log_results) = uploadFiles(foldername, uploadDescription, pathPrefixFinal)
        # print('error code' + str(error_code))
        print('log results ' + str(log_results))
        result_url = setup.createResultNotebook2(foldername)
        log_results['result_url'] = result_url
        log_results['exptParams'] = expt_params
        outputs.append(log_results)

    return outputs
