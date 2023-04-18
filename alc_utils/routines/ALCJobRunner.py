#!/usr/bin/env python
# This contains python code for executing ALC jobs
# Based on the jobtype, it invokes the appropriate functions in alc_utils.
from __future__ import print_function

import os
import json
import sys
from alc_utils import common as alc_common
from alc_utils.routines import setup, ver_setup, run_expt
from shutil import copyfile

# method invoked to run the jobs


def run(data, deployment_dict, setupJupyterNB, params, campcount=-1):
    print(sys.version)
    #print('Expt Params :' + str(params))

    eParams = json.loads(params)
    jobInfo = eParams['jobInfo']
    if jobInfo['jobType'] == 'UPLOAD':
        return setup.runUpload(eParams['exptInfo'])

    exptParams = alc_common.dict_convert_key_case(eParams['exptInfo'], "lower")
    campaignParams = eParams['campaignInfo']
    lecInfo = eParams['lecInfo']
    lecCodeInfo = eParams['leccodeInfo']
    fileInfo = eParams['fileInfo']
    containerInfo = eParams.get('containerInfo', None)
    projectName = jobInfo['project']
    print('+++++++++++++ projectName ' + str(projectName))

    setupFolderPath = ''
    if not setup.checkAlreadySetup():
        setupFolderPath, mainurl, localdir = setup.setupInteractive2(jobInfo)
        if (setupJupyterNB and (
                (jobInfo['jobType'] != 'EVALUATIONSETUP') and (jobInfo['jobType'] != 'VERIFICATIONSETUP') and (
                jobInfo['jobType'] != 'SYSTEMIDSETUP') and (jobInfo['jobType'] != 'VALIDATIONSETUP'))):
            print('+++++++++++++ job type ' + str(jobInfo['jobType']))
            return mainurl

    if setupFolderPath:
        os.chdir(setupFolderPath)

    # get the required models, data and update params with their location
    if data:
        filesdir = setup.getModelData2(data.val)
    else:
        filesdir = os.path.join(os.getcwd(), 'ModelData')

    print('**********************')
    print(' job data downloaded to ' + filesdir)
    print('**********************')
    print(' job info ' + str(jobInfo))
    print('**********************')
    print(' file info ' + str(fileInfo))
    print('**********************')
    print(' lec code info ' + str(lecCodeInfo))

    print('**********************')
    print('downloading data')
    downloadDataInfo, downloadDataInfo_metadata = setup.downloadDatas2(
        filesdir, fileInfo)
    print('data info ' + str(downloadDataInfo))
    print('**********************')
    print('downloading Model')
    downloadModelInfo = setup.downloadModels2(lecInfo)
    print('model info ' + str(downloadModelInfo))
    updatedModelInfo = setup.updateModelInfo2(downloadModelInfo)
    print('updatedModelInfo ' + str(updatedModelInfo))
    print('**********************')

    # print('**********************')
    #print(' expt info ' + str(exptParams))
    print('**********************')
    print(' campaign info ' + str(campaignParams))
    # return at this point for evaluation setup
    if jobInfo['jobType'] == 'EVALUATIONSETUP':
        return setup.setupExpt2(jobInfo)
    # 'EvalData','TrainingData','PlantData'
    if jobInfo['jobType'] == 'VERIFICATIONSETUP':
        return ver_setup.setupExptVerification(jobInfo, downloadModelInfo, downloadDataInfo['PlantData'],
                                               downloadDataInfo['EvalData'])

    if jobInfo['jobType'] == 'SYSTEMIDSETUP':
        print('======================SYSTEMIDSETUP++++++++')
        return ver_setup.setupExptSysID(jobInfo, downloadModelInfo, downloadDataInfo['PlantData'],
                                        downloadDataInfo['TrainingData'])

    if jobInfo['jobType'] == 'VALIDATIONSETUP':
        return ver_setup.setupValidation(jobInfo, downloadModelInfo, downloadDataInfo['PlantData'],
                                         downloadDataInfo['EvalData'])

    updatedExptParams = setup.updateExptParamsWithLEC(
        exptParams, updatedModelInfo)

    rl_code_dir = None
    if fileInfo.has_key('RLModel'):
        rl_code_dir = os.path.join(filesdir, 'RLModel')

    lec_code_dir = None
    lec_data_loader = None
    if fileInfo.has_key('LECModel'):
        lec_code_dir = os.path.join(filesdir, 'LECModel')
        data_loader_path = os.path.join(filesdir, 'LECModel', 'data_loader.py')
        if (os.path.exists(data_loader_path)):
            lec_data_loader = data_loader_path

    lec_model_dir = None
    if updatedModelInfo:
        ukeys = updatedModelInfo.keys()
        if 'TrainedLECModel' in ukeys:
            lec_model_dir = updatedModelInfo['TrainedLECModel']

    rl_model_dir = None
    if updatedModelInfo:
        ukeys = updatedModelInfo.keys()
        if 'rl_model_dir' in ukeys:
            rl_model_dir = updatedModelInfo['rl_model_dir']

    # execute based on job type
    if jobInfo['jobType'] == 'EXPERIMENTSETUP':
        return run_expt.run(updatedExptParams, campaignParams, deployment_dict, projectName, containerInfo, campaign_idx=campcount)

    if jobInfo['jobType'] == 'SLTRAININGSETUP':
        from alc_utils.routines import run_training

        training_data = []
        training_metadata = []
        output_dir = ''
        validation_data = []
        evaluation_data = []
        validation_metadata = []
        evaluation_metadata = []

        if exptParams.get('dataset_name') == "CUSTOM":
            if lec_data_loader is None:
                raise IOError(
                    "Parameters specify a CUSTOM dataset, but no custom dataset code was found.")
            exptParams['dataset_name'] = lec_data_loader

        if len(downloadDataInfo['TrainingData']):
            training_data = downloadDataInfo['TrainingData']
            training_data_metadata = downloadDataInfo_metadata['TrainingData']

        if len(downloadDataInfo['ValidationData']):
            print('setting validation data')
            validation_data = downloadDataInfo['ValidationData']
            validation_metadata = downloadDataInfo_metadata['ValidationData']
            print('validation data ' + str(validation_metadata))

        if len(downloadDataInfo['EvalData']):
            print('setting eval data')
            evaluation_data = downloadDataInfo['EvalData']
            evaluation_metadata = downloadDataInfo_metadata['EvalData']
            print('eval data ' + str(evaluation_metadata))

        if (len(validation_metadata) == 0):
            validation_metadata = None

        if (len(evaluation_metadata) == 0):
            evaluation_metadata = None

        if training_data and lec_code_dir:
            output_dir = setup.createTrainingOutputDirectory()

        if not training_data or not lec_code_dir or not output_dir:
            ret = {"status": "check training data, model"}
            return ret
        param_dict = alc_common.dict_convert_key_case(exptParams, "lower")

        model_module = None
        lec_model_path = os.path.join(lec_code_dir, 'LECModel.py')
        if os.path.exists(lec_model_path):
            model_module = alc_common.load_python_module(lec_model_path)
            optimizer = param_dict.get("optimizer")
            if optimizer and optimizer == 'CUSTOM':
                optimizer = model_module.get_optimizer(**param_dict)
                param_dict["optimizer"] = optimizer
            loss = param_dict.get("loss")
            if loss and loss == 'CUSTOM':
                loss = model_module.get_loss(**param_dict)
                param_dict["loss"] = loss
            metrics = param_dict.get("metrics")
            if metrics and metrics == 'CUSTOM':
                metrics = model_module.get_metrics(**param_dict)
                param_dict["metrics"] = metrics
            callbacks = param_dict.get("callbacks")
            if callbacks and callbacks == 'CUSTOM':
                callbacks = model_module.get_callbacks(**param_dict)
                param_dict["callbacks"] = callbacks
        ret = run_training.run_training(param_dict, training_data_metadata, training_data, lec_code_dir, output_dir,
                                        lec_model_dir, validation_metadata, evaluation_metadata)
        ret['result_url'] = setup.createResultNotebook2(output_dir)
        if (not (ret.has_key('exptParams'))):
            ret['exptParams'] = eParams['exptInfo']
        return ret

    if jobInfo['jobType'] == 'RLTRAININGSETUP':

        if rl_code_dir:
            print('rl code dir ' + str(rl_code_dir))
        else:
            print('rl code dir ---none')

        if rl_model_dir:
            print('rl model dir ' + str(rl_model_dir))
        else:
            print('rl model dir ---none')

        if not rl_code_dir and not rl_model_dir:
            print(' RL model is not specified')
            ret = {"status": "check training model. cannot find RL code or RL model"}
            return ret
        return run_expt.runRL(rl_code_dir, rl_model_dir, updatedExptParams, campaignParams, deployment_dict, projectName, containerInfo, campaign_idx=campcount)

    if jobInfo['jobType'] == 'ASSURANCEMONITORSETUP':
        # from alc_utils.routines import train_assurance_monitor
        # if len(downloadDataInfo['TrainingData']):
        #     training_data = downloadDataInfo['TrainingData']
        # keys = downloadModelInfo.keys()
        # model_dir = ''
        # if len(keys):
        #     model_dir = downloadModelInfo[keys[0]]

        # am_data_formatter = ''
        # if lec_code_dir:
        #     am_data_formatter = os.path.join(lec_code_dir, 'data_formatter.py')

        # cur_dir = os.getcwd()
        # base_name = os.path.basename(cur_dir)
        # output_dir = setup.copyModelDir(model_dir, cur_dir)
        # print('output dir ' + str(output_dir))
        # train_assurance_monitor.run_assurance_monitor_training(updatedExptParams, training_data, model_dir,
        #                                                        output_dir, am_data_formatter_path=am_data_formatter)
        # return setup.runUploadAMResults(updatedExptParams, base_name, output_dir)
        from alc_utils.routines import train_assurance_monitor

        training_data = []
        training_data_metadata = []
        validation_data = []
        validation_metadata = []
        evaluation_data = []
        evaluation_metadata = []
        output_dir = ''

        if updatedExptParams.get('dataset_name') == "CUSTOM":
            if lec_data_loader is None:
                raise IOError(
                    "Parameters specify a CUSTOM dataset, but no custom dataset code was found.")
            updatedExptParams['dataset_name'] = lec_data_loader

        if len(downloadDataInfo['TrainingData']):
            training_data = downloadDataInfo['TrainingData']
            training_data_metadata = downloadDataInfo_metadata['TrainingData']

        if len(downloadDataInfo['ValidationData']):
            print ('setting validation data')
            validation_data = downloadDataInfo['ValidationData']
            validation_metadata = downloadDataInfo_metadata['ValidationData']
            print (
                '*********************validation_data {0}'.format(validation_data))
            print (
                '*********************validation_metadata  {0}'.format(validation_metadata))

        if len(downloadDataInfo['EvalData']):
            evaluation_data = downloadDataInfo['EvalData']
            evaluation_metadata = downloadDataInfo_metadata['EvalData']
            print (
                '*********************evaluation data {0}'.format(evaluation_data))
            print (
                '*********************evaluation data metadata  {0}'.format(evaluation_metadata))

        if (len(validation_metadata) == 0):
            print ('*********************validation data metadata is null ')

            validation_metadata = None

        if (len(evaluation_metadata) == 0):
            print ('*********************evaluation data metadata is null ')
            evaluation_metadata = None
            exit(0)

        if training_data and lec_code_dir:
            output_dir = setup.createTrainingOutputDirectory()

        if not training_data or not lec_code_dir or not output_dir:
            ret = {"status": "check training data, model"}
            return ret
        param_dict = alc_common.dict_convert_key_case(exptParams, "lower")

        model_module = None
        lec_model_path = os.path.join(lec_code_dir, 'LECModel.py')
        if os.path.exists(lec_model_path):
            model_module = alc_common.load_python_module(lec_model_path)
            optimizer = param_dict.get("optimizer")
            if optimizer and optimizer == 'CUSTOM':
                optimizer = model_module.get_optimizer(**param_dict)
                param_dict["optimizer"] = optimizer
            loss = param_dict.get("loss")
            if loss and loss == 'CUSTOM':
                loss = model_module.get_loss(**param_dict)
                param_dict["loss"] = loss
            metrics = param_dict.get("metrics")
            if metrics and metrics == 'CUSTOM':
                metrics = model_module.get_metrics(**param_dict)
                param_dict["metrics"] = metrics
            callbacks = param_dict.get("callbacks")
            if callbacks and callbacks == 'CUSTOM':
                callbacks = model_module.get_callbacks(**param_dict)
                param_dict["callbacks"] = callbacks

        model_dir = lec_model_dir

        print('downloadModelInfo {0}'.format(downloadModelInfo))

        am_data_formatter = ''
        if lec_code_dir:
            am_data_formatter = os.path.join(lec_code_dir, 'data_formatter.py')

        cur_dir = os.getcwd()
        base_name = os.path.basename(cur_dir)
        output_dir = setup.copyModelDir(model_dir, cur_dir, True)

        print('output dir ' + str(output_dir))
        data_formatter_dstpath = os.path.join(output_dir, 'data_formatter.py')
        copyfile(am_data_formatter, data_formatter_dstpath)

        am_definition_path = os.path.join(lec_code_dir, 'am_net.py')
        am_defn_dstpath = os.path.join(output_dir, 'am_net.py')
        if (os.path.exists(am_definition_path)):
            copyfile(am_definition_path, am_defn_dstpath)

        lec_definition_path = os.path.join(lec_code_dir, 'LECModel.py')
        lec_defn_dstpath = os.path.join(output_dir, 'LECModel.py')
        if (os.path.exists(lec_definition_path)):
            copyfile(lec_definition_path, lec_defn_dstpath)

        train_assurance_monitor.run_assurance_monitor_training(updatedExptParams,
                                                               training_data,
                                                               model_dir,
                                                               output_dir,
                                                               am_data_formatter_path=am_data_formatter,
                                                               validation_data_dirs=validation_data,
                                                               testing_data_dirs=evaluation_data)
        return setup.runUploadAMResults(updatedExptParams, base_name, output_dir)
