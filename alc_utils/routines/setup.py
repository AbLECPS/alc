# this code includes support functions for setup of experiments

import os
import shutil
import errno
import pickle
import sys
import time
import json
import glob
import alc_utils
from alc_utils import file_uploader, file_downloader


# this code includes functions that work with the current version of the ALC as well as older versions.
# hence some of the functions have a suffix number. this needs to be cleaned up at somepoint.

def checkSetupInteractive(setupJupyterNB):
    if not setupJupyterNB:
        return False
    nbfiles = glob.glob('*.ipynb')
    if nbfiles:
        return False
    return True


def checkAlreadySetup():
    nbfiles = glob.glob('*.ipynb')
    if nbfiles:
        return True
    return False


def setupInteractive(paramstr):
    print ('starting setup interactive')
    params = json.loads(paramstr)
    prjname = 'project'
    modelname = 'model'
    datetime = time.strftime("%Y_%m_%d_%H_%M_%S", time.gmtime())

    if (paramstr):
        params = json.loads(paramstr)
        if (params):
            paramkeys = params.keys()
            if ('project' in paramkeys):
                prjname = params['project']
            if ('model' in paramkeys):
                modelname = params['model']
            if ('datetime' in paramkeys):
                datetime = str(params['datetime'])

    localdir = os.path.join(prjname, modelname, datetime)
    copyfilestojupyter(localdir)
    ret = createNotebook(localdir, 'main.ipynb')
    return ret


def setupInteractive2(params):
    print ('starting setup interactive')
    prjname = 'project'
    modelname = 'model'
    datetime = time.strftime("%Y_%m_%d_%H_%M_%S", time.gmtime())
    if (params):
        paramkeys = params.keys()
        if ('project' in paramkeys):
            prjname = params['project']
        if ('model' in paramkeys):
            modelname = params['model']
        if ('datetime' in paramkeys):
            datetime = str(params['datetime'])
    localdir = os.path.join(prjname, modelname, datetime)
    folderpath = copyfilestojupyter(localdir)
    ret_url = ''
    if (params['jobType'] == 'EVALUATIONSETUP'):
        ret_url = createInitNotebook2(folderpath, localdir)
    else:
        ret_url = createNotebook2(folderpath, localdir)
    return folderpath, ret_url, localdir


def setupExpt(data, paramstr):
    print ('starting setup experiment')
    print ('filename ', data.val)
    datas, codes, destdir = getModelData(data.val)
    (downloadedDirs, topdownloadfolder) = downloadData(datas)
    params = json.loads(paramstr)
    localdir = os.path.join(
        params['project'], params['model'], str(params['datetime']))
    copyfilestojupyter(localdir)
    ret = createInitNotebook(localdir, 'main.ipynb')
    return ret


def setupExpt2(params):
    localdir = os.path.join(
        params['project'], params['model'], str(params['datetime']))
    copyfilestojupyter(localdir)
    ret = createInitNotebook(localdir, 'main.ipynb')
    return ret


def setupExptVerification(params, modelinfo):
    localdir = os.path.join(
        params['project'], params['model'], str(params['datetime']))
    copyfilestomatlabjupyter(localdir, modelinfo)
    ret = createInitMatlabNotebook(localdir, 'main.ipynb')
    return ret


def setupExpt3(folderpath, localdir):
    localdir = '.'
    ret = createInitNotebook2(folderpath, localdir)
    return ret


def downloadContent(info):
    dd = alc_utils.config.DOWNLOAD_DIRECTORY
    print (' in download content')
    contents = info
    if (type(contents) is str):
        contents = json.loads(contents)
    if (type(contents) is list):
        contents = contents[0]
    if (type(contents) is dict):
        x = contents.keys()
        if ((len(x) > 0) and ('directory' not in x)):
            contents = contents[x[0]]

    #print 'contents in download Content ' + str(contents)

    d = file_downloader.FileDownloader()
    (result_code, downloaded_dir) = d.download(contents, dd)
    fullpath = ''
    if (result_code == 0):
        print ('download_dir length ', str(len(downloaded_dir)))
        print ('download dir', downloaded_dir[0])
        fullpath = os.path.abspath(downloaded_dir[0])
    return fullpath, contents


def downloadDatas2(filesdir, fileInfo):
    fileInfoKeys = fileInfo.keys()
    datakeys = ['ScenarioData', 'EvalData',
                'TrainingData', 'PlantData', 'ValidationData']
    ret = {}
    ret_metadata = {}
    for k in datakeys:
        ret[k] = []
        ret_metadata[k] = []
        if (k not in fileInfoKeys):
            continue
        for filename in fileInfo[k]:
            print ('files dir ', filesdir)
            print (' k ', k)
            print ('filename ', filename)
            fname = os.path.join(filesdir, k, filename)
            print (' fname ', fname)
            datainfo = getDataInfo(fname)  # pickle.load( open( fname, "rb" ) )
            print (' data info ', str(datainfo))
            downloaddir, metainfo = downloadContent(datainfo)
            if (downloaddir):
                ret[k].append(downloaddir)
                ret_metadata[k].append(metainfo)
                print (
                    'Download Data {0} --- Directory {1}'.format(filename, str(downloaddir)))
            else:
                print ('Download Data {0} was not successful'.format(filename))

    retkeys = ret.keys()
    if ('EvalData' in retkeys):
        downloaded_file_dirs = {}
        downloaded_file_dirs["datadirs"] = ret['EvalData']
        with open('datafiles.json', 'w') as outfile:
            json.dump(downloaded_file_dirs, outfile)

    return ret, ret_metadata


def downloadModels2(lecInfo):
    lecInfoKeys = lecInfo.keys()
    ret = {}
    for k in lecInfoKeys:
        modelinfo = lecInfo[k]
        downloaddir, metainfo = downloadContent(modelinfo)
        if (downloaddir):
            ret[k] = downloaddir
            print (
                'Download Model {0} --- Directory {1}'.format(k, str(downloaddir)))
        else:
            print ('Download Model {0} was not successful'.format(k))
    return ret


def downloadModels3(lecInfo):
    lecInfoKeys = lecInfo.keys()
    ret = {}
    ret_metadata = {}
    for k in lecInfoKeys:
        modelinfo = lecInfo[k]
        downloaddir, metainfo = downloadContent(modelinfo)
        if (downloaddir):
            ret[k] = downloaddir
            ret_metadata[k] = metainfo
            print (
                'Download Model {0} --- Directory {1}'.format(k, str(downloaddir)))
        else:
            print ('Download Model {0} was not successful'.format(k))
    return ret, ret_metadata


def updateModelInfo2(modelfolders):
    keys = modelfolders.keys()
    ret = modelfolders
    for k in keys:
        foldername = modelfolders[k]
        ret[k] = foldername
        if (foldername):
            updated = getdirwithin(foldername)
            if (updated):
                ret[k] = updated
    return ret


def updateExptParams2(exptParams, updatedModelInfo):
    if not updatedModelInfo:
        return exptParams
    updatedExptParams = updateExptParamsWithLEC(exptParams, lecpaths)
    print ('updatedExptParams ', str(updatedExptParams))
    return updatedExptParams


def setupLECTraining(data, leccode):
    downloadedDirs = {}
    topdownloadfolder = ''
    leccodepath = ''
    modelModule = ''
    downloadurl = {}
    if type(data) is ArchiveData:
        print ('starting setup experiment')
        print ('filename ', data.val)
        datas, codes, destdir = getModelData(data.val)
        (downloadedDirs, topdownloadfolder) = downloadData(datas)
        downloadurl = datas
    elif type(data) is str:
        datas = {}
        datas["file"] = json.loads(data)
        (downloadedDirs, topdownloadfolder) = downloadData(datas)
        downloadurl = data

    if (leccode):
        leccodepath = setupLECCode(leccode)

    outputdirname = createTrainingOutputDirectory()

    return (downloadedDirs["datadirs"], leccodepath, outputdirname, downloadurl)


# This function download and configures the necessary data for continuing training from a previously trained model
def setupContinuedLECTraining(dataset_storage_metadata, previous_model_storage_metadata):
    # Create output directory for this training run
    output_dir = createTrainingOutputDirectory()

    # Check type of previous_model_storage_metadata and load appropriately
    if type(previous_model_storage_metadata) is str:
        previous_model_storage_metadata = json.loads(
            previous_model_storage_metadata)
    elif type(previous_model_storage_metadata) is dict:
        pass
    else:
        raise TypeError("Specified previous model storage metadata has unexpected type (%s)." % str(
            type(previous_model_storage_metadata)))

    # Download previously trained model to default downloads directory
    downloader = file_downloader.FileDownloader()
    result_code, download_dirs = downloader.download(
        previous_model_storage_metadata)
    if len(download_dirs) != 1:
        raise ValueError(
            "Downloader returned multiple directories when fetching previous model files")

    # Get downloaded model directory and various filenames
    model_dir = download_dirs[0]
    previous_trained_model_file = os.path.join(model_dir, 'model.keras')
    previous_data_formatter_file = os.path.join(model_dir, 'data_formatter.py')
    previous_model_metadata_file = os.path.join(
        model_dir, 'model_metadata.json')
    if not (os.path.isfile(previous_trained_model_file)):
        raise IOError(
            "Downloaded directory (%s) does not contain a 'model.keras' file.")

    # Copy data formatter to new output directory, if it exists
    if os.path.isfile(previous_data_formatter_file):
        new_formatter_path = os.path.join(output_dir, 'data_formatter.py')
        shutil.copy(previous_data_formatter_file, new_formatter_path)

    # Load previous model metadata and get a list of all datasets this model has been trained on
    with open(previous_model_metadata_file, 'r') as metadata_fp:
        previous_model_metadata = json.load(metadata_fp)
    previous_training_metadata = alc_utils.common.get_complete_training_metadata(
        previous_model_metadata)

    # Check type of dataset_storage_metadata and load appropriately
    if type(dataset_storage_metadata) is str:
        dataset_storage_metadata = json.loads(dataset_storage_metadata)
    elif type(dataset_storage_metadata) is dict:
        pass
    else:
        raise TypeError(
            "Specified dataset storage metadata has unexpected type (%s)" % str(type(dataset_storage_metadata)))

    # Compare dataset_storage_metadata with datasets previous network has already trained on
    # Remove any duplicates from dataset_storage_metadata
    # FIXME: This is an inefficient method. Should be improved
    pruned_dataset_metadata = []
    for data_info in dataset_storage_metadata:
        is_duplicate = False
        for previous_data_info in previous_training_metadata:
            if alc_utils.common.dataset_metadata_is_equal(data_info, previous_data_info):
                is_duplicate = True
                break

        # Only add this item to list if it is NOT a duplicate
        if not (is_duplicate):
            pruned_dataset_metadata.append(data_info)

    # Download each dataset in the pruned_dataset_metadata list
    result_code, downloaded_dirs = downloader.download(pruned_dataset_metadata)

    return downloaded_dirs, previous_trained_model_file, output_dir, pruned_dataset_metadata, previous_model_metadata


def setupRLTrainingAndUpdateParams(data, lParams, eParams):
    uparams = setupLECsAndUpdateParam(lParams, eParams)
    rldirname = ''
    foldername = ''
    print (' data type ', str(type(data)))
    if (data):
        print ('starting setup experiment')
        print ('filename ', data.val)
        datas, codes, destdir = getModelData(data.val)
        rldirname = destdir
        print (' folder name ', destdir)
        if (datas):
            (downloadedDirs, topdownloadfolder) = downloadData(datas)
            downloadurl = datas
            foldername = downloadedDirs[0]

    updatedparams = uparams
    if (rldirname):
        updatedparams['rl_model_dir'] = os.path.abspath(rldirname)
    if (foldername):
        updatedparams['rl_download_dir'] = os.path.abspath(foldername)

    return rldirname, foldername, updatedparams


def setupLECCode(leccode):
    # Make directory for storing output
    modeldir = os.path.join(os.getcwd(), 'LECCode')
    if not os.path.isdir(modeldir):
        os.makedirs(modeldir)

    # Assume we are using the old method with no DataFormatter class, then try newer method.
    lecdefinition = leccode
    lecdataformatter = None

    # Try loading leccode as JSON-formatted string
    # This will throw an exception if using the older method without a DataFormatter class
    try:
        code_dict = json.loads(leccode)
        lecdefinition = code_dict['leccode']
        lecdataformatter = code_dict['lecdataformatter']
    except ValueError as e:
        print ("Did not recognize LEC code string as a JSON object. Using older non-DataFormatter method instead.")
        pass

    dstpath = os.path.join(modeldir, 'LECModel.py')
    with open(dstpath, 'w') as outfile:
        outfile.write(lecdefinition)

    if (lecdataformatter):
        dstpath1 = os.path.join(modeldir, 'data_formatter.py')
        with open(dstpath1, 'w') as outfile:
            outfile.write(lecdataformatter)

    return dstpath


def setupLECsAndUpdateParam(lecParams, exptParams):
    if not lecParams:
        return exptParams
    lecpaths = setupLECs(lecParams)
    print ('lecpaths ', str(lecpaths))
    updatedExptParams = updateExptParamsWithLEC(exptParams, lecpaths)
    print ('updatedExptParams ', str(updatedExptParams))
    return updatedExptParams


def setupLECs(lecParams):
    lecpaths = {}
    if not lecParams:
        return lecpaths
    for l in lecParams.keys():
        data = lecParams[l]
        path = downloadModels(data)

        # FIXME: What was this supposed to do? Does not work right
        # lecpath = getdirwithin(path)
        # if lecpath:
        #     lecpaths[l] = lecpath

        lecpaths[l] = path
    return lecpaths


# FIXME: This needs to be revised. File names may change based on which ML library is being used, etc.
#        Can't assume these names are the only valid ones
def getdirwithin(folder):
    print ('folder ', str(folder))
    # Check if RL file exists and return path if so
    rlfile = os.path.join(folder, 'RLAgent.py')
    if os.path.exists(rlfile):
        return folder

    kerasfile = os.path.join(folder, 'model.keras')
    if os.path.exists(kerasfile):
        return folder

    kerasfile = os.path.join(folder, 'model.h5')
    if os.path.exists(kerasfile):
        return folder

    kerasfile = os.path.join(folder, 'model_weights.h5')
    if os.path.exists(kerasfile):
        return folder

    modelfile = os.path.join(folder, 'model.pkl')
    if os.path.exists(modelfile):
        return folder

    amfile = os.path.join(folder, 'assurancemonitor.pkl')
    if os.path.exists(amfile):
        return folder

    contents = os.listdir(folder)

    for l in contents:
        if (l.find('.ipynb_checkpoints') >= 0):
            continue

        fname = os.path.join(folder, l)
        if os.path.isdir(fname):
            return fname
    return ''


def updateExptParamsWithLEC(exptParams, lecpaths):
    if not lecpaths:
        return exptParams
    outputParams = exptParams.copy()
    keys = exptParams.keys()

    # FIXME: Currently this requires that the LEC parameter already exist in exptParams and the value will be updated.
    #   This means exptParams must have a placeholder (eg. {"lec_model_dir": "LEC_PERCEPTION"}) that gets overridden
    #   Can we just allow new parameters to be defined instead?
    for l in lecpaths.keys():
        outputParams[l] = lecpaths[l]

    return outputParams


def getModelData2(filename):
    import zipfile
    zip_ref = zipfile.ZipFile(filename, 'r')
    dd = str(os.getcwd())
    localdir = 'ModelData'
    destdir = os.path.join(dd, localdir)
    if not os.path.isdir(os.path.join(destdir)):
        os.makedirs(destdir)
    zip_ref.extractall(destdir)
    zip_ref.close()
    return destdir


def getModelData(filename):
    import zipfile
    zip_ref = zipfile.ZipFile(filename, 'r')
    dd = str(os.getcwd())
    localdir = 'ModelData'
    destdir = os.path.join(dd, localdir)
    if not os.path.isdir(os.path.join(destdir)):
        os.makedirs(destdir)
    zip_ref.extractall(destdir)
    zip_ref.close()
    modelfiles = []
    for (dirpath, dirnames, filenames) in os.walk(destdir):
        print ('file ', str(filenames))
        modelfiles.extend(filenames)
        break
    retdata = {}
    retcode = {}
    for mf in modelfiles:
        if mf.endswith('.pkl'):
            fname = os.path.join(destdir, mf)
            contents = getDataInfo(fname)  # pickle.load(open(fname, "rb"))
            retdata[mf] = contents
        if mf.endswith('.py'):
            fname = os.path.join(destdir, mf)
            retcode[mf] = fname
    return retdata, retcode, destdir


def downloadData(datas):
    downloaded_file_dirs = {}

    # dd = str(os.getcwd())
    dd = alc_utils.config.DOWNLOAD_DIRECTORY
    localdir = 'DownloadedData'
    destdir = os.path.join(dd, localdir)
    if not os.path.isdir(os.path.join(destdir)):
        os.makedirs(destdir)
    dirnames = []
    for d in datas.keys():
        datainfo = datas[d]
        downloaded_dirs = setupfiles(datainfo, destdir)
        for dirname in downloaded_dirs:
            dirnames.append(dirname)

    downloaded_file_dirs["datadirs"] = dirnames

    with open('datafiles.json', 'w') as outfile:
        json.dump(downloaded_file_dirs, outfile)

    return (downloaded_file_dirs, destdir)


def getDataInfo(fname):
    try:
        content = pickle.load(open(fname, "rb"))
        return content
    except:
        try:
            with open(fname) as content_file:
                content = json.load(content_file)
            return content
        except:
            print ('unable to read data: ', fname)
            return


def downloadModels(modelinfo):
    downloaded_file_dirs = {}
    # dd = str(os.getcwd())
    dd = alc_utils.config.DOWNLOAD_DIRECTORY
    localdir = 'DownloadedModel'
    destdir = os.path.join(dd, localdir)
    if not os.path.isdir(os.path.join(destdir)):
        os.makedirs(destdir)

    d = file_downloader.FileDownloader()
    dwndir = alc_utils.config.DOWNLOAD_DIRECTORY
    (result_code, downloaded_dir) = d.download(modelinfo, destdir)
    if (result_code == 0):
        return downloaded_dir[0]
    return ''


def setupfiles(inpdata, downloadfolder=''):
    contents = inpdata
    if (type(inpdata) is str):
        print ('data contents ', str(inpdata))
        contents = json.loads(inpdata)
    print ('contents ', str(contents))

    keys = contents.keys()

    dd = downloadfolder
    if (dd == ''):
        dd = os.getcwd()

    downloaded_dirs = []
    # import scenarios
    d = file_downloader.FileDownloader()
    dwndir = alc_utils.config.DOWNLOAD_DIRECTORY
    for k in keys:
        print (' k ', str(k))
        print (' contents[k]', str(contents[k]))

        (result_code, downloaded_dir) = d.download(contents[k], dd)
        if (result_code == 0):
            print ('download dir', str(downloaded_dir[0]))
            fullpath = os.path.abspath(downloaded_dir[0])
            print ('fullpath ', fullpath)
            downloaded_dirs.append(fullpath)
    return downloaded_dirs


def copyfilestojupyter(localdir):
    if not alc_utils.config.JUPYTER_WORK_DIR:
        mesg = 'JUPYTER_WORK_DIR environment variable is required'
        print(mesg)
        raise Exception(mesg)
    jupyterworkdir = alc_utils.config.JUPYTER_WORK_DIR
    destdir = os.path.join(jupyterworkdir, localdir)
    if not os.path.isdir(os.path.join(destdir)):
        srcdir = os.getcwd()
        try:
            shutil.copytree(srcdir, destdir)
        except OSError as exc:  # python >2.5
            if exc.errno == errno.ENOTDIR:
                shutil.copy(srcdir, destdir)
            else:
                raise
    return destdir


def copyfilestomatlabjupyter(localdir, modelinfo):
    if not alc_utils.config.JUPYTER_MATLAB_WORK_DIR:
        mesg = 'JUPYTER_MATLAB_WORK_DIR environment variable is required'
        print(mesg)
        raise Exception(mesg)
    jupyterworkdir = alc_utils.config.JUPYTER_MATLAB_WORK_DIR
    destdir = os.path.join(jupyterworkdir, localdir)
    if not os.path.isdir(os.path.join(destdir)):
        srcdir = os.getcwd()
        try:
            shutil.copytree(srcdir, destdir)
        except OSError as exc:  # python >2.5
            if exc.errno == errno.ENOTDIR:
                shutil.copy(srcdir, destdir)
            else:
                raise
    verification_file = os.path.join(
        destdir, 'ModelData', 'Verification', 'run.m')
    if (os.path.exists(verification_file)):
        data = ''
        with open(verification_file, 'r') as vfile:
            data = vfile.read()
        if (data):
            lecs = modelinfo.keys()
            for l in lecs:
                s = '<<' + l + '_DIRNAME>>'
                d = updateLECFolder(modelinfo[l])
                data = data.replace(s, d)
                s = '<<' + l + '_FILENAME>>'
                d = updateLECMeta(modelinfo[l])
                data = data.replace(s, d)
        with open(verification_file, 'w') as vfile:
            vfile.write(data)

    return destdir


def updateLECFolder(f):
    if os.path.exists(f):
        if (os.path.isdir(f)):
            f1 = os.path.join(f, 'weights')
            if (os.path.isdir(f1)):
                return f1
    return f


def updateLECMeta(f):
    if os.path.exists(f):
        if (os.path.isdir(f)):
            f1 = os.path.join(f, 'weights', 'checkpoint')
            if (os.path.exists(f1)):
                with open(f1, 'r') as cfile:
                    l = cfile.readline()
                lvals = l.split(':')[1]
                lvals = lvals.strip()
                lvals = lvals[1:-1]
                lvals += '.meta'
                f2 = os.path.join(f, 'weights', lvals)
                return f2
    return f


def createNotebook2(folderpath, localdir):
    if not alc_utils.config.JUPYTER_WORK_DIR:
        mesg = 'JUPYTER_WORK_DIR environment variable is required'
        print(mesg)
        raise Exception(mesg)
    jupyterworkdir = alc_utils.config.JUPYTER_WORK_DIR
    dstpath = os.path.join(folderpath, 'main.ipynb')
    srcpath = os.path.join(jupyterworkdir, 'initnb_interactive')
    shutil.copy(srcpath, dstpath)
    ret = {}
    ret["url"] = os.path.join(localdir, 'main.ipynb')
    return ret


def createInitNotebook2(folderpath, localdir):
    if not alc_utils.config.JUPYTER_WORK_DIR:
        mesg = 'JUPYTER_WORK_DIR environment variable is required'
        print(mesg)
        raise Exception(mesg)
    jupyterworkdir = alc_utils.config.JUPYTER_WORK_DIR
    dstpath = os.path.join(folderpath, 'main.ipynb')
    srcpath = os.path.join(jupyterworkdir, 'initnb')
    shutil.copy(srcpath, dstpath)
    ret = {}
    ret["url"] = os.path.join(localdir, 'main.ipynb')
    return ret


def writeResultScript(folderpath, srcpath, dstpath):
    ret = ''
    postprocesspath = ''
    postprocesspath1 = os.path.join(
        folderpath, '..', 'ModelData', 'PostProcess', 'PostProcess.py')
    postprocesspath2 = os.path.join(
        folderpath, '..', '..', 'ModelData', 'PostProcess', 'PostProcess.py')
    if (os.path.exists(postprocesspath1) and os.path.isfile(postprocesspath1)):
        postprocesspath = os.path.join(
            '..', 'ModelData', 'PostProcess', 'PostProcess.py')
    if (os.path.exists(postprocesspath2) and os.path.isfile(postprocesspath2)):
        postprocesspath = os.path.join(
            '..', '..', 'ModelData', 'PostProcess', 'PostProcess.py')
    if (postprocesspath):
        ret = '\"%matplotlib inline\\n\",'
        ret += '\"%load ' + postprocesspath + '\\n\"'
    else:
        ret = '\" \"'
    f1 = open(srcpath, 'r')
    contents = f1.read()
    contents = contents.replace('<POSTPROCESSSCRIPT>', ret)
    f2 = open(dstpath, 'w')
    f2.write(contents)
    f1.close()
    f2.close()
    return ret


def createResultNotebook2(folderpath):
    if not alc_utils.config.JUPYTER_WORK_DIR:
        mesg = 'JUPYTER_WORK_DIR environment variable is required'
        print(mesg)
        raise Exception(mesg)
    jupyterworkdir = alc_utils.config.JUPYTER_WORK_DIR
    dstpath = os.path.join(folderpath, 'result.ipynb')
    srcpath = os.path.join(jupyterworkdir, 'resultnb')
    # shutil.copy(srcpath, dstpath)
    import json
    dirinfo = [folderpath]
    filename = os.path.join(folderpath, 'workdir.json')
    with open(filename, 'w') as outfile:
        json.dump(dirinfo, outfile)
    l = len(jupyterworkdir)
    writeResultScript(folderpath, srcpath, dstpath)
    ret = dstpath[l + 1:]
    # ret = os.path.join(localdir, 'result.ipynb')
    return ret


def createInitNotebook(localdir, filename):
    jupyterworkdir = alc_utils.config.JUPYTER_WORK_DIR
    destdir = os.path.join(jupyterworkdir, localdir)
    if not os.path.isdir(os.path.join(destdir)):
        os.makedirs(destdir)
    dstpath = os.path.join(destdir, filename)
    srcpath = os.path.join(jupyterworkdir, 'initnb')
    shutil.copy(srcpath, dstpath)
    ret = {}
    ret["url"] = os.path.join(localdir, filename)
    return ret


def createResultNotebook(localdir, workdir):
    jupyterworkdir = alc_utils.config.JUPYTER_WORK_DIR
    destdir = os.path.join(jupyterworkdir, 'results', localdir)
    if not os.path.isdir(os.path.join(destdir)):
        os.makedirs(destdir)
    dstpath = os.path.join(destdir, 'result.ipynb')
    srcpath = os.path.join(jupyterworkdir, 'resultnb')
    shutil.copy(srcpath, dstpath)
    import json
    dirinfo = [workdir]
    filename = os.path.join(destdir, 'workdir.json')
    with open(filename, 'w') as outfile:
        json.dump(dirinfo, outfile)
    ret = os.path.join('results', localdir, 'result.ipynb')
    return ret


def createInitMatlabNotebook(localdir, filename):
    jupyterworkdir = alc_utils.config.JUPYTER_MATLAB_WORK_DIR
    destdir = os.path.join(jupyterworkdir, localdir)
    ldir = localdir

    if not os.path.isdir(os.path.join(destdir)):
        os.makedirs(destdir)

    verdir = os.path.join(destdir, 'ModelData', 'Verification')
    if (os.path.isdir(verdir)):
        destdir = verdir
        ldir = os.path.join(localdir, 'ModelData', 'Verification')

    dstpath = os.path.join(destdir, filename)
    srcpath = os.path.join(jupyterworkdir, 'initmatlabnb')
    shutil.copy(srcpath, dstpath)
    ret = {}
    ret["url"] = os.path.join(ldir, filename)
    ret["result_url"] = ret["url"]

    changeFolderMode(ldir, filename)
    return ret


def changeFolderMode(ldir, filename):
    jupyterworkdir = alc_utils.config.JUPYTER_MATLAB_WORK_DIR
    destdir = os.path.join(jupyterworkdir, ldir)
    os.chmod(destdir, 0o777)
    fname = os.path.join(destdir, filename)
    os.chmod(fname, 0o666)


def createNotebook(localdir, filename):
    jupyterworkdir = alc_utils.config.JUPYTER_WORK_DIR
    destdir = os.path.join(jupyterworkdir, localdir)
    if not os.path.isdir(os.path.join(destdir)):
        os.makedirs(destdir)
    dstpath = os.path.join(destdir, filename)
    srcpath = os.path.join(jupyterworkdir, 'initnb_interactive')
    shutil.copy(srcpath, dstpath)
    ret = {}
    ret["url"] = os.path.join(localdir, filename)
    return ret


def getModelModule(leccodepath):
    modelModule = ''
    modelfilename = leccodepath
    dirname = os.path.dirname(modelfilename)
    modelbasename = os.path.basename(modelfilename)
    (modulename, extension) = os.path.splitext(modelbasename)
    sys.path.append(dirname)
    modelModule = __import__(modulename)
    return modelModule


def createTrainingOutputDirectory():
    x = time.strftime("%Y_%m_%d_%H_%M_%S", time.gmtime())
    dirname = "TrainingResult_" + x
    try:
        os.mkdir(dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
        pass
    return os.path.abspath(dirname)


def createAMTrainingOutputDirectory():
    x = time.strftime("%Y_%m_%d_%H_%M_%S", time.gmtime())
    dirname = "AMTrainingResult_" + x
    fullpath = os.path.join(alc_utils.config.WORKING_DIRECTORY, dirname)

    try:
        os.mkdir(fullpath)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
        pass
    return os.path.abspath(fullpath)


def createExptBasePath(params, localdirname):
    fullpath = os.path.join(alc_utils.config.WORKING_DIRECTORY, localdirname)
    if not os.path.isdir(os.path.join(fullpath)):
        os.makedirs(fullpath)
    return fullpath, localdirname


class ArchiveData():
    def __init__(self, val):
        self.val = val


def dump(data, outfile):  # Create tar ball of custom_objects and model
    # Create the tmp directory
    return


def load(infile):
    d = ArchiveData(infile.name)
    return d


def printArchiveData(d):
    print ('uuvsim file name = ' + str(d.val))


def uploadFiles(folderpath, uploadDescription, pathPrefix, params):
    x1 = file_uploader.FileUploader()
    print ('description')
    print (uploadDescription)
    print ('path prefix')
    print (pathPrefix)
    print ('folder ')
    print (folderpath)
    # , description=uploadDescription, additional_prefix=pathPrefix)
    return x1.upload_with_params(folderpath, params)


def runUpload(params):
    foldertoupload = params['uploaddir']
    prefix = params.get('prefix', None)
    description = 'Manual upload'
    ret = uploadFiles(foldertoupload, description, prefix, params)
    return ret


def runUploadAMResults(params, base_name, foldertoupload):
    prefix = params.get('fs_path_prefix', 'amtraining')
    prefix_final = os.path.join(prefix, base_name)
    uploadparams = params
    uploadparams['fs_path_prefix'] = prefix_final
    x1 = file_uploader.FileUploader()
    ret = x1.upload_with_params(foldertoupload, uploadparams)
    # result_url = createResultNotebook2(foldertoupload)
    # ret['result_url']=result_url
    ret['exptParams'] = params
    return ret


def copyModelDir(model_dir, cur_dir, force=False):
    model_folder = getdirwithin(model_dir)
    base_name = os.path.basename(model_folder)
    if base_name != 'RLModel':
        base_name = 'SLModel'
    destdir = os.path.join(cur_dir, base_name)
    if (base_name == 'SLModel'):
        if (not os.path.exists(destdir)):
            os.makedirs(destdir)
        if (not force):
            return destdir
    srcdir = model_folder
    if (not srcdir or not os.path.exists(srcdir)):
        return destdir
    print('src dir : ' + srcdir)
    print('dst dir : ' + destdir)
    try:
        from distutils.dir_util import copy_tree
        copy_tree(srcdir, destdir)
    except OSError as exc:  # python >2.5
        if exc.errno == errno.ENOTDIR:
            shutil.copy(srcdir, destdir)
        else:
            raise
    return destdir
