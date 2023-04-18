#!/usr/bin/env python
import os
import shutil
import errno
import alc_utils.config as alc_config
from alc_utils import file_uploader
import sysid_utils

# FIXME: This file needs to be refactored/cleaned up


def setupExptVerification(params, modelinfo, plantinfo, evaldatainfo):
    localdir = os.path.join(
        params['project'], params['model'], str(params['datetime']))
    copyfilestomatlabjupyter(localdir, modelinfo, plantinfo, evaldatainfo, [])
    ret = createInitMatlabNotebook(localdir, 'main.ipynb')
    return ret


def copyfilestomatlabjupyter(localdir, modelinfo, plantinfo, evaldatainfo, traindatainfo):
    print 'in copyfilestomatlabjupyter'
    if not alc_config.JUPYTER_MATLAB_WORK_DIR:
        mesg = 'JUPYTER_MATLAB_WORK_DIR environment variable is required'
        print(mesg)
        raise Exception(mesg)
    jupyterworkdir = alc_config.JUPYTER_MATLAB_WORK_DIR
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
        destdir, 'ModelData', 'Verification', 'run_init.m')
    if (os.path.exists(verification_file)):
        data = ''
        with open(verification_file, 'r') as vfile:
            data = vfile.read()
        if (data):
            s = '<<HOME_DIR>>'
            d = os.path.join(destdir, 'ModelData', 'Verification')
            data = data.replace(s, getPathRelativeToALC(d))
        with open(verification_file, 'w') as vfile:
            vfile.write(data)

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
                data = data.replace(s, getPathRelativeToALC(d))
                s = '<<' + l + '_FILENAME>>'
                d = updateLECMeta(modelinfo[l])
                data = data.replace(s, getPathRelativeToALC(d))

            pd = ''
            s = '<<PLANTDATA>>'
            if (plantinfo):
                pd = updatePlantFolder(plantinfo)
            data = data.replace(s, getPathRelativeToALC(pd))

            ed = ''
            pipe = ''
            obs = ''
            s = '<<EVALDATA>>'
            if (evaldatainfo):
                ed, pipe, obs = updateEvalDataFolder(evaldatainfo)
            data = data.replace(s, getPathRelativeToALC(ed))
            s = '<<PIPEDATA>>'
            data = data.replace(s, getPathRelativeToALC(pipe))
            s = '<<OBSDATA>>'
            data = data.replace(s, getPathRelativeToALC(obs))

            s = '<<TRAINDATA>>'
            td = ''
            if (traindatainfo):
                td = updateTrainingDataFolder(traindatainfo)
            data = data.replace(s, td)

        with open(verification_file, 'w') as vfile:
            vfile.write(data)

    return destdir


'''
def updateLECFolder(f):
    if os.path.exists(f):
        if (os.path.isdir(f)):
            f1 = os.path.join(f,'weights','checkpoint')
            if (os.path.exists(f1)):
                return f1
    return f
'''


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


def updatePlantFolder(plantData):
    f = plantData[0]
    if os.path.exists(f):
        if (os.path.isdir(f)):
            f1 = os.path.join(f, 'sysID.mat')
            if (os.path.exists(f1)):
                return f1
            f2 = os.path.join(f, '..', 'ModeData', 'Verification', 'sysID.mat')
            if (os.path.exists(f2)):
                return f2
    return ''


def updateTrainingDataFolder(folders):
    ret = ''
    for f in folders:
        if os.path.exists(f):
            f1_10.f1_utils.routines.sysid_utils.generateSysIDData([f])
            x = f1_10.f1_utils.routines.sysid_utils.getallsysidfiles(f)
            for fname in x:
                x1 = getPathRelativeToALC(fname)
                if (ret):
                    ret += ', '
                ret += '\'' + x1 + '\''

    return ret


def updateEvalDataFolder(folders):
    f = folders[0]
    retx = ''
    rety = ''
    retz = ''
    if os.path.exists(f):
        f1_10.f1_utils.routines.sysid_utils.generateSysIDData([f])
        x = f1_10.f1_utils.routines.sysid_utils.getallsysidfiles(f)
        y = f1_10.f1_utils.routines.sysid_utils.getallpipefiles(f)
        z = f1_10.f1_utils.routines.sysid_utils.getallobstaclefiles(f)
        if x:
            retx = x[0]
        if y:
            rety = y[0]
        if z:
            retz = z[0]

    return retx, rety, retz


def getPathRelativeToALC(f):
    if f == '':
        return f
    x = f.find('jupyter')
    if x == -1:
        return f

    f1 = f[x:]
    return f1


def runUploadResults(foldertoupload):
    uploadparams = {}
    uploadparams['upload'] = False
    uploadparams['fs_path_prefix'] = 'ver_results'
    x1 = file_uploader.FileUploader()
    ret = x1.upload_with_params(foldertoupload, uploadparams)
    return ret


def createInitMatlabNotebook(localdir, filename):
    print 'in createInitMatlabNotebook'
    jupyterworkdir = alc_config.JUPYTER_MATLAB_WORK_DIR
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

    ret = runUploadResults(destdir)
    ret["url"] = os.path.join(ldir, filename)
    ret["result_url"] = ret["url"]

    changeFolderMode(ldir, filename)
    return ret


def changeFolderMode(ldir, filename):
    jupyterworkdir = alc_config.JUPYTER_MATLAB_WORK_DIR
    destdir = os.path.join(jupyterworkdir, ldir)
    os.chmod(destdir, 0o777)
    fname = os.path.join(destdir, filename)
    os.chmod(fname, 0o666)


def setupExptSysID(params, modelinfo, plantinfo, traininfo):
    print ' reached sysid 1'
    localdir = os.path.join(
        params['project'], params['model'], str(params['datetime']))
    print ' reached sysid 2'
    copyfilestomatlabjupyter(localdir, modelinfo, plantinfo, [], traininfo)
    print ' reached sysid 3'
    ret = createInitMatlabNotebook(localdir, 'main.ipynb')
    print ' reached sysid 4'
    return ret


def setupValidation(params, modelinfo, plantinfo, evalinfo):
    localdir = os.path.join(
        params['project'], params['model'], str(params['datetime']))
    copyfilestomatlabjupyter(localdir, modelinfo, plantinfo, evalinfo, [])
    ret = createInitMatlabNotebook(localdir, 'main.ipynb')
    return ret
