#!/usr/bin/env python
# This contains python code for executing ALC jobs
# Based on the jobtype, it invokes the appropriate functions in alc_utils.
from __future__ import print_function

import os
import json
import sys
from alc_utils import common as alc_common


# method invoked to run the jobs
def run(params):
    # execute based on job type
    ret = {}

    from alc_utils.routines import nodeEnv
    from alc_utils.routines import verDep

    depDict = verDep.dep_dict

    from alc_utils import execution_runner
    eParams = json.loads(params)
    depDict["base_dir"] = eParams.get("specific_notebook_directory", None)
    if (not depDict["base_dir"]):
        return ret

    print("*********** STARTING EXPERIMENT IN DOCKER *************")
    configfilename = os.path.join(depDict['base_dir'], 'config.json')
    with open(configfilename, 'w') as outfile:
        json.dump(depDict, outfile)
    runner = execution_runner.ExecutionRunner(configfilename)
    result, resultdir = runner.run()

    import alc_utils
    if (result == 0):
        ret["exptParams"] = eParams
        ret["directory"] = depDict["base_dir"]
        jupyterworkdir = alc_utils.config.JUPYTER_WORK_DIR
        result_file = os.path.join(depDict['base_dir'], 'robustness.ipynb')
        result_url = result_file[len(jupyterworkdir)+1:]
        ret["result_url"] = 'ipython/notebooks/'+result_url

    return ret
