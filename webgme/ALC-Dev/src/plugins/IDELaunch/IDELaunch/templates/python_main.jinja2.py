import sys
import os
from alc_utils.routines.{{job_type|string}} import ActivityInterpreter

import json

workingFolderPath = os.getcwd()
resultFilePath = os.path.join(workingFolderPath,'result_metadata.json')


ai = ActivityInterpreter.ActivityInterpreter(workingFolderPath)
ret = ai.run()

if ret:
    with open(resultFilePath, 'w') as f:
        json.dump(ret, f)
