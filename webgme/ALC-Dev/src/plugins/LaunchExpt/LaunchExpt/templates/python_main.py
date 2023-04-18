import sys
import os
sys.path.append(os.path.join(sys.path[0],'../../'))
from utils.ALCJob import run
import json

paramFilePath = {{"\'" + param_file_path|string + "\'"}}
resultFilePath = {{"\'" + result_file_path|string + "\'"}}
params='{}'
with open(paramFilePath, 'r') as pfile:
    params = pfile.read()
ret = run('', {{ setup_jupyter_nb|string }}, params,{{camp_count}})
if ret:
    with open(resultFilePath, 'w') as f:
        json.dump(ret, f)
