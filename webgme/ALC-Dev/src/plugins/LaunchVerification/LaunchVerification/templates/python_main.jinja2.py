from alc_utils.routines.ALCVerificationJobRunner import run
import json

paramFilePath = {{"\'" + param_file_path|string + "\'"}}
resultFilePath = {{"\'" + result_file_path|string + "\'"}}
params='{}'
with open(paramFilePath, 'r') as pfile:
    params = pfile.read()
ret = run(params)
if ret:
    with open(resultFilePath, 'w') as f:
        json.dump(ret, f, indent=4, sort_keys=True)
