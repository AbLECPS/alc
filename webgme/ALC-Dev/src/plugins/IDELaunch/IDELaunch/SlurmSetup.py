import math
import json
import os
import stat
from pathlib import Path
import alc_utils.slurm_executor as slurm_executor
from alc_utils.slurm_executor import WebGMEKeys
from alc_utils.update_job_status_daemon import Keys as UpdateKeys
import time

# Useful macros
SECONDS_PER_MINUTE = 60.0
SLURM_GRACE_TIME_MIN = 5

bash_command = "bash"
bash_script_name = "run.sh"
activity_setup_job_type = 'LaunchIDE'
slurm_job_params_filename = "slurm_params.json"


def generate_slurm_job_params(folder_path,slurm_job_params):
    # write slurm params so that they can be used from workflow
    with Path(folder_path, slurm_job_params_filename).open("w") as json_fp:
        json.dump(slurm_job_params, json_fp, indent=4, sort_keys=True)


def setup_job(
        plugin_object,
        name,
        seconds_since_epoch,
        timeout_param,
        job_type,
        deploy_job=True,
        result_folder=None,
        logger=None
):
    # Get any user configured SLURM settings (make sure all dict keys are lower case)
    # Handle special case parameters if User has not set them explicitly
    # Make sure job type is included in slurm params since this is used to determine execution defaults
    project_info = plugin_object.project.get_project_info()
    slurm_job_params = {
        WebGMEKeys.job_type_key: activity_setup_job_type+'-'+job_type,
        WebGMEKeys.project_owner_key: project_info[WebGMEKeys.project_owner_key],
        WebGMEKeys.project_name_key: project_info[WebGMEKeys.project_name_key],
        WebGMEKeys.result_dir_key: str(result_folder),
        WebGMEKeys.command_for_srun_key: "{0} {1}".format(
            bash_command, bash_script_name
        )
    }

    # FOR TESTING (WITH PORT 8000)
    if hasattr(plugin_object, 'slurm_params') and isinstance(plugin_object.slurm_params, dict):
        slurm_job_params.update(plugin_object.slurm_params)

    if slurm_job_params.get(WebGMEKeys.job_name_key, None) is None:
        slurm_job_params[WebGMEKeys.job_name_key] = name
    
    if slurm_job_params.get(WebGMEKeys.time_limit_key, None) is None and timeout_param:
        slurm_job_params[WebGMEKeys.time_limit_key] = \
            int(math.ceil(timeout_param / SECONDS_PER_MINUTE) + SLURM_GRACE_TIME_MIN)

    generate_slurm_job_params(str(result_folder),slurm_job_params)
    
    if deploy_job:

        slurm_executor.slurm_deploy_job(str(result_folder), job_params=slurm_job_params)
    else:
        print('job is not deployed')

    return slurm_job_params


