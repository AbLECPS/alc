import math
import json
import os
from pathlib import Path
import TemplateManager
import alc_utils.slurm_executor
import alc_utils.common as alc_common

# Useful macros
SECONDS_PER_MINUTE = 60.0
SLURM_GRACE_TIME_MIN = 5

command_for_srun_key = "command_for_srun"
config_task_key = "task"
config_relative_path_key = "relative_path"

project_name_key = "name"
project_owner_key = "owner"

result_dir_key = "result_dir"

slurm_jobtype_key = "jobtype"
slurm_job_name_key = "job_name"
slurm_time_key = "time"

task_destination_key = "dst"
task_name_key = "name"
task_file_name_key = "task_file_name"
task_one_key = "1"
task_result_folder_key = "result_folder"
task_source_key = "src"

jupyter_dir_name = 'jupyter'
main_py_file_name = 'main.py'
result_metadata_file_name = "result_metadata.json"
update_model_config_file_name = "update_model_config.json"
update_model_sh_file_name = "update_model.sh"
update_model_task_file_name = "update_model_task.json"

verification_robustness_setup_job_type = 'VERIFICATION_ROBUSTNESS_SETUP'

update_result_metadata_task_name = "Update_Result_Metadata"

alc_work_env_var_name = "ALC_WORK"
python_command = "python"
python_script_name = "main.py"


def setup_job(project_info, exec_name, result_node_path, slurm_params, timeout_param, folder_path, param_filename):
    # Get any user configured SLURM settings (make sure all dict keys are lower case)
    # Handle special case parameters if User has not set them explicitly
    # Make sure job type is included in slurm params since this is used to determine execution defaults
    slurm_job_params = alc_common.dict_convert_key_case(slurm_params, "lower")
    slurm_job_params[slurm_jobtype_key] = verification_robustness_setup_job_type
    slurm_job_params[project_owner_key] = project_info[project_owner_key]
    slurm_job_params[project_name_key] = project_info[project_name_key]
    slurm_job_params[result_dir_key] = str(folder_path)
    slurm_job_params[task_file_name_key] = update_model_config_file_name
    slurm_job_params[command_for_srun_key] = "{0} {1}".format(python_command, python_script_name)

    if slurm_job_params.get(slurm_job_name_key, None) is None:
        slurm_job_params[slurm_job_name_key] = exec_name
    if slurm_job_params.get(slurm_time_key, None) is None:
        if timeout_param:
            slurm_job_params[slurm_time_key] = int(math.ceil(timeout_param / SECONDS_PER_MINUTE) + SLURM_GRACE_TIME_MIN)

    generate_files(folder_path, param_filename, result_node_path)
    alc_utils.slurm_executor.slurm_deploy_job(str(folder_path), job_params=slurm_job_params)
    return slurm_job_params
    

def generate_files(folder_path, param_filename, result_node_path):
    alc_wkdir = Path(os.getenv(alc_work_env_var_name, ''))
    if not alc_wkdir:
        raise RuntimeError('Environment variable {0} is unknown or not set'.format(alc_work_env_var_name))
    if not alc_wkdir.is_dir():
        raise RuntimeError('{0}: {1} does not exist'.format(alc_work_env_var_name, alc_wkdir))
    jupyter_dir = Path(alc_wkdir, jupyter_dir_name)
    if not jupyter_dir.is_dir():
        raise RuntimeError('{0} directory : {1} does not exist in {2}'.format(
            jupyter_dir_name, jupyter_dir, alc_work_env_var_name
        ))

    param_file_path = Path(folder_path, param_filename)
    result_dir = folder_path
    relative_result_dir = result_dir
    if jupyter_dir_name in str(result_dir):
        result_dir_parts = result_dir.parts
        jupyter_pos = result_dir_parts.index(jupyter_dir_name)
        relative_result_dir = Path(*result_dir_parts[jupyter_pos:])
    generate_main_file(param_file_path, result_dir)
    generate_result_update_files(result_dir, relative_result_dir, result_node_path)


def generate_main_file(param_file_path, result_dir):
    # Fill out top-level launch template
    result_file_path = Path(result_dir, result_metadata_file_name)
    main_template = TemplateManager.python_main_template
    code = main_template.render(
        param_file_path=str(param_file_path),
        result_file_path=str(result_file_path)
    )

    # Write python main code to path
    main_path = Path(result_dir, main_py_file_name)
    with main_path.open("w") as f:
        f.write(code)


def generate_result_update_files(result_dir, relative_result_dir, result_node_path):

    update_model_config_path = Path(result_dir, update_model_config_file_name)
    update_model_task_path = Path(result_dir, update_model_task_file_name)

    relative_update_model_task_path = Path(relative_result_dir, update_model_task_file_name)

    config_json = {
        config_task_key: str(relative_update_model_task_path),
        config_relative_path_key: True
    }
    with update_model_config_path.open('w') as f:
        json.dump(config_json, f, indent=4, sort_keys=True)

    task_json = {
        task_one_key: {
            task_name_key: update_result_metadata_task_name,
            task_source_key: [""],
            task_destination_key: [str(result_node_path)],
            task_result_folder_key: str(relative_result_dir)
        }
    }
    with update_model_task_path.open('w') as f:
        json.dump(task_json, f, indent=4, sort_keys=True)
