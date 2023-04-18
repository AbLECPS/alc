import sys
import logging
from pathlib import Path
from shutil import move
from subprocess import run, Popen
from time import sleep


logger = logging.Logger("WORKFLOW_LOOP")
logger.addHandler(logging.StreamHandler())

python_path_env_var_name = "PYTHONPATH"
python_executable_name = "python3.6"
wrapper_name = "wrapper_script.py"

python_dir_path = Path("/alc/webgme/automate/gradle/src/main/python")

if not python_dir_path.is_dir():
    message = "ERROR: directory \"{0}\" must exist".format(python_dir_path)
    logger.error(message)
    print(message, file=sys.stderr, flush=True)
    sys.exit(1)


wrapper_path = Path(python_dir_path, wrapper_name)


def get_wrapper_command_args(local_script_path, config_path):
    args_list = [
        str(python_executable_name),
        str(local_script_path),
        str(config_path)
    ]

    return args_list


process_dir_path = Path("/alc/workflows/config")
process_dir_path.mkdir(parents=True, exist_ok=True)

process_temp_dir_path = Path("/alc/workflows/config_temp")
process_temp_dir_path.mkdir(parents=True, exist_ok=True)

popen_list = []

while True:
    config_file_list = list(process_dir_path.glob('*'))

    if len(config_file_list) > 0:

        config_file_orig_path = config_file_list[0]
        config_file_temp_path = Path(process_temp_dir_path, config_file_orig_path.name)
        move(str(config_file_orig_path), str(config_file_temp_path))

        popen_object = Popen(get_wrapper_command_args(wrapper_path, config_file_temp_path))

        popen_list.append(popen_object)

    new_popen_list = []
    for popen_object in popen_list:

        return_code = popen_object.poll()
        if return_code is None:
            new_popen_list.append(popen_object)

    popen_list = new_popen_list

    sleep(1)
