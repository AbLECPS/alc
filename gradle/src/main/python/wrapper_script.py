import sys
import os
import copy
import json
import logging
import argparse
import traceback
from pathlib import Path
import ConfigKeys
import DiagnosticKeys
import ScriptFileNames
from ProjectParameters import ProjectParameters
from WorkflowReportTask import WorkflowReportTask


logger = logging.Logger("SCRIPT_WRAPPER")
logger.addHandler(logging.StreamHandler())

python_path_env_var_name = "PYTHONPATH"
repo_home_env_var_name = "REPO_HOME"
config_file_option_string = "config_file"
python_executable_name = "python3.6"

report_task_dir_name = "execute"

start_message = "Workflow executing ..."
success_message = "Workflow executed successfully."
failure_message = "Workflow failed to execute."


def exec_path(python_script_path, sys_argv=None, environ=None, cwd=None, stdin=None, stdout=None, stderr=None):
    save_sys_argv = sys.argv
    if sys_argv is not None:
        sys.argv = sys_argv

    save_environ = os.environ
    if environ is not None:
        os.environ = environ

    save_cwd = os.getcwd()
    if cwd is not None:
        os.chdir(str(cwd))

    save_stdin = sys.stdin
    if stdin is not None:
        sys.stdin = stdin

    save_stdout = sys.stdout
    if stdout is not None:
        sys.stdout = stdout

    save_stderr = sys.stderr
    if stderr is not None:
        sys.stderr = stderr

    try:
        exec(python_script_path.read_text("utf-8"), globals())
    except SystemExit as e:
        return int(str(e)), None, None
    except BaseException as e:
        return 1, e, traceback.format_exc()
    finally:
        sys.argv = save_sys_argv
        os.environ = save_environ
        os.chdir(save_cwd)
        sys.stdin = save_stdin
        sys.stdout = save_stdout
        sys.stderr = save_stderr

    return 0, None, None


python3_dir_path = Path("/alc/webgme/automate/gradle/src/main/pythonep")
# python3_dir_path = Path("/home/ninehs/ALC/alc/gradle/src/main/python3")

if not python3_dir_path.is_dir():
    message = "ERROR: directory \"{0}\" must exist".format(python3_dir_path)
    logger.error(message)
    print(message, file=sys.stderr, flush=True)
    sys.exit(1)

generator_path = Path(python3_dir_path, ScriptFileNames.generator_name)

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("config_file", type=str)

arguments = argument_parser.parse_args()

config_file_orig_path = Path(arguments.config_file)

with config_file_orig_path.open() as config_file_fp:
    config_json = json.load(config_file_fp)

working_dir = config_json.get(ConfigKeys.working_dir_key, None)

if working_dir is None:
    message = "ERROR: config file must contain \"{0}\" as key, with working directory as value".format(
        ConfigKeys.working_dir_key
    )
    logger.error(message)
    print(message, file=sys.stderr, flush=True)
    sys.exit(1)

working_dir_path = Path(working_dir)
working_dir_path.mkdir(parents=True, exist_ok=True)


generator_io_dir_path = Path(working_dir_path, ScriptFileNames.generator_io_dir_name)
generator_io_dir_path.mkdir(parents=True, exist_ok=True)

generator_stdout_path = Path(generator_io_dir_path, ScriptFileNames.generator_stdout_file_name)
generator_stderr_path = Path(generator_io_dir_path, ScriptFileNames.generator_stderr_file_name)
generator_exceptions_path = Path(generator_io_dir_path, ScriptFileNames.generator_exceptions_file_name)

script_dir_path = Path(working_dir_path, ScriptFileNames.script_dir_name)
script_dir_path.mkdir(parents=True, exist_ok=True)

script_path = Path(script_dir_path, ScriptFileNames.script_file_name)
script_stderr_path = Path(script_dir_path, ScriptFileNames.script_stderr_file_name)
script_stdout_path = Path(script_dir_path, ScriptFileNames.script_stdout_file_name)
failing_task_path_file_path = Path(script_dir_path, ScriptFileNames.failing_task_path_file_name)
exceptions_file_path = Path(script_dir_path, ScriptFileNames.exceptions_file_name)

new_environ = copy.deepcopy(os.environ)

old_pythonpath = new_environ.get(python_path_env_var_name)
if old_pythonpath:
    new_environ[python_path_env_var_name] = "{0}/script:{1}:{2}".format(
        working_dir_path, python3_dir_path, old_pythonpath
    )
else:
    new_environ[python_path_env_var_name] = "{0}/script:{1}".format(working_dir_path, python3_dir_path)

repo_dir = config_json.get(ConfigKeys.repo_home_key, None)
if (repo_dir):
    new_environ[repo_home_env_var_name] = repo_dir

config_file_path = Path(generator_io_dir_path, "config.json")
with config_file_path.open("w") as config_file_fp:
    json.dump(config_json, config_file_fp, indent=4, sort_keys=True)
config_file_orig_path.unlink()


with generator_stdout_path.open("w") as generator_stdout_fp, generator_stderr_path.open("w") as generator_stderr_fp:
    generation_exit_status, generation_exception, generation_traceback = exec_path(
        generator_path,
        sys_argv=[str(generator_path), str(config_file_path)],
        environ=new_environ,
        stdout=generator_stdout_fp,
        stderr=generator_stderr_fp,
        cwd=str(working_dir_path)
    )

if generation_exit_status != 0:
    with generator_exceptions_path.open("w") as exceptions_fp:
        print(
            "exit_status = {0}\nException:\n{1}\n".format(
                generation_exit_status, generation_traceback
            ),
            file=sys.stderr
        )
    sys.exit(1)


project_name = config_json.get(ConfigKeys.project_name_key, "ERROR GETTING PROJECT NAME")
workflow_name = config_json.get(ConfigKeys.workflow_name_key, "ERROR GETTING WORKFLOW NAME")
logger.info("Executing workflow \"{0}\" of project \"{1}\"".format(workflow_name, project_name))


with script_stdout_path.open("w") as script_stdout_fp, script_stderr_path.open("w") as script_stderr_fp:
    execution_exit_status, execution_exception, execution_traceback = exec_path(
        script_path,
        sys_argv=[str(script_path)],
        environ=new_environ,
        stdout=script_stdout_fp,
        stderr=script_stderr_fp,
        cwd=str(working_dir_path)
    )

if execution_exception:
    ProjectParameters.set_generic_active_node(config_json.get(ConfigKeys.generic_active_node_key))
    ProjectParameters.set_owner(config_json.get(ConfigKeys.owner_key))
    ProjectParameters.set_project_name(config_json.get(ConfigKeys.project_name_key))
    ProjectParameters.set_status_node(config_json.get(ConfigKeys.status_path_key))

    report_task_dir_path = Path(working_dir_path, report_task_dir_name)

    workflow_report_task = WorkflowReportTask()
    extra_data = {
        DiagnosticKeys.script_file_key: str(script_path.absolute()),
        DiagnosticKeys.stderr_file_key: str(script_stderr_path.absolute()),
        DiagnosticKeys.stdout_file_key: str(script_stdout_path.absolute()),
        DiagnosticKeys.exceptions_file_key: str(exceptions_file_path.absolute())
    }
    workflow_report_task.execute_start(start_message, extra_data)

    with exceptions_file_path.open("w") as exceptions_fp:
        print(
            "exit_status = {0}\nExceptions:\n{1}\n".format(execution_exit_status, execution_traceback),
            file=exceptions_fp
        )

    workflow_report_task.execute_finish(False, failure_message)
