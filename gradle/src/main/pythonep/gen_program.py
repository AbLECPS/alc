import json
import re
import argparse
from pathlib import Path
import ConfigKeys
import ScriptFileNames
import TemplateManager


root_job_var = "root_job"

alc_job_type = "ALC_Job"
branch_type = "Branch"
launch_activity_type = "LaunchActivity"
launch_experiment_type = "LaunchExperiment"
loop_type = "Loop"
transform_job_type = "Transform"

branch_format_string = "branch_{0}"
module_format_string = "Module_{0}"
module_file_format_string = "{0}.py"
parameter_manager_format_string = "parameter_manager_{0}"
threaded_loop_format_string = "threaded_loop_{0}"
transform_format_string = "transform_{0}"
while_loop_format_string = "while_loop_{0}"
workflow_job_format_string = "workflow_job_{0}"

module_name_list = []
function_map = {}

configuration_argument_names = [
    (ConfigKeys.operation_config_key, ConfigKeys.operation_key),
    (ConfigKeys.source_list_config_key, ConfigKeys.source_list_key),
    (ConfigKeys.destination_list_config_key, ConfigKeys.destination_list_key),
    (ConfigKeys.destination_lec_config_key, ConfigKeys.destination_lec_key)
]

unique_id = 0

modules_dir_path = Path(".")  # GIVE Path VALUE INSTEAD OF None TO QUIET IDE WARNING


def get_unique_num():
    global unique_id
    retval = unique_id
    unique_id += 1
    return retval


def get_function_name(function_text):

    if function_text[0] == "'":
        function_text = function_text[1:]
    if function_text[-1] == "'":
        function_text = function_text[:-1]

    function_text = "{0}\n".format(function_text.strip())

    match = re.search(r"def\s*([^\s(]+)\s*\(", function_text)
    function_name = match.group(1)

    module_name = module_format_string.format(get_unique_num())
    module_file_name = module_file_format_string.format(module_name)
    module_file_path = Path(modules_dir_path, module_file_name)
    with module_file_path.open("w") as module_fp:
        print(function_text, file=module_fp)

    full_function_name = "{0}.{1}".format(module_name, function_name)
    return full_function_name, module_name


def get_use_parameters_list(job):
    use_parameters_list = ["all"]
    if ConfigKeys.use_parameters_key in job:
        use_parameters_list = job.get(ConfigKeys.use_parameters_key)
        if not isinstance(use_parameters_list, list):
            use_parameters_list = []

    return use_parameters_list


def get_launch_activity_job(job, parent_job_var_name, prototype_map):

    call_list = []

    for input_name, output_path_list in job.get(ConfigKeys.inputs_key, {}).items():
        call_list.append(TemplateManager.add_input_template.render(input_name=input_name, input_path=output_path_list))

    if ConfigKeys.parameter_update_script_key in job:
        use_parameters_list = get_use_parameters_list(job)

        function_name = "workflow_data.get_parent_parameter_updates"
        call_list.append(TemplateManager.add_parameter_updates_template.render(
            update_spec=TemplateManager.lambda_template.render(
                function_name=function_name,
                arg_list=[use_parameters_list]
            )
        ))

    use_parameters_list = get_use_parameters_list(job)
    if len(use_parameters_list) > 0:
        call_list.append(TemplateManager.add_parameter_filter_template.render(
            filter_spec=use_parameters_list
        ))

    previous_job_name_list = job.get(ConfigKeys.previous_jobs_key, [])
    next_job_name_list = job.get(ConfigKeys.next_jobs_true_key, [])

    activity_name = job.get("activities_name")
    return TemplateManager.add_launch_activity_job_trivial_template.render(
        job_name=job.get("job_name"),
        activity_name=activity_name,
        activity_node=prototype_map.get(activity_name, ""),
        previous_job_name_list=previous_job_name_list,
        next_job_name_list=next_job_name_list,
        parent_job=parent_job_var_name
    ) if len(call_list) == 0 else TemplateManager.add_launch_activity_job_template.render(
        var_name=workflow_job_format_string.format(get_unique_num()),
        call_list=call_list,
        job_name=job.get("job_name"),
        activity_name=activity_name,
        activity_node=prototype_map.get(activity_name, ""),
        previous_job_name_list=previous_job_name_list,
        next_job_name_list=next_job_name_list,
        parent_job=parent_job_var_name
    )


def get_launch_experiment_job(job, parent_job_var_name):
    call_list = []
    for init in job.get(ConfigKeys.init_list_key, []):
        argument_list = []
        for argument_config_name, argument_name in configuration_argument_names:
            if argument_config_name in init:
                argument_value = init.get(argument_config_name)
                if isinstance(argument_value, str):
                    if argument_value == "":
                        continue
                    argument_value = '"' + argument_value + '"'
                argument_list.append(TemplateManager.named_argument_template.render(
                    argument_name=argument_name, argument_value=argument_value
                ))
        call_list.append(TemplateManager.add_configuration_template.render(named_argument_list=argument_list))

    for local_input in job.get(ConfigKeys.inputs_key, {}):
        argument_list = []
        for argument_config_name, argument_name in configuration_argument_names:
            if argument_config_name in local_input:
                argument_value = local_input.get(argument_config_name)
                if isinstance(argument_value, str):
                    if argument_value == "":
                        continue
                    argument_value = '"' + argument_value + '"'
                argument_list.append(TemplateManager.named_argument_template.render(
                    argument_name=argument_name, argument_value=argument_value
                ))
        if ConfigKeys.input_job_key in local_input and ConfigKeys.source_list_key not in local_input:
            input_job_path = local_input.get(ConfigKeys.input_job_key)

            source_list_lambda = TemplateManager.lambda_template.render(
                function_name="workflow_data.get_value",
                arg_list=[input_job_path, "[ResultAux.get_list_paths]"]
            )
            argument_list.append(TemplateManager.named_argument_template.render(
                argument_name="source_list", argument_value=source_list_lambda
            ))
        call_list.append(TemplateManager.add_configuration_template.render(named_argument_list=argument_list))

    for activity_name, activity_path in job.get(ConfigKeys.activities_map_key).items():
        call_list.append(
            TemplateManager.add_activity_template.render(activity_name=activity_name, activity_path=activity_path)
        )

    if ConfigKeys.parameter_update_script_key in job:
        use_parameters_list = get_use_parameters_list(job)

        function_name = "workflow_data.get_parent_parameter_updates"
        call_list.append(TemplateManager.add_parameter_updates_template.render(
            update_spec=TemplateManager.lambda_template.render(
                function_name=function_name,
                arg_list=[use_parameters_list]
            )
        ))

    use_parameters_list = get_use_parameters_list(job)
    if len(use_parameters_list) > 0:
        call_list.append(TemplateManager.add_parameter_filter_template.render(
            filter_spec=use_parameters_list
        ))

    previous_job_name_list = job.get(ConfigKeys.previous_jobs_key, [])
    next_job_name_list = job.get(ConfigKeys.next_jobs_true_key, [])

    return TemplateManager.add_launch_experiment_job_template.render(
        var_name=workflow_job_format_string.format(get_unique_num()),
        call_list=call_list,
        job_name=job.get("job_name"),
        previous_job_name_list=previous_job_name_list,
        next_job_name_list=next_job_name_list,
        parent_job=parent_job_var_name
    )


def get_transform(job, parent_job_var_name, parent_path_name_tuple):
    job_name = job.get(ConfigKeys.job_name_key)

    function_name, module_name = get_function_name(job.get(ConfigKeys.script_key))
    module_name_list.append(module_name)
    function_map[parent_path_name_tuple + (job_name,)] = function_name

    previous_job_name_list = job.get(ConfigKeys.previous_jobs_key, [])
    next_job_name_list = job.get(ConfigKeys.next_jobs_true_key, [])

    call_list = []
    for input_name, output_path_list in job.get(ConfigKeys.inputs_key, {}).items():
        call_list.append(TemplateManager.add_input_template.render(input_name=input_name, input_path=output_path_list))

    return TemplateManager.add_transform_trivial_template.render(
        job_name=job_name,
        previous_job_name_list=previous_job_name_list,
        next_job_name_list=next_job_name_list,
        parent_job=parent_job_var_name,
        function=function_name
    ) if len(call_list) == 0 else TemplateManager.add_transform_template.render(
        var_name=transform_format_string.format(get_unique_num()),
        call_list=call_list,
        job_name=job_name,
        previous_job_name_list=previous_job_name_list,
        next_job_name_list=next_job_name_list,
        parent_job=parent_job_var_name,
        function=function_name
    )


def get_branch(job, parent_job_var_name):

    function_name, function_text = get_function_name(job.get(ConfigKeys.script_key))
    module_name_list.append(function_text)

    job_name = job.get(ConfigKeys.job_name_key)

    previous_job_name_list = job.get(ConfigKeys.previous_jobs_key, [])

    true_job_name_list = job.get(ConfigKeys.next_jobs_true_key)

    false_job_name_list = job.get(ConfigKeys.next_jobs_false_key)

    call_list = []
    for input_name, output_path_list in job.get(ConfigKeys.inputs_key, {}).items():
        call_list.append(TemplateManager.add_input_template.render(input_name=input_name, input_path=output_path_list))

    return TemplateManager.add_branch_point_trivial_template.render(
        parent_job=parent_job_var_name,
        job_name=job_name,
        previous_job_name_list=previous_job_name_list,
        false_job_name_list=false_job_name_list,
        condition=function_name,
        true_job_name_list=true_job_name_list
    ) if len(call_list) == 0 else TemplateManager.add_branch_point_template.render(
        call_list=call_list,
        var_name=branch_format_string.format(get_unique_num()),
        parent_job=parent_job_var_name,
        job_name=job_name,
        previous_job_name_list=previous_job_name_list,
        false_job_name_list=false_job_name_list,
        condition=function_name,
        true_job_name_list=true_job_name_list
    )


def get_parameterized_loop_job(job, parent_job_var_name, parent_path_name_tuple, prototype_map, job_map):
    loop_job_list = job.get(ConfigKeys.loop_jobs_key)

    parameter_manager_name = parameter_manager_format_string.format(get_unique_num())

    loop_name = job.get(ConfigKeys.job_name_key)
    loop_var_name = threaded_loop_format_string.format(get_unique_num())

    previous_job_name_list = job.get(ConfigKeys.previous_jobs_key, [])
    next_job_name_list = job.get(ConfigKeys.next_jobs_true_key, [])

    output_loop_job_list = get_body(
        loop_job_list, loop_var_name, parent_path_name_tuple + (loop_name,), prototype_map, job_map
    )
    parameter_map = job.get(ConfigKeys.loop_vars_key, {})
    return TemplateManager.threaded_parameterized_loop_template.render(
        var_name=parameter_manager_name,
        parameter_map=parameter_map,
        parent_job_name=parent_job_var_name,
        loop_name=loop_name,
        loop_var=loop_var_name,
        previous_job_name_list=previous_job_name_list,
        next_job_name_list=next_job_name_list,
        job_list=output_loop_job_list
    )


def get_while_loop_job(job, parent_job_var_name, parent_path_name_tuple, prototype_map, job_map):

    loop_job_list = job.get(ConfigKeys.loop_jobs_key)

    loop_name = job.get(ConfigKeys.job_name_key)
    loop_var_name = while_loop_format_string.format(get_unique_num())

    previous_job_name_list = job.get(ConfigKeys.previous_jobs_key, [])
    next_job_name_list = job.get(ConfigKeys.next_jobs_true_key, [])

    function_name, function_text = get_function_name(job.get(ConfigKeys.script_key))
    module_name_list.append(function_text)

    condition_lambda = TemplateManager.lambda_template.render(
        function_name=function_name,
        arg_list=["workflow_data"]
    )

    parameter_lambda = None
    if ConfigKeys.parameter_update_script_key in job:
        for child_job in loop_job_list:
            child_job[ConfigKeys.parameter_update_script_key] = True

        parameter_function_name, parameter_function_test = get_function_name(
            job.get(ConfigKeys.parameter_update_script_key)
        )
        module_name_list.append(parameter_function_test)

        parameter_lambda = TemplateManager.lambda_template.render(
            function_name=parameter_function_name,
            arg_list=["workflow_data"]
        )

    output_loop_job_list = get_body(
        loop_job_list, loop_var_name, parent_path_name_tuple + (loop_name,), prototype_map, job_map
    )

    return TemplateManager.while_loop_template.render(
        parent_job_name=parent_job_var_name,
        loop_name=loop_name,
        previous_job_name_list=previous_job_name_list,
        next_job_name_list=next_job_name_list,
        condition=condition_lambda,
        loop_var=loop_var_name,
        job_list=output_loop_job_list,
        parameter_function=parameter_lambda
    )


def get_body(job_list_arg, parent_job_var_name, parent_path_name_tuple, prototype_map, job_map):

    local_output_job_list = []

    for job in job_list_arg:
        job_name = job.get(ConfigKeys.job_name_key)
        job_path_name_tuple = parent_path_name_tuple + (job_name,)
        job_map[job_path_name_tuple] = job

    for job in job_list_arg:

        job_type = job.get(ConfigKeys.job_type_key, alc_job_type)

        if job_type == loop_type:
            if job.get(ConfigKeys.loop_vars_key, {}):
                local_output_job_list.append(
                    get_parameterized_loop_job(job, parent_job_var_name, parent_path_name_tuple, prototype_map, job_map)
                )
            else:
                local_output_job_list.append(
                    get_while_loop_job(job, parent_job_var_name, parent_path_name_tuple, prototype_map, job_map)
                )
        elif job_type == branch_type:
            local_output_job_list.append(get_branch(job, parent_job_var_name))
        elif job_type == transform_job_type:
            local_output_job_list.append(get_transform(job, parent_job_var_name, parent_path_name_tuple))
        elif job_type == launch_experiment_type:
            local_output_job_list.append(get_launch_experiment_job(job, parent_job_var_name))
        else:
            local_output_job_list.append(get_launch_activity_job(job, parent_job_var_name, prototype_map))

    return local_output_job_list


def get_data_store_job_list(data_store_job_list, local_root_job_var, job_list):

    local_output_job_list = []

    # ADD DataStore JOBS TO PREVIOUS JOBS OF FIRST NON-DataStore JOBS IN THE WORKFLOW
    # ADD FIRST NON-DataStore JOBS IN THE WORKFLOW TO NEXT JOBS OF DataStore JOBS
    data_store_job_name_list = []
    for data_store_job in data_store_job_list:
        data_store_job_name_list.append(data_store_job.get(ConfigKeys.job_name_key))

    data_store_job_name_set = set(data_store_job_name_list)

    job_name_list = []
    for job in job_list:
        previous_job_name_set = set(job.get(ConfigKeys.previous_jobs_key))
        difference = previous_job_name_set.difference(data_store_job_name_set)
        if len(difference) == 0:
            job_name_list.append(job.get(ConfigKeys.job_name_key))
            job[ConfigKeys.previous_jobs_key] = data_store_job_name_list
    # END ADDING JOB NAMES TO PREVIOUS/NEXT JOBS LISTS

    for data_store_job in data_store_job_list:

        job_name = data_store_job.get(ConfigKeys.job_name_key)

        json_data = {}
        if ConfigKeys.json_data_key in data_store_job:
            json_data[ConfigKeys.json_data_key] = data_store_job.get(ConfigKeys.json_data_key)

        if ConfigKeys.job_data_key in data_store_job:
            json_data[ConfigKeys.job_data_key] = data_store_job.get(ConfigKeys.job_data_key)

        if len(json_data) == 0:
            raise Exception("DataStore:  no valid label for job \"{0}\"".format(job_name))

        local_output_job_list.append(
            TemplateManager.add_data_store_job_template.render(
                parent_job=local_root_job_var,
                job_name=job_name,
                next_job_name_list=job_name_list,
                json_data=json.dumps(json_data, indent=4, sort_keys=True)
            )
        )

    return local_output_job_list


def generate():

    global modules_dir_path

    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("config_file", type=str)

    arguments = argument_parser.parse_args()

    config_file = Path(arguments.config_file)

    with config_file.open() as config_fp:
        config_json = json.load(config_fp)

    status_path = config_json.get(ConfigKeys.status_path_key)

    working_dir_root = config_json.get(ConfigKeys.working_dir_key)
    working_dir_path = Path(working_dir_root)
    working_dir_path.mkdir(parents=True, exist_ok=True)

    script_dir_path = Path(working_dir_path, ScriptFileNames.script_dir_name)
    script_dir_path.mkdir(parents=True, exist_ok=True)

    script_path = Path(script_dir_path, ScriptFileNames.script_file_name)
    script_stdout_path = Path(script_dir_path, ScriptFileNames.script_stdout_file_name)
    script_stderr_path = Path(script_dir_path, ScriptFileNames.script_stderr_file_name)
    failing_task_path_file_path = Path(script_dir_path, ScriptFileNames.failing_task_path_file_name)
    exceptions_file_path = Path(script_dir_path, ScriptFileNames.exceptions_file_name)

    modules_dir_path = Path(script_dir_path, ScriptFileNames.modules_dir_name)
    modules_dir_path.mkdir(parents=True, exist_ok=True)
    Path(modules_dir_path, ScriptFileNames.init_file_name).touch(exist_ok=True)

    generic_active_node = config_json.get(ConfigKeys.generic_active_node_key)

    data_store_job_list = config_json.get(ConfigKeys.data_store_list_key, [])
    job_list = config_json.get(ConfigKeys.job_list_key)

    owner = config_json.get(ConfigKeys.owner_key)

    project_name = config_json.get(ConfigKeys.project_name_key)
    prototype_map = config_json.get(ConfigKeys.prototype_map_key, {})
    output_data_store_job_list = get_data_store_job_list(data_store_job_list, root_job_var, job_list)
    output_job_list = get_body(job_list, root_job_var, tuple(), prototype_map, {})

    function_imports = "" if len(module_name_list) == 0 else TemplateManager.module_imports_template.render(
        scripts_dir_path=script_dir_path,
        modules_dir_name=ScriptFileNames.modules_dir_name,
        module_name_list=module_name_list
    )

    job_list = list(output_data_store_job_list)
    job_list.extend(output_job_list)

    program_text = TemplateManager.root_template.render(
        exceptions_file=exceptions_file_path.absolute(),
        failing_task_path_file=failing_task_path_file_path.absolute(),
        generic_active_node=generic_active_node,
        job_list=job_list,
        function_imports=function_imports,
        output_dir=working_dir_path,
        owner=owner,
        project_name=project_name,
        status_path=status_path,
        stderr_file=script_stderr_path.absolute(),
        stdout_file=script_stdout_path.absolute()
    )

    with script_path.open("w") as script_path_fp:
        print(program_text, file=script_path_fp)


if __name__ == "__main__":
    generate()
