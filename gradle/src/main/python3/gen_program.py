import json
import re
import copy
import argparse
from pathlib import Path
import ConfigKeys
import ScriptFileNames
import TemplateManager


root_job_var = "root_job"

alc_job_type = "ALC_Job"
branch_type = "Branch"
loop_type = "Loop"
transform_job_type = "Transform"

parameter_manager_format_string = "parameter_manager_{0}"
threaded_loop_format_string = "threaded_loop_{0}"
while_loop_format_string = "while_loop_{0}"
workflow_job_format_string = "workflow_job_{0}"

user_function_list = []
function_map = {}

configuration_argument_names = [
    (ConfigKeys.operation_config_key, ConfigKeys.operation_key),
    (ConfigKeys.source_list_config_key, ConfigKeys.source_list_key),
    (ConfigKeys.destination_list_config_key, ConfigKeys.destination_list_key),
    (ConfigKeys.destination_lec_config_key, ConfigKeys.destination_lec_key)
]

unique_id = 0


def get_unique_num():
    global unique_id
    retval = unique_id
    unique_id += 1
    return retval


def get_function_name(function_text):
    local_function_text = copy.copy(function_text)
    if local_function_text[0] == "'":
        local_function_text = local_function_text[1:]
    if local_function_text[-1] == "'":
        local_function_text = local_function_text[:-1]

    local_function_text = local_function_text.strip()

    match = re.search(r"def\s*([^\s(]+)\s*\(", local_function_text)
    function_name = match.group(1)
    new_function_name = "{0}_{1}".format(function_name, get_unique_num())

    new_function_text = re.sub(function_name, new_function_name, local_function_text, count=1)
    return new_function_name, new_function_text


def resolve_adjacent_job_names(start_adjacent_job_name_list, parent_job_name_tuple, job_map, job_name_list_key):
    adjacent_job_name_set = set(start_adjacent_job_name_list)
    actual_next_job_name_set = set()
    while adjacent_job_name_set:
        adjacent_job_name = list(adjacent_job_name_set)[0]
        adjacent_job_name_set.remove(adjacent_job_name)

        adjacent_job_name_tuple = parent_job_name_tuple + (adjacent_job_name,)
        adjacent_job = job_map.get(adjacent_job_name_tuple)
        if adjacent_job.get(ConfigKeys.job_type_key, None) == transform_job_type:
            adjacent_job_name_set = adjacent_job_name_set.union(set(adjacent_job.get(job_name_list_key, [])))
        else:
            actual_next_job_name_set.add(adjacent_job_name)

    return list(actual_next_job_name_set)


def get_adjacent_job_name_list(job, parent_job_name_tuple, job_map, job_name_list_key):
    return resolve_adjacent_job_names(job.get(job_name_list_key, []), parent_job_name_tuple, job_map, job_name_list_key)


def resolve_previous_job_name_list(start_previous_job_name_list, parent_job_name_tuple, job_map):
    return resolve_adjacent_job_names(
        start_previous_job_name_list, parent_job_name_tuple, job_map, ConfigKeys.previous_job_name_list_key
    )


def get_previous_job_name_list(job, parent_job_name_tuple, job_map):
    return get_adjacent_job_name_list(job, parent_job_name_tuple, job_map, ConfigKeys.previous_job_name_list_key)


def resolve_next_job_name_list(start_next_job_name_list, parent_job_name_tuple, job_map):
    return resolve_adjacent_job_names(
        start_next_job_name_list, parent_job_name_tuple, job_map, ConfigKeys.next_job_name_list_key
    )


def get_next_job_name_list(job, parent_job_name_tuple, job_map):
    return get_adjacent_job_name_list(job, parent_job_name_tuple, job_map, ConfigKeys.next_job_name_list_key)


def get_path_list_tuple(path_list):
    local_tuple = tuple()
    for item in path_list:
        if isinstance(item, list):
            local_tuple += (item[0],)
        else:
            local_tuple += (item,)

    return local_tuple


def get_origin_path_and_function_list(input_job_path_list, job_map):

    prior_input_job_path_list = input_job_path_list
    function_list = []
    input_job_path_list_tuple = get_path_list_tuple(prior_input_job_path_list)
    while input_job_path_list_tuple in function_map:
        function_list.insert(0, function_map.get(input_job_path_list_tuple))
        input_job = job_map.get(input_job_path_list_tuple)
        prior_input_job_path_list = input_job.get(ConfigKeys.input_list_key)[0].get(ConfigKeys.input_job_key)
        input_job_path_list_tuple = get_path_list_tuple(prior_input_job_path_list)

    return prior_input_job_path_list, function_list


def get_list_of_functions_string(function_name_list):
    return "[" + ", ".join(function_name_list) + "]"


def get_use_parameters_list(job):
    use_parameters_list = ["all"]
    if ConfigKeys.use_parameters_key in job:
        use_parameters_list = job.get(ConfigKeys.use_parameters_key)
        if not isinstance(use_parameters_list, list):
            use_parameters_list = []

    return use_parameters_list


def get_job(job, parent_job_var_name, parent_job_name_tuple, job_map):
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
        call_list.append(TemplateManager.configuration_template.render(named_argument_list=argument_list))

    for local_input in job.get(ConfigKeys.input_list_key, {}):
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

            origin_path_list, function_list = get_origin_path_and_function_list(input_job_path, job_map)
            function_list.append("ResultAux.get_list_paths")

            source_list_lambda = TemplateManager.lambda_template.render(
                function_name="workflow_data.get_value",
                arg_list=[origin_path_list, get_list_of_functions_string(function_list)]
            )
            argument_list.append(TemplateManager.named_argument_template.render(
                argument_name="source_list", argument_value=source_list_lambda
            ))
        call_list.append(TemplateManager.configuration_template.render(named_argument_list=argument_list))

    for activity_name, activity_path in job.get(ConfigKeys.activities_map_key).items():
        call_list.append(
            TemplateManager.add_activity_template.render(activity_name=activity_name, activity_path=activity_path)
        )

    if ConfigKeys.parameter_manager_var_key in job:
        use_parameters_list = get_use_parameters_list(job)

        parameter_updates_var = job.get(ConfigKeys.parameter_manager_var_key)
        function_name = "{0}.get_combination".format(parameter_updates_var)
        call_list.append(TemplateManager.add_parameter_updates_template.render(
            update_spec=TemplateManager.lambda_template.render(
                function_name=function_name,
                arg_list=["workflow_data", use_parameters_list]
            )
        ))

    if ConfigKeys.parameter_update_script_key in job:
        use_parameters_list = get_use_parameters_list(job)

        function_name = "workflow_data.get_parent_parameter_updates"
        call_list.append(TemplateManager.add_parameter_updates_template.render(
            update_spec=TemplateManager.lambda_template.render(
                function_name=function_name,
                arg_list=[use_parameters_list]
            )
        ))

    previous_job_name_list = get_previous_job_name_list(job, parent_job_name_tuple, job_map)
    next_job_name_list = get_next_job_name_list(job, parent_job_name_tuple, job_map)
    return TemplateManager.add_job_template.render(
        var_name=workflow_job_format_string.format(get_unique_num()),
        call_list=call_list,
        job_name=job.get("job_name"),
        previous_job_name_list=previous_job_name_list,
        next_job_name_list=next_job_name_list,
        parent_job=parent_job_var_name
    )


def get_branch(job, parent_job_var_name, parent_path_name_tuple, job_map):

    function_name, function_text = get_function_name(job.get(ConfigKeys.script_key))
    user_function_list.append(function_text)

    job_name = job.get(ConfigKeys.job_name_key)

    previous_job_name_list = get_previous_job_name_list(job, parent_path_name_tuple, job_map)

    true_job_name_list = job.get(ConfigKeys.true_branch_job_name_list_key)
    branch_job_name_list = resolve_next_job_name_list(true_job_name_list, parent_path_name_tuple, job_map)

    false_job_name_list = job.get(ConfigKeys.false_branch_job_list_key)
    next_job_name_list = resolve_next_job_name_list(false_job_name_list, parent_path_name_tuple, job_map)

    input_job_path = job.get(ConfigKeys.input_list_key)[0].get(ConfigKeys.input_job_key)
    origin_path_list, function_list = get_origin_path_and_function_list(input_job_path, job_map)
    function_list.append(function_name)

    condition_lambda = TemplateManager.lambda_template.render(
        function_name="workflow_data.get_value",
        arg_list=[origin_path_list, get_list_of_functions_string(function_list)]
    )

    return TemplateManager.add_branch_point_template.render(
        parent_job=parent_job_var_name,
        job_name=job_name,
        previous_job_name_list=previous_job_name_list,
        next_job_name_list=next_job_name_list,
        condition=condition_lambda,
        branch_job_name_list=branch_job_name_list
    )


def get_parameterized_loop_job(job, parent_job_var_name, parent_path_name_tuple, job_map):
    loop_job_list = job.get(ConfigKeys.loop_jobs_key)

    parameter_manager_name = parameter_manager_format_string.format(get_unique_num())
    for child_job in loop_job_list:
        child_job[ConfigKeys.parameter_manager_var_key] = parameter_manager_name

    loop_name = job.get(ConfigKeys.job_name_key)
    loop_var_name = threaded_loop_format_string.format(get_unique_num())

    previous_job_name_list = get_previous_job_name_list(job, parent_path_name_tuple, job_map)
    next_job_name_list = get_next_job_name_list(job, parent_path_name_tuple, job_map)

    output_loop_job_list = get_body(loop_job_list, loop_var_name, parent_path_name_tuple + (loop_name,), job_map)
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


def get_while_loop_job(job, parent_job_var_name, parent_path_name_tuple, job_map):

    loop_job_list = job.get(ConfigKeys.loop_jobs_key)

    loop_name = job.get(ConfigKeys.job_name_key)
    loop_var_name = while_loop_format_string.format(get_unique_num())

    previous_job_name_list = get_previous_job_name_list(job, parent_path_name_tuple, job_map)
    next_job_name_list = get_next_job_name_list(job, parent_path_name_tuple, job_map)

    function_name, function_text = get_function_name(job.get(ConfigKeys.script_key))
    user_function_list.append(function_text)

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
        user_function_list.append(parameter_function_test)

        parameter_lambda = TemplateManager.lambda_template.render(
            function_name=parameter_function_name,
            arg_list=["workflow_data"]
        )

    output_loop_job_list = get_body(loop_job_list, loop_var_name, parent_path_name_tuple + (loop_name,), job_map)

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


def get_body(job_list_arg, parent_job_var_name, parent_path_name_tuple, job_map):

    local_output_job_list = []

    for job in job_list_arg:
        job_name = job.get(ConfigKeys.job_name_key)
        job_path_name_tuple = parent_path_name_tuple + (job_name,)
        job_map[job_path_name_tuple] = job

    for job in job_list_arg:

        job_name = job.get(ConfigKeys.job_name_key)
        job_type = job.get(ConfigKeys.job_type_key, alc_job_type)

        if job_type == loop_type:
            if job.get(ConfigKeys.loop_vars_key, {}):
                local_output_job_list.append(
                    get_parameterized_loop_job(job, parent_job_var_name, parent_path_name_tuple, job_map)
                )
            else:
                local_output_job_list.append(
                    get_while_loop_job(job, parent_job_var_name, parent_path_name_tuple, job_map)
                )
        elif job_type == branch_type:
            local_output_job_list.append(get_branch(job, parent_job_var_name, parent_path_name_tuple, job_map))
        elif job_type == transform_job_type:
            function_name, function_text = get_function_name(job.get(ConfigKeys.script_key))
            user_function_list.append(function_text)
            function_map[parent_path_name_tuple + (job_name,)] = function_name
        else:
            local_output_job_list.append(get_job(job, parent_job_var_name, parent_path_name_tuple, job_map))

    return local_output_job_list


def generate():
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

    generic_active_node = config_json.get(ConfigKeys.generic_active_node_key)

    job_list = config_json.get(ConfigKeys.job_list_key)

    owner = config_json.get(ConfigKeys.owner_key)

    project_name = config_json.get(ConfigKeys.project_name_key)

    output_job_list = get_body(job_list, root_job_var, tuple(), {})

    script_dir_path = Path(working_dir_path, ScriptFileNames.script_dir_name)
    script_dir_path.mkdir(parents=True, exist_ok=True)

    script_path = Path(script_dir_path, ScriptFileNames.script_file_name)
    script_stdout_path = Path(script_dir_path, ScriptFileNames.script_stdout_file_name)
    script_stderr_path = Path(script_dir_path, ScriptFileNames.script_stderr_file_name)
    failing_task_path_file_path = Path(script_dir_path, ScriptFileNames.failing_task_path_file_name)
    exceptions_file_path = Path(script_dir_path, ScriptFileNames.exceptions_file_name)

    program_text = TemplateManager.root_template.render(
        generic_active_node=generic_active_node,
        job_list=output_job_list,
        output_dir=working_dir_path,
        owner=owner,
        project_name=project_name,
        status_path=status_path,
        stderr_file=script_stderr_path.absolute(),
        stdout_file=script_stdout_path.absolute(),
        failing_task_path_file=failing_task_path_file_path.absolute(),
        exceptions_file=exceptions_file_path.absolute(),
        user_function_list=user_function_list
    )

    with script_path.open("w") as script_path_fp:
        print(program_text, file=script_path_fp)


if __name__ == "__main__":
    generate()
