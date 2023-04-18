import math
import json
import os
import stat
from pathlib import Path
import TM as TemplateManager
import alc_utils.slurm_executor as slurm_executor
from alc_utils.slurm_executor import WebGMEKeys
from alc_utils.update_job_status_daemon import Keys as UpdateKeys
import alc_utils.alc_model_updater as model_updater
import time

# Useful macros
SECONDS_PER_MINUTE = 60.0
SLURM_GRACE_TIME_MIN = 5

jupyter_dir_name = 'jupyter'
main_py_file_name = 'main.py'
run_sh_file_name = "run.sh"

bash_command = "bash"
bash_script_name = "run.sh"

activity_setup_job_type = 'ACTIVITY_SETUP'

update_result_metadata_task_name = "Update_Result_Metadata"

alc_work_env_var_name = "ALC_WORK"

slurm_job_params_filename = "slurm_params.json"


def find_meta_child(plugin_object, node, pattern):
    meta_node = plugin_object.core.get_meta_type(node)

    found_meta_child = None
    for meta_child_path in plugin_object.core.get_children_meta(meta_node):
        meta_child = plugin_object.core.load_by_path(plugin_object.root_node, meta_child_path)
        meta_child_name = plugin_object.core.get_fully_qualified_name(meta_child)
        if meta_child_name.endswith(pattern):
            found_meta_child = meta_child
            break

    return found_meta_child


def find_meta_children(plugin_object, node, meta_node):

    meta_node_name = plugin_object.core.get_fully_qualified_name(meta_node)

    child_list = []
    for child_path in plugin_object.core.get_children_paths(node):
        child = plugin_object.core.load_by_path(plugin_object.root_node, child_path)
        child_meta = plugin_object.core.get_meta_type(child)
        child_meta_name = plugin_object.core.get_fully_qualified_name(child_meta)
        if child_meta_name == meta_node_name:
            child_list.append(child)

    return child_list


def should_append_number(name, data_node_name_set):

    append_number = False
    for data_node_name in data_node_name_set:
        if data_node_name.startswith(name):
            remainder = data_node_name[len(name):]
            if len(remainder) == 0 or len(remainder) >= 2 and remainder[0] == '-' and remainder[1].isnumeric():
                append_number = True
                break

    return append_number


def append_number_to_name(name, data_node_name_set):

    new_name = None

    check_number = True
    new_number = -1
    while check_number:
        check_number = False
        new_number += 1
        new_name = "{0}-{1}".format(name, new_number)
        for data_node_name in data_node_name_set:
            if data_node_name.startswith(new_name):
                remainder = data_node_name[len(name):]
                if remainder == "" or remainder.startswith('-'):
                    check_number = True
                    break

    return new_name

def create_data_node_from_router(plugin_object, active_node, config, result_folder,logger,createdAt):
    execution_name = config.get("name", 'data')
    project_info   = plugin_object.project.get_project_info()
    project_owner  = project_info[WebGMEKeys.project_owner_key]
    project_name   = project_info[WebGMEKeys.project_name_key]
    active_node_path = plugin_object.core.get_path(active_node)
    modifications = {
        "createdAt": int(createdAt)* 1000,
        "jobstatus": "Submitted",
        "resultDir": result_folder
    }

    return model_updater.create_data_node(logger, project_owner, project_name, active_node_path, execution_name, modifications)




def create_data_node(plugin_object, active_node, config, result_folder,logger,createdat):

    return create_data_node_from_router(plugin_object, active_node, config, result_folder,logger,createdat)

    name = config.get("name", 'data')

    result_meta_node = find_meta_child(plugin_object, active_node, ".Result")
    if result_meta_node is None:
        raise Exception("Could not create data node -- no result meta node!")

    result_node_list = find_meta_children(plugin_object, active_node, result_meta_node)
    if len(result_node_list) == 0:
        raise Exception("Could not create data node -- no result node parent!")

    result_node = result_node_list[0]

    data_meta_node = find_meta_child(plugin_object, result_node, ".pipeline.Data")

    data_node_list = find_meta_children(plugin_object, result_node, data_meta_node)
    data_node_name_set = set([plugin_object.core.get_fully_qualified_name(x) for x in data_node_list])

    new_name = append_number_to_name(name, data_node_name_set) \
        if should_append_number(name, data_node_name_set) else name

    new_data_node = plugin_object.core.create_child(result_node, data_meta_node)
    plugin_object.core.set_attribute(new_data_node, "name", new_name)
    plugin_object.core.set_attribute(
        new_data_node, "activity", plugin_object.core.get_fully_qualified_name(active_node)
    )
    plugin_object.core.set_attribute(new_data_node, "createdAt", int(time.time() * 1000))
    plugin_object.core.set_attribute(new_data_node, "resultDir", str(result_folder))

    plugin_object.util.save(plugin_object.root_node, plugin_object.commit_hash, 'master', "new data node for activity")

    return plugin_object.core.get_path(new_data_node)


def setup_job(
        plugin_object,
        name,
        seconds_since_epoch,
        timeout_param,
        job_type,
        setup_folder,
        deploy_job=True,
        result_folder=None,
        setup_workflow_campaign = False,
        camp_definition = None,
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
    if (plugin_object.repo_home):
        os.environ[WebGMEKeys.repo_home_key] = plugin_object.repo_home
    else:
        os.environ[WebGMEKeys.repo_home_key] = ''
    #slurm_job_params[WebGMEKeys.repo_home_key] = plugin_object.repo_home

    result_dir_root = result_folder
    if (setup_workflow_campaign and camp_definition):
        result_folder = Path(str(result_dir_root), 'Prototype', name)
        if not result_folder.exists():
            result_folder.mkdir(parents=True)


    folder_path = generate_files(project_info, name, seconds_since_epoch, job_type, setup_folder, slurm_job_params, result_folder)

    if (setup_workflow_campaign and camp_definition):
        generate_workflow_json(plugin_object, name,project_info, camp_definition,result_dir_root)


    print('folder_path ', folder_path)
    if deploy_job:
        print('job is deployed')
        data_node_path = create_data_node(
            plugin_object, plugin_object.active_node, plugin_object.config, result_folder,logger,seconds_since_epoch
        )
        slurm_job_params[UpdateKeys.data_node_path_key] = data_node_path
        slurm_executor.slurm_deploy_job(str(folder_path), job_params=slurm_job_params)
    else:
        print('job is not deployed')

    return slurm_job_params


def generate_files(
        project_info, 
        activenode_name, 
        seconds_since_epoch, 
        job_type, 
        setup_folder, 
        slurm_job_params,  
        result_folder):

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
    
    if not result_folder:
        result_dir = Path(
            alc_wkdir,
            jupyter_dir_name,
            project_info[WebGMEKeys.project_owner_key] + '_' + project_info[WebGMEKeys.project_name_key],
            activenode_name,
            str(seconds_since_epoch)
        )
    else:
        result_dir = result_folder
    
    


    generate_main_file(result_dir, job_type,setup_folder)
    generate_scripts(result_dir)
    generate_slurm_job_params(slurm_job_params, result_dir)

    return result_dir


def generate_scripts(result_dir):
    # Fill out top-level launch template
    run_template = TemplateManager.run_script_template
    top_script = run_template.render()

    # Write "from_plugin" to file
    top_script_path = Path(result_dir, run_sh_file_name)
    with top_script_path.open('w') as f:
        f.write(top_script)
    st = top_script_path.stat()
    top_script_path.chmod(st.st_mode | stat.S_IEXEC)


def generate_slurm_job_params(slurm_job_params, result_dir):
    # write slurm params so that they can be used from workflow
    with Path(result_dir, slurm_job_params_filename).open("w") as json_fp:
        json.dump(slurm_job_params, json_fp, indent=4, sort_keys=True)


def generate_main_file(result_dir, job_type,setup_folder):
    # Fill out top-level launch template
    main_template = TemplateManager.python_main_template
    code = main_template.render(job_type=str(job_type), setup_folder=str(setup_folder))
    # Write python main code to path
    main_path = Path(result_dir, main_py_file_name)
    with main_path.open("w") as f:
        f.write(code)


def generate_workflow_json(
    plugin_object, 
    name,
    project_info, 
    camp_definition,
    result_dir_root):

    workflow_template = TemplateManager.workflow_json_template
    owner_name = project_info[WebGMEKeys.project_owner_key]
    project_name = project_info[WebGMEKeys.project_name_key]
    core = plugin_object.core
    active_node = plugin_object.active_node
    activity_node_path = core.get_path(active_node)
    alc_node_path = '/'+activity_node_path.split('/')[1]
    print(alc_node_path)
    camp_keys = list(camp_definition.keys())
    camp_keys.sort()
    camp_vars = ','.join(camp_keys)
    repo_home_val = ''

    if (plugin_object.repo_home):
        repo_home_val = plugin_object.repo_home
    
        

    camp_var_map = {}
    for k in camp_keys:
        camp_var_map[k]=str(camp_definition[k])

    workflow_output = workflow_template.render(
                        activity_exec_name = name,
                        activity_name = name,
                        activity_node_path = activity_node_path,
                        alc_node_path = alc_node_path,
                        campaign_variable_names = camp_vars,
                        campaign_variable_list = json.dumps(camp_keys),
                        campaign_variable_map = json.dumps(camp_var_map),
                        owner_name = owner_name,
                        project_name = project_name,
                        working_dir = result_dir_root,
                        repo_home_dir = repo_home_val,
                        )
    json_file_name = "WF-Campaign-{0}.json".format(name)
    json_path = Path(result_dir_root, json_file_name)
    with json_path.open("w") as f:
        f.write(workflow_output)

    wf_launch_path = Path("/alc","workflows","config", json_file_name)
    with wf_launch_path.open("w") as f:
        f.write(workflow_output)
