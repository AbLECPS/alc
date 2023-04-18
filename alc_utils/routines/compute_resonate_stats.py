import sys
import os
import imp
resonatepath = os.path.join(os.environ['ALC_HOME'], 'resonate', 'resonate')
sys.path.append(resonatepath)
import general_analyze
alc_working_dir_env_var_name = "ALC_WORKING_DIR"
alc_working_dir_name = os.environ.get(alc_working_dir_env_var_name, None)


default_dataprocessorpath = os.path.join(
    os.environ['ALC_HOME'], 'resonate', 'resonate', 'UUV_datafile_default.py')


def fix_folder_path(folder_path):
    if (not folder_path):
        return None
    pos = folder_path.find('jupyter')
    if (pos == -1):
        return folder_path
    folder_path = folder_path[pos:]
    if (alc_working_dir_name):
        ret = os.path.join(alc_working_dir_name, folder_path)
        return ret
    return None


def parse_folder_list(folder_list):
    folders = []
    for f in folder_list:
        if isinstance(f, str):
            fixed_path = fix_folder_path(f)
            if (fixed_path):
                folders.append(fixed_path)
            continue
        if isinstance(f, dict):
            if (f.has_key('directory') and f.get('directory', None)):
                fixed_path = fix_folder_path(f['directory'])
                if (fixed_path):
                    folders.append(fixed_path)
            continue

    return folders


#DATA_DIR = "/hdd2/alc-repo-r0/alc_workspace/resonate-data-pruned-bluerov-faults"
def run(folder_list=[], data_processor_path=default_dataprocessorpath):
    folders = parse_folder_list(folder_list)
    dataprocessor_module = imp.load_source(
        'EventDataFile', data_processor_path)
    print ("\n****************************************\n")
    print (" Computing Statistics for Resonate .....\n")
    general_analyze.run_general_analyze_script(
        False, df_class=dataprocessor_module.EventDataFile, top_folders=folders)
    print ("\n****************************************")
