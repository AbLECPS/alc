"""
This file can be used as the entry point when debugging the python portion of the plugin.
Rather than relying on be called from a node-process with a corezmq server already up and running
(which is the case for run_plugin.py) this script starts such a server in a sub-process.

To change the context (project-name etc.) modify the CAPITALIZED options passed to the spawned node-js server.

Note! This must run with the root of the webgme-repository as cwd.
"""

import sys
import os
import subprocess
import signal
import logging
import json
import shutil
import time
import alc_utils.config as alc_config
import alc_utils.common as alc_common
from webgme_bindings import WebGME, PluginBase
import webgme_bindings.exceptions
from builtins import input


# Define a custom logging handler so we can easily keep track of all warnings/errors reported
class WarningRecorder(logging.StreamHandler):
    def __init__(self):
        super(WarningRecorder, self).__init__()
        self.warning_msgs = []
        self.error_msgs = []

    def emit(self, record):
        msg = self.format(record)
        if record.levelno == logging.WARNING:
            self.warning_msgs.append(msg)
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            self.error_msgs.append(msg)

    def get_warning_msgs(self):
        return self.warning_msgs

    def get_error_msgs(self):
        return self.error_msgs


# Setup the logger
logger = logging.getLogger('DataCleanup')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)  # By default it logs to stderr..
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
# Add additional handler
record_handler = WarningRecorder()
record_handler.setLevel(logging.WARNING)
record_handler.setFormatter(formatter)
logger.addHandler(record_handler)


class DataCleanup(PluginBase):
    def __init__(self, *args, **kwargs):
        # Call base class init
        super(DataCleanup, self).__init__(*args, **kwargs)

        # Support both "" and "ALCMeta" namespaces.
        if self.namespace == "":
            self.namespace_prefix = "ALCMeta."
        elif self.namespace == "ALCMeta":
            self.namespace_prefix = ""
        else:
            raise ValueError("Unsupported namespace (%s)." % self.namespace)

    # Wrapper function to automatically use correct META prefix
    # eg. "AssemblyModel" may be "ALCMeta.AssemblyModel" or just "AssemblyModel"
    def get_meta(self, meta_type):
        return self.META[self.namespace_prefix + meta_type]

    def get_project_data(self):
        core = self.core
        root_node = self.root_node
        active_node = self.active_node
        project_info = self.project.get_project_info()
        logger.debug('ProjectInfo: {0}'.format(json.dumps(project_info, indent=4, sort_keys=True)))

        # Find primary ALC object (direct child of ROOT)
        alc_node = None
        for child in self.core.load_children(root_node):
            if self.core.get_meta_type(child) == self.get_meta("ALC"):
                alc_node = child
                break
        if alc_node is None:
            logger.error("Cannot find ALC node as a direct child of the project root.")
            raise IOError("Cannot find ALC node as a direct child of the project root.")

        # Search entire sub-tree starting from ALC object for any metadata objects pointing to existing datasets.
        metadata_nodes = []
        for node_info in self.core.load_own_sub_tree(alc_node):
            child = self.core.load_by_path(root_node, node_info["nodePath"])
            if self.core.get_meta_type(child) == self.get_meta("DEEPFORGE.pipeline.Data"):
                metadata_nodes.append(child)

        # Summarize discovered metadata objects
        if len(metadata_nodes) == 0:
            logger.warning("Did not find any metadata in this project. "
                           "Please confirm you are using the correct project.")
        else:
            logger.info("Found %d metadata objects within project %s/%s." % (len(metadata_nodes),
                                                                             project_info["owner"],
                                                                             project_info["name"]))

        # Parse relevant info from metadata objects
        metadatas = []
        for metadata_node in metadata_nodes:
            metadata_path = self.core.get_path(metadata_node)

            # Try to find where the actual data corresponding to this metadata object is stored.
            # There have been several changes to how this information is stored. Try various ways from newest to oldest.
            metadata_dict = None
            result_dir = self.core.get_attribute(metadata_node, "resultDir")
            data_info = self.core.get_attribute(metadata_node, "datainfo")
            if result_dir is not None and len(result_dir) > 0:
                metadata_dict = {"directory": result_dir}
            if metadata_dict is None and data_info is not None and len(data_info) > 0:
                logger.info("DATA_INFO: %s" % data_info)
                notebook_path = json.loads(data_info).get("url", None)
                if notebook_path is not None:
                    metadata_dict = {"directory": os.path.dirname(notebook_path)}
            if metadata_dict is None:
                # Load metadata from BLOB storage as string
                data_blob_hash = self.core.get_attribute(metadata_node, "data")
                if data_blob_hash is None:
                    logger.warning("Metadata object (name: %s, path: %s) does not contain any data reference." %
                                   (self.core.get_attribute(metadata_node, "name"), metadata_path))
                    # unknown_metadata_list.append(metadata_node)
                    continue
                try:
                    metadata_str = self.get_file(data_blob_hash)
                except webgme_bindings.exceptions.JSError as e:
                    logger.warning("Metadata object (name: %s, path: %s) contains a data reference that is not a valid "
                                   "BLOB object." %
                                   (self.core.get_attribute(metadata_node, "name"), metadata_path))
                    # unknown_metadata_list.append(metadata_node)
                    continue

                # Parse metadata string to dictionary
                if len(metadata_str) == 0:
                    logger.warning("Metadata object (name: %s, path: %s) contained empty metadata. Skipping." %
                                   (self.core.get_attribute(metadata_node, "name"), metadata_path))
                    continue
                else:
                    try:
                        # FIXME: Metadata string is sometimes a list containing a single dictionary. Why?
                        parsed_json = json.loads(metadata_str)
                        if type(parsed_json) is list:
                            metadata_dict = json.loads(metadata_str)[0]
                        else:
                            metadata_dict = json.loads(metadata_str)
                    except ValueError as e:
                        logger.error("Failed to parse metadata string: %s" % metadata_str)
                        raise e

            # Confirm the metadata was parsed correctly and append to list of metadatas
            if metadata_dict is None:
                logger.warning("Could not parse any valid metadata from metadata object (name: %s, path: %s). "
                               "Skipping." % (self.core.get_attribute(metadata_node, "name"), metadata_path))
            else:
                metadatas.append({"path": metadata_path, "valid": None, "metadata": metadata_dict})

        return metadatas


def get_data_dir_list(root_path, project_basename):
    # Confirm project directory exists
    project_dir = os.path.join(root_path, project_basename)
    if not os.path.isdir(project_dir):
        logger.warning("Project data directory not found at path %s." % project_dir)
        return []
    else:
        logger.info("Found Project data directory at path %s. Searching for dataset directories..." % project_dir)

    # Build list of the directories containing project artifacts. Store paths relative to top of ALC Workspace
    # This assumes directory is structured in the typical 3-level hierarchy format:
    #   <PROJECT_OWNER>_<PROJECT_NAME>/<MODEL_NAME>/<DATASET_ID>/
    dataset_paths_list = []
    for model_dir_rel in os.listdir(project_dir):
        model_dir = os.path.join(project_dir, model_dir_rel)
        if os.path.isdir(model_dir):
            for dataset_dir_rel in os.listdir(model_dir):
                dataset_dir = os.path.join(model_dir, dataset_dir_rel)
                if os.path.isdir(dataset_dir):
                    dataset_path = os.path.join(project_basename,
                                                model_dir_rel,
                                                dataset_dir_rel)
                    dataset_paths_list.append(dataset_path)

    # TODO: Python docs indicate scan_dir is likely more efficient for this application, but it requires Python 3.
    #       Leaving here for future reference
    # for model_dir_entry in os.scandir(project_dir):
    #     if model_dir_entry.is_dir():
    #         for dataset_dir_entry in os.scandir(model_dir_entry.path):
    #             if dataset_dir_entry.is_dir():
    #                 dataset_rel_path = os.path.join(project_dir_basename,
    #                                                 model_dir_entry.name,
    #                                                 dataset_dir_entry.name)
    #                 dataset_rel_paths.append(dataset_rel_path)

    # Summarize data directory results
    if len(dataset_paths_list) == 0:
        logger.warning("Did not find any stored datasets owned by this project. "
                       "Please confirm you are using the correct project.")
    else:
        logger.info("Found %d dataset directories under path %s." % (len(dataset_paths_list), root_path))

    return dataset_paths_list


def cross_validate_metadata(project_to_metadatas_dict, project_to_dir_list_dict, root_path=alc_config.JUPYTER_WORK_DIR):
    """This function compares all identified metadata objects to the available dataset directories to identify """
    # Flatten and invert dictionary of data directories for more efficient searching later
    # ie. "if key in <dict>" should scale better than "if key in <list>"
    # Also initialize a dictionary with empty arrays for holding each project's results
    data_dir_to_info_dict = {}
    project_info_dict = {}
    for project_name, data_dir_list in project_to_dir_list_dict.items():
        project_info_dict[project_name] = {"dir_infos": [], "metadatas": []}
        for data_dir in data_dir_list:
            data_dir_to_info_dict[data_dir] = {"project": project_name, "metadata_info": None}

    # For every metadata object, find corresponding dataset directory.
    # Update dataset dictionary so that dataset_path -> metadata_node (or None if no matching metadata found)
    # Build two dicts: One containing valid metadata objects (eg. have matching data) and one with invalid objects.
    # Both dictionaries map metadata_node -> data_directory (relative path)
    for project_name, metadatas in project_to_metadatas_dict.items():
        for metadata_info in metadatas:
            # Get result directory from metadata and convert to relative path if needed
            metadata_dict = metadata_info["metadata"]
            result_dir = metadata_dict["directory"]
            if os.path.isabs(result_dir):
                if result_dir.startswith(root_path):
                    result_dir = os.path.relpath(result_dir, root_path)
                else:
                    logger.warning("Metadata object points to a directory (path: %s) "
                                   "which does not exist in current ALC_WORKSPACE. Skipping." % result_dir)
                    metadata_info["valid"] = False
                    project_info_dict[project_name]["metadatas"].append(metadata_info)
                    continue

            # If this path exists in our dictionary of data directories OR is a valid subdirectory of a data dir,
            # update entry appropriately and mark as valid. Otherwise, mark as invalid.
            # TODO: It is possible that 2+ metadata objects point to the same data directory.
            #   With this algorithm, only store one (whichever is processed last). Does this matter?
            candidate_dir = result_dir
            while len(candidate_dir) > 0:
                if candidate_dir in data_dir_to_info_dict:
                    # This metadata points to a valid data directory. Mark it as valid and add to correct project.
                    metadata_info["valid"] = True
                    project_info_dict[project_name]["metadatas"].append(metadata_info)

                    # This data directory is NOT an orphan. Store the corresponding metadata
                    data_dir_to_info_dict[candidate_dir]["metadata_info"] = metadata_info
                    break
                else:
                    # Remove one level of the directory hierarchy and check again
                    # This is a simple algorithm to check if the original result_dir is a sub-directory
                    candidate_dir = os.path.split(candidate_dir)[0]

    # Invert data_dir_to_info_dict and add to results
    for path, dir_info in data_dir_to_info_dict.items():
        project_info_dict[dir_info["project"]]["dir_infos"].append({"path": path,
                                                                    "metadata_info": dir_info["metadata_info"]})

    return project_info_dict


def print_info_summary(project_info):
    # Count how many data directories are valid
    dir_infos = project_info["dir_infos"]
    total_data_count = len(dir_infos)
    orphaned_data_count = 0
    for dir_info in dir_infos:
        if dir_info["metadata_info"] is None:
            orphaned_data_count += 1
    valid_data_count = total_data_count - orphaned_data_count

    # Print info about dataset directories
    logger.info("Found %d total data directories:\n\t%d valid\n\t%d orphaned" %
                (total_data_count, valid_data_count, orphaned_data_count))

    # Count how many metadatas are valid
    metadatas = project_info["metadatas"]
    total_metadata_count = len(metadatas)
    valid_metadata_count = 0
    for metadata_info in metadatas:
        if metadata_info["valid"]:
            valid_metadata_count += 1
    invalid_metadata_count = total_metadata_count - valid_metadata_count

    # Print info about metadata objects
    logger.info("Found %d total metadata objects:\n\t%d valid\n\t%d invalid" %
                (total_metadata_count, valid_metadata_count, invalid_metadata_count))


def print_info_details(project_info):
    # List every invalid metadata (ie. points to non-existent data directory)
    logger.info("Invalid metadata node paths:")
    for metadata in project_info["metadatas"]:
        if not metadata["valid"]:
            logger.info("\t%s" % metadata["path"])

    # List every orphaned directory (ie. no corresponding metadata)
    logger.info("Orphaned directories:")
    for dir_info in project_info["dir_infos"]:
        if dir_info["metadata_info"] is None:
            logger.info("\t%s" % dir_info["path"])


def main():
    # Modify these or add option or parse from sys.argv (as in done in run_plugin.py)
    PORT = '6061'
    USER = 'admin:vanderbilt'
    START_ZMQ_SERVER = True
    ZMQ_SERVER_STARTUP_DELAY_S = 3
    BRANCH_NAME = 'master'
    ACTIVE_NODE_PATH = ''
    ACTIVE_SELECTION_PATHS = []
    NAMESPACE = ''
    ALC_WORKSPACE_PATH = alc_config.JUPYTER_WORK_DIR
    ARCHIVE_DIRECTORY = os.path.join(ALC_WORKSPACE_PATH, "archive")

    # List of projects to be searched. Specified as (<owner>, <project_name>) 2-tuples
    PROJECT_INFOS = [("admin", "BlueROV")]
    

    # Find Core ZMQ server file
    COREZMQ_SERVER_FILE = os.path.join(os.getcwd(), 'node_modules', 'webgme-bindings', 'bin', 'corezmq_server.js')
    if not os.path.isfile(COREZMQ_SERVER_FILE):
        COREZMQ_SERVER_FILE = os.path.join(os.getcwd(), 'bin', 'corezmq_server.js')

    # Iterate over list of projects and get complete list of all directories and metadata objects (across all projects)
    project_to_dir_list_dict = {}
    project_to_metadatas_dict = {}
    for project_owner, project_name in PROJECT_INFOS:
        project_full_name = "%s/%s" % (project_owner, project_name)

        # Get list of all dataset directories in project workspace
        logger.info("Searching for data directories associated with project %s..." % project_full_name)
        project_sub_dir = "%s_%s" % (project_owner, project_name)
        project_to_dir_list_dict[project_full_name] = get_data_dir_list(ALC_WORKSPACE_PATH, project_sub_dir)

        # # Star the server (see bin/corezmq_server.js for more options e.g. for how to pass a pluginConfig)
        if START_ZMQ_SERVER:
            node_process = subprocess.Popen(['node', COREZMQ_SERVER_FILE, project_name,
                                             '-p', PORT,
                                             '-u', USER,
                                             '-o', project_owner],
                                            stdout=sys.stdout, stderr=sys.stderr)
            logger.info('Node-process running at PID {0}'.format(node_process.pid))

            # If the specified project does not exist, subprocess will exit with error and later calls webgme will
            # block indefinitely (no timeout option currently exists unfortunately). Simple polling loop here allows
            # time for ZMQ server to start (and fail if project is invalid), preventing lockups later.
            # Use "do-while" style loop here to make sure we wait at least as long as specified delay period.
            logger.info('Waiting to ensure node-process has started successfully...')
            startup_delay = 0
            zmq_startup_successful = True
            while True:
                node_rc = node_process.poll()
                if node_rc is not None:
                    warning_msg = "ZMQ Server exited with result code %d when trying to access project %s. " \
                                  "Will continue under the ASSUMPTION that this project no longer exists. " \
                                  % (node_rc, project_full_name)
                    logger.warning(warning_msg)
                    zmq_startup_successful = False
                    break
                if startup_delay > ZMQ_SERVER_STARTUP_DELAY_S:
                    # Assume ZMQ server has started successfully
                    logger.info('Node-process appears to have started without error.')
                    break
                time.sleep(1)
                startup_delay += 1

            if not zmq_startup_successful:
                continue


        # Create an instance of WebGME and the plugin
        logger.info("Accessing project %s through WebGME..." % project_full_name)
        webgme = WebGME(port=PORT, logger=logger)
        commit_hash = webgme.project.get_branch_hash(BRANCH_NAME)
        plugin = DataCleanup(webgme, commit_hash, BRANCH_NAME, ACTIVE_NODE_PATH, ACTIVE_SELECTION_PATHS, NAMESPACE)

        # Get dict of all metadata objects from current project and add to cumulative dict
        project_to_metadatas_dict[project_full_name] = plugin.get_project_data()

        # Cleanup for next iteration
        webgme.disconnect()
        if START_ZMQ_SERVER:
            node_process.send_signal(signal.SIGTERM)
            node_process.wait()
        logger.info("Finished accessing project %s/%s." % (project_owner, project_name))

    # Validate metadatas against directories
    project_info_dict = cross_validate_metadata(project_to_metadatas_dict, project_to_dir_list_dict, root_path=ALC_WORKSPACE_PATH)

    # TODO: Right now, only offer option to cleanup orphaned directories.
    #       Could also offer option to cleanup invalid metadata objects from models.
    #### Very simple UI with a few options ####
    # Print list of any warnings and ask user to acknowledge before continuing
    print("Before continuing, please review the following warning and error messages encountered thus far:")
    print("*********************************** WARNINGS ***********************************")
    for msg in record_handler.get_warning_msgs():
        print(msg)
    print("*********************************** ERRORS ***********************************")
    for msg in record_handler.get_error_msgs():
        print(msg)
    print("********************************************************************************")
    while True:
        user_cmd = input("Please type 'yes' to acknowledge the above warnings and continue, or 'no' to exit. ")
        if user_cmd.lower() == "yes":
            print("User acknowledged warning messages. Continuing.")
            break
        elif user_cmd.lower() == "no":
            print("Exiting.")
            exit(0)
        else:
            print("Unrecognized command. Please enter yes or no.")


    # Now iterate through each project and provide user options for what to do
    for project_name, project_info in project_info_dict.items():
        logger.info("*** Results for project %s ***" % project_name)
        print_info_summary(project_info)
        orphaned_data_moved = False
        while True:
            user_cmd = input("[move] orphaned data to archive, [list] details, or continue to [next] project? ")
            if user_cmd.lower() == "move":
                # Make sure we have not already moved this data
                if orphaned_data_moved:
                    logger.error("It appears the 'move' command has already been executed on this project.")
                    continue

                # Ensure archive directory exists
                if not os.path.isdir(ARCHIVE_DIRECTORY):
                    logger.error("No directory exists at provided archive path %s. "
                                 "Please create this directory and try again." % ARCHIVE_DIRECTORY)
                    continue

                # Move each orphaned directory
                logger.info("Moving orphaned data to archive directory at %s..." % ARCHIVE_DIRECTORY)
                for dir_info in project_info["dir_infos"]:
                    if dir_info["metadata_info"] is None:
                        # Make path absolute before moving.
                        # Preserve relative path in destination directory so move can be undone if necessary.
                        src_path = os.path.join(ALC_WORKSPACE_PATH, dir_info["path"])
                        dst_path = os.path.join(ARCHIVE_DIRECTORY, os.path.dirname(dir_info["path"]))
                        alc_common.mkdir_p(dst_path)
                        logger.info("Moving directory %s to %s..." % (src_path, dst_path))
                        shutil.move(src_path, dst_path)
                        logger.info("Done.")
                logger.info("Done moving orphaned data.")
                orphaned_data_moved = True

            elif user_cmd.lower() == "list":
                print_info_details(project_info)
            elif user_cmd.lower() == "next":
                logger.info("Continuing to next project...")
                break
            else:
                logger.info("Unrecognized command.")

    print("No more projects remaining, exiting.")


if __name__ == "__main__":
    main()
