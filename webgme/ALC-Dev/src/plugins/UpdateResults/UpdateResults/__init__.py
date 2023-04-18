"""
This is where the implementation of the plugin code goes.
The UpdateResults-class is imported from both run_plugin.py and run_debug.py
"""
import sys
import traceback
import logging
import json
import re
import time
import os
from webgme_bindings import PluginBase

# Setup a logger
logger = logging.getLogger('UpdateResults')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)  # By default it logs to stderr..
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

result_node_meta_name    = 'ALCMeta.DEEPFORGE.pipeline.Data'
result_node_meta_name_ep = 'ALC_EP_Meta.DEEPFORGE.pipeline.Data'
result_node_meta_names   = [result_node_meta_name, result_node_meta_name_ep]

status_attribute_name    = 'jobstatus'
data_attribute_name        = 'datainfo'
result_dir_attribute_name = 'resultDir'

finished_status          = ['Finished', 'Finished_w_Errors']

slurm_exec_status_filename = 'slurm_exec_status.txt'      
result_metadata_filename   = 'result_metadata.json'



class UpdateResults(PluginBase):

    def __init__(self, webgme, commit_hash, BRANCH_NAME, ACTIVE_NODE_PATH, ACTIVE_SELECTION_PATHS, NAMESPACE):
        PluginBase.__init__(self, webgme, commit_hash, BRANCH_NAME, ACTIVE_NODE_PATH, ACTIVE_SELECTION_PATHS, NAMESPACE)
        self.updated = False
    
    def get_descendant_nodes(self, parent_node, meta_type_names):
        child_node_list = self.core.load_sub_tree(parent_node)
        node_list = []
        for child_node in child_node_list:
            if not child_node:
                continue
            child_node_meta_type = self.core.get_meta_type(child_node)
            child_node_meta_type_name = self.core.get_fully_qualified_name(child_node_meta_type)
            if child_node_meta_type_name not in meta_type_names:
                continue
            node_list.append(child_node)
        return node_list

    def update_result_nodes(self, result_nodes):
        for node in result_nodes:
            node_name = self.core.get_attribute(node,'name')
            logger.info('looking into node : {0}'.format(node_name))
            status = self.core.get_attribute(node, status_attribute_name)
            if (status in finished_status):
                continue
            result_dir = self.core.get_attribute(node, result_dir_attribute_name)
            if (result_dir == '' or not os.path.exists(result_dir)):
                continue
            slurm_exec_status = os.path.join(result_dir,slurm_exec_status_filename)
            if (not os.path.exists(slurm_exec_status)):
                logger.info('file does not exist : {0}'.format(slurm_exec_status))
                continue

            with open(slurm_exec_status,'r') as f:
                jobstatus = f.read()
            jobstatus = re.sub(r"[\n\t\s]*", "", jobstatus)
            if (jobstatus == status):
                logger.info('status did not change : {0}'.format(jobstatus))
                #continue
            else:
                self.core.set_attribute(node,status_attribute_name, jobstatus)
                self.updated = True
                logger.info('************updated status  : {0}'.format(jobstatus))
            result_meta_data = os.path.join(result_dir,result_metadata_filename)
            if (not os.path.exists(result_meta_data)):
                logger.info('file does not exist : {0}'.format(result_meta_data))
                continue
            with open(result_meta_data,'r') as f:
                result_val = f.read()
            try:
                result_json = json.loads(result_val)
                rval = json.dumps(result_json[0], indent=4, sort_keys=True)
                self.core.set_attribute(node,status_attribute_name, 'Finished')
                self.core.set_attribute(node, data_attribute_name, rval)
                logger.info('************updated datainfo')
            except Exception as e:
                    logger.info('failed to parse result_metadata.json in file : {0}'.format(result_meta_data))
                    logger.info('exception message "{0}"'.format(e))

    def main(self):
        try:
            core = self.core
            root_node = self.root_node
            active_node = self.active_node
            result_nodes = self.get_descendant_nodes(active_node,result_node_meta_names)
            self.update_result_nodes(result_nodes)
            commit_info = self.util.save(root_node, self.commit_hash, 'master', 'UpdateResults plugin')
            logger.info('committed :{0}'.format(commit_info))
        except Exception as err:
            msg = str(err)
            logger.info("exception {0}".format(msg))
            traceback_msg = traceback.format_exc()
            logger.info(traceback_msg)
            sys_exec_info_msg = sys.exc_info()[2]
            logger.info(sys_exec_info_msg)
            self.create_message(self.active_node, msg, 'error')
            self.create_message(self.active_node, traceback_msg, 'error')
            self.create_message(self.active_node, str(sys_exec_info_msg), 'error')
            self.result_set_error('UpdateResults Plugin: Error encountered.  Check result details.')
            self.result_set_success(False)
            exit()
