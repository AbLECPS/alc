"""
This is where the implementation of the plugin code goes.
The MetaParser-class is imported from both run_plugin.py and run_debug.py
"""
import sys
import traceback
import logging
import json
import re
import time
import os
import copy
from pathlib import Path
from webgme_bindings import PluginBase
from urllib.parse import urlunsplit, urljoin
import urllib.request
sys.path.append(str(Path(__file__).absolute().parent))
from future.utils import iteritems

# Setup a logger
logger = logging.getLogger('MetaParser')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)  # By default it logs to stderr..
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class MetaParser(PluginBase):
    def __init__(
            self,
            webgme, commit_hash, BRANCH_NAME, ACTIVE_NODE_PATH, ACTIVE_SELECTION_PATHS, NAMESPACE,
            config=None, **kwargs
    ):
        PluginBase.__init__(self, webgme, commit_hash, BRANCH_NAME, ACTIVE_NODE_PATH, ACTIVE_SELECTION_PATHS, NAMESPACE)
        self.meta_nodes     = {}
        self.json_meta      = {}
        self.attribute_meta = {}
        self.children_meta  = {}
        self.pointer_meta   = {}
        self.set_meta       = {}
        self.is_connection  = {}
        self.meta_names     = {}
        self.json_fqn_meta  = {}
    
    def build_meta_info(self,meta_path):
        node = self.meta_nodes[meta_path]
        self.meta_names[meta_path] = self.core.get_fully_qualified_name(node)

        self.json_meta[meta_path] = self.core.get_json_meta(node)
        #self.json_meta[meta_path]['name']=self.meta_names[meta_path]

        self.attribute_meta[meta_path] = {}
        attribute_names = self.core.get_attribute_names(node)
        for aname in attribute_names:
            self.attribute_meta[meta_path][aname]=self.core.get_attribute_meta(node,aname)
            self.attribute_meta[meta_path][aname]["value"] = self.core.get_attribute(node,aname)
        
        attribute_names = self.json_meta[meta_path]["attributes"].keys()
        for aname in attribute_names:
            self.json_meta[meta_path]["attributes"][aname]["value"]=self.core.get_attribute(node,aname)

        self.children_meta[meta_path] = self.core.get_children_meta(node)

        pointer_names = self.core.get_pointer_names(node)
        set_names     = self.core.get_set_names(node)

        self.pointer_meta[meta_path] = {}
        self.set_meta[meta_path] = {}
        for pointer_name in pointer_names:
            meta_info = self.core.get_pointer_meta(node,pointer_name)
            if (pointer_name in set_names):
                self.set_meta[meta_path][pointer_name]=meta_info
                continue
            self.pointer_meta[meta_path][pointer_name]=meta_info

        self.is_connection[meta_path] = self.core.is_connection(node)
        self.json_meta[meta_path]['is_connection']=self.is_connection[meta_path]
        self.json_meta[meta_path]['GUID']=meta_path
        self.json_fqn_meta[self.meta_names[meta_path]] = copy.deepcopy(self.json_meta[meta_path])
    
    def build_files(self):
        ret = {}
        #ret['meta_nodes.json'] = json.dumps(self.meta_names, indent =4, sort_keys=True)
        ret['json_meta.json'] = json.dumps(self.json_meta, indent=4, sort_keys=True)
        ret['json_fqn_meta.json'] = json.dumps(self.json_fqn_meta, indent=4, sort_keys=True)
        # ret['attribute_meta.json'] = json.dumps(self.attribute_meta, indent=4, sort_keys=True)
        # ret['children_meta.json'] = json.dumps(self.children_meta, indent=4, sort_keys=True)
        # ret['pointer_meta.json'] = json.dumps(self.pointer_meta, indent=4, sort_keys=True)
        # ret['set_meta.json'] = json.dumps(self.set_meta, indent=4, sort_keys=True)
        # ret['is_connection_meta.json'] = json.dumps(self.is_connection, indent=4, sort_keys=True)
        return ret

    def get_item_fqn(self, input):
        ret = {}
        counter = -1
        for guid in input["items"]:
            counter +=1
            fqn = self.meta_names[guid]
            ret[fqn]=[input["minItems"][counter],input["maxItems"][counter]]
        if "max" in input.keys():
            ret["max"] = input["max"]
        if "min" in input.keys():
            ret["min"] = input["min"]
        return ret
             
    def build_json_fqn_meta(self):
        for name in self.json_fqn_meta.keys():
            item_fqn = self.get_item_fqn(self.json_fqn_meta[name]["children"])
            self.json_fqn_meta[name]["children"] = item_fqn
            for pointer in self.json_fqn_meta[name]["pointers"].keys():
                item_fqn = self.get_item_fqn(self.json_fqn_meta[name]["pointers"][pointer])
                self.json_fqn_meta[name]["pointers"][pointer] = item_fqn

    def get_meta_info(self):
        self.meta_nodes = self.core.get_all_meta_nodes(self.active_node)
        for meta_key in self.meta_nodes.keys():
            self.build_meta_info(meta_key)
        self.build_json_fqn_meta()

    def main(self):
        try:
            self.get_meta_info()
            ret = self.build_files()
            self.add_artifact('result',ret)
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
            self.result_set_error('MetaParser Plugin: Error encountered.  Check result details.')
            self.result_set_success(False)
            exit()
        