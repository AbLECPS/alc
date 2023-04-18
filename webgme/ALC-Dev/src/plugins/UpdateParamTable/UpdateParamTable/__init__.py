"""
This is where the implementation of the plugin code goes.
The UpdateParamTable-class is imported from both run_plugin.py and run_debug.py
"""

import json
import logging
import sys

from webgme_bindings import PluginBase


# get_member_paths(node, name)
# get_parent(node)[source]
# get_path(node)[source]
# get_pointer_path(node, name)[source]
# load_children(node)[source]
# load_members(node, set_name)[source]
# load_pointer(node, pointer_name)[source]
# get_attribute(node, name)[source]
# set_attribute(node, name, value)[source]
# set_pointer(node, name, target)[source]
# add_member(node, name, member)[source]
# del_member(node, name, path)[source]
# del_pointer(node, name)[source]
# delete_node(node)[source]


# Setup a logger
logger = logging.getLogger('UpdateParamTable')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)  # By default it logs to stderr..
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class UpdateParamTable(PluginBase):
    def __init__(self, *args, **kwargs):
        super(UpdateParamTable, self).__init__(*args, **kwargs)
        self.params = {}
        self.errorValues = []

    def convertParam(self, params):
        definition = self.core.get_attribute(params, 'Definition')
        if (definition == ''):
            return
        logger.info(' definition = ' + definition)
        try:
            jsonval = json.loads(definition, strict=False)
        except Exception as e:
            logger.error(e)
            pfname = self.core.get_fully_qualified_name(params)
            estr = 'Unable to parse JSON input for parameter ' + pfname + ' value = ' + definition
            self.errorValues.append(estr)
            logger.error(estr)
            return

        try:
            self.processParams(jsonval, params)
        except Exception as e:
            logger.error(e)
            pfname = self.core.get_fully_qualified_name(params)
            estr = 'Unable to finish process params  ' + pfname + ' value = ' + definition
            self.errorValues.append(estr)
            logger.error(estr)

    def processParams(self, jsonval, node):
        pnodes = self.core.load_children(node)
        pinfo = {}
        for p in pnodes:
            pname = self.core.get_attribute(p, 'name')
            pinfo[pname] = p

        numchildren = len(pnodes)

        keys = jsonval.keys()
        for k in keys:
            numchildren += 1
            if k in pinfo.keys():
                p = pinfo[k]
            else:
                p = self.core.create_child(node, self.META['ALCMeta.parameter'])
                # logger.info('4')
                pos = {}
                pos['x'] = 100
                pos['y'] = 100 * (numchildren + 1)
                self.core.set_registry(p, 'position', pos)
                self.core.set_attribute(p, 'name', k)
            value = jsonval[k]
            typeval = type(value)
            typestr = 'String'
            if typeval == str:
                typestr = 'String'
            if typeval == list:
                typestr = 'Array'
                value = str(value)
            if typeval == dict:
                typestr = 'Dictionary'
                value = str(value)
            if typeval == bool:
                typestr = 'Boolean'
            if typeval == int or typeval == float:
                typestr = 'Number'
            
            self.core.set_attribute(p, 'value', value)
            self.core.set_attribute(p, 'type', typestr)

    def main_old(self):
        core = self.core
        root_node = self.root_node
        active_node = self.active_node
        active_node_meta_type = self.core.get_meta_type(self.active_node)
        if (active_node_meta_type != self.META['ALCMeta.Params']):
            self.result_set_success(False)
            self.result_set_error('UpdateParamtable can be run only on Param block')
            exit()

        base_val = self.core.get_base(self.active_node)
        if base_val == self.META['ALCMeta.Params']:
            logger.info('is base')
        else:
            logger.info('is not base')

        name = core.get_attribute(active_node, 'name')
        logger.info('ActiveNode at "{0}" has name {1}'.format(core.get_path(active_node), name))
        self.convert(self.active_node)

    def updateParameterTables(self, all_nodes):
        other_nodes = []
        for node in all_nodes:
            node_meta_type = self.core.get_meta_type(node)
            if (node_meta_type != self.META['ALCMeta.Params']):
                continue
            base_val = self.core.get_base(self.active_node)
            if (base_val == self.META['ALCMeta.Params']):
                self.convertParam(node)
            else:
                other_nodes.append(node)

        for node in other_nodes:
            self.convertParam(node)

    def convertBlock(self, block):
        role = self.core.get_attribute(block, 'Role')
        if role == 'Node Bridge':
            self.core.set_attribute(block, 'Role', 'Driver')
        elif role == 'Simulation Component':
            self.core.set_attribute(block, 'Role', 'Simulation')
        elif role == 'Subroutine':
            self.core.set_attribute(block, 'Role', 'Module')
        elif role == 'Other':
            self.core.set_attribute(block, 'Role', 'Block')

        child_nodes = self.core.load_children(block)

        for cnode in child_nodes:
            node_meta_type = self.core.get_meta_type(cnode)
            if node_meta_type != self.META['ALCMeta.SignalPort']:
                continue
            porttype = self.core.get_attribute(cnode, 'PortType')
            if porttype == 'Other':
                self.core.set_attribute(cnode, 'PortType', 'Signal')

    def updateBlocks(self, all_nodes):
        other_nodes = []
        for node in all_nodes:
            node_meta_type = self.core.get_meta_type(node)
            if node_meta_type != self.META['ALCMeta.Block']:
                continue
            base_val = self.core.get_base(self.active_node)
            if base_val == self.META['ALCMeta.Block']:
                self.convertBlock(node)
            else:
                other_nodes.append(node)

        for node in other_nodes:
            self.convertBlock(node)

    def main(self):
        all_nodes = self.core.load_sub_tree(self.root_node)
        self.updateParameterTables(all_nodes)
        self.updateBlocks(all_nodes)

        commit_info = self.util.save(self.root_node, self.commit_hash, 'master', 'Updated blocks and param tables.')
        logger.info('committed :{0}'.format(commit_info))
