"""
This is where the implementation of the plugin code goes.
The ROSLaunchImporter-class is imported from both run_plugin.py and run_debug.py
"""
from __future__ import print_function
from future.utils import viewitems

import sys
import logging
import xml.etree.ElementTree as ET
from webgme_bindings import PluginBase

# Setup a logger
logger = logging.getLogger('ROSLaunchImporter')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)  # By default it logs to stderr..
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


# Parse available arguments (and default values) from ROS launch XML file
def parse_launch_args(launch_file_text):
    args_dict = {}
    tree_root = ET.fromstring(launch_file_text)
    for arg in tree_root.findall('arg'):
        default_value = arg.attrib.get("default", None)
        args_dict[arg.attrib["name"]] = default_value
    return args_dict


class ROSLaunchImporter(PluginBase):
    def __init__(self, *args, **kwargs):
        # Call base class init
        super(ROSLaunchImporter, self).__init__(*args, **kwargs)

        # Support both "" and "ALCMeta" namespaces.
        if self.namespace == "":
            self.namespace_prefix = "ALCMeta."
        elif self.namespace == "ALCMeta":
            self.namespace_prefix = ""
        else:
            raise ValueError("Unsupported namespace (%s)." % self.namespace)

    def main(self):
        try:
                
            # Each ROS Node can only have 1 ROS Info block. Make sure none already exist.
            for child in self.core.load_children(self.active_node):
                meta_type = self.core.get_meta_type(child)
                if meta_type == self.META["ALCMeta.ROSInfo"]:
                    raise RuntimeError("Selected parent node (%s) already contains a ROS Info node." % self.active_node)

            # Load plugin arguments and check validity
            config_dict = self.get_current_config()

            launch_file_asset = config_dict["launch_file"]
            if (launch_file_asset is None) or (len(launch_file_asset) == 0):
                raise IOError("Provided ROS Launch file has empty asset hash.")

            launch_file_text = self.get_file(launch_file_asset)
            if (launch_file_text is None) or (len(launch_file_text) == 0):
                raise IOError("Provided ROS Launch file is empty.")

            # Check for optional arguments
            launch_file_path = config_dict.get("launch_path", None)
            import_as_parameters = config_dict.get("import_as_parameters", False)

            # Parse arguments from launch file
            args_dict = parse_launch_args(launch_file_text)

            # Create and commit appropriate block type
            if import_as_parameters:
                commit_info = self.create_param_block(args_dict)
            else:
                commit_info = self.create_ros_info(args_dict, launch_file_path)
            logger.info('committed :{0}'.format(commit_info))
            self.result_set_success(True)
        except Exception as err:
            msg = str(err)
            self.create_message(self.active_node, msg, 'error')
            self.result_set_error('ROSLaunchImporter: Error encoutered. Check result details.')
            self.result_set_success(False)
            exit()
            
    def create_ros_info(self, args_dict, launch_file_path):
        # Create ROS Info node and update attributes
        ros_info_node = self.core.create_child(self.active_node, self.META["ALCMeta.ROSInfo"])
        self.core.set_registry(ros_info_node, "position", {'x': 100, 'y': 100})
        if (launch_file_path is not None) and (len(launch_file_path) > 0):
            self.core.set_attribute(ros_info_node, "LaunchFileLocation", launch_file_path)

        # Create ROS Argument child objects for each argument
        self.create_parameter_children(args_dict, ros_info_node, self.META["ALCMeta.ROSArgument"])

        # Commit results
        commit_info = self.util.save(self.root_node, self.commit_hash, 'master',
                                     'Imported ROS Launch file into ROS Info block for node %s' % self.active_node)
        return commit_info

    def create_param_block(self, args_dict):
        # Create Params block
        param_node = self.core.create_child(self.active_node, self.META["ALCMeta.Params"])
        self.core.set_registry(param_node, "position", {'x': 200, 'y': 100})
        self.core.set_attribute(param_node, "name", self.core.get_attribute(self.active_node, "name"))

        # Create parameter child objects for each argument
        self.create_parameter_children(args_dict, param_node, self.META["ALCMeta.parameter"])

        # Commit results
        commit_info = self.util.save(self.root_node, self.commit_hash, 'master',
                                     'Imported ROS Launch file into Parameter block for node %s' % self.active_node)
        return commit_info

    def create_parameter_children(self, args_dict, parent_node, child_meta_type):
        # Create a child node for each argument found in 'args_dict'
        # Set position of child blocks explicitly so they are created in location visible to user
        pos = {'x': 100, 'y': 100}
        for name, value in viewitems(args_dict):
            arg_node = self.core.create_child(parent_node, child_meta_type)
            self.core.set_registry(arg_node, "position", pos)
            self.core.set_attribute(arg_node, "name", name)

            # Slightly different logic for different meta types
            if child_meta_type == self.META["ALCMeta.ROSArgument"]:
                if value is not None:
                    self.core.set_attribute(arg_node, "default", value)
            elif child_meta_type == self.META["ALCMeta.parameter"]:
                self.core.set_attribute(arg_node, "value", value)
                # TODO: Should we do type-inferencing here?
                # self.core.set_attribute(arg_node, "type", #TYPE)
            else:
                raise TypeError("Unsupported meta object type.")

            # Update position for next attribute block. After a certain Y-coord, start a new column.
            pos['y'] += 50
            if pos['y'] >= 700:
                pos['x'] += 150
                pos['y'] = 100
