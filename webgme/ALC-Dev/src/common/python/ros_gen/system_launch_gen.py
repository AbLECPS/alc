"""
Class to generate a system-level ROS launch file from WebGME models
Author: Charlie Hartsell <charles.a.hartsell@vanderbilt.edu>

Notes:  This file uses dictionaries excessively.
        Refactoring to use classes instead would make the code more understandable and easier to maintain.
"""

import os
import jinja2
import re
import datetime
import json
import xml.etree.ElementTree as ET
from future.utils import iteritems
from plugin_utils import PluginExtensionBase

_package_directory = os.path.dirname(__file__)
_template_directory = os.path.join(_package_directory, "templates")

parseable_block_roles = ["node", "node bridge", "driver"]


class SystemLaunchGen(PluginExtensionBase):
    def __init__(self, *args, **kwargs):
        # Call base class init
        super(SystemLaunchGen, self).__init__(*args, **kwargs)

        self.core = self.parent_plugin.core
        self.root_node = self.parent_plugin.root_node
        self.target_launch_filename= ''
        self.target_launch_file=''
        self.target_ros_master_ip=''
        self.target_ros_master_port=11311
        self.local_ros_master_port_mapping=-1
        self.lec_deployment_key= {}

    def get_lec_folder(self, node_id):

        self.logger.info('*********** node_id {0}'.format(node_id))
        node = self.core.load_by_path(self.root_node, node_id)
        dataval = self.core.get_attribute(node, 'datainfo')
        self.logger.info('*********** dataval {0}'.format(dataval))
        folder = ''
        if dataval:
            # dataval = re.sub("null", 'NaN',dataval)
            try:
                datadict = json.loads(dataval, strict=False)
                self.logger.info('*********** datadict {0}'.format(datadict))
                folder  = datadict.get("directory",'')
                self.logger.info('*********** folder {0}'.format(folder))

            except Exception as e:
                self.logger.info('exception')
                pass
        return folder

    def gen_launch_file(self, active_node):
        # Tell Jinja where to find template files
        template_loader = jinja2.FileSystemLoader(searchpath=_template_directory)
        template_env = jinja2.Environment(loader=template_loader)

        # Check that active node is a valid META type
        if self.core.get_meta_type(active_node) != self.get_meta("AssemblyModel") and \
                self.core.get_meta_type(active_node) != self.get_meta("ExperimentSetup"):
            self.raise_and_notify("Active node has an invalid META type.", IOError)

        # If plugin was invoked on an Experiment model, save node
        experiment_node = None
        if self.core.get_meta_type(active_node) == self.get_meta("ExperimentSetup"):
            experiment_node = active_node

        # Find the AssemblyModel (should either be the active_node or a child)
        assembly_node = None
        if self.core.get_meta_type(active_node) != self.get_meta("AssemblyModel"):
            # Search for assembly model as a child of active_node
            for child in self.core.load_children(active_node):
                if self.core.get_meta_type(child) == self.get_meta("AssemblyModel"):
                    self.logger.info("Found Assembly Model object (%s) as a child of active node (%s)."
                                     % (self.core.get_path(child), self.core.get_path(active_node)))
                    assembly_node = child

            if assembly_node is None:
                err_msg = "Unable to find an Assembly Model as a child of active node (%s)" \
                          % self.core.get_path(active_node)
                self.raise_and_notify(err_msg, IOError)
        else:
            assembly_node = active_node

        # Check that the active assembly model contains a System Model and/or Deployment Model
        system_model_node = None
        deployment_model_node = None
        for child in self.core.load_children(assembly_node):
            if self.core.get_meta_type(child) == self.get_meta("SystemModel"):
                system_model_node = child
            if self.core.get_meta_type(child) == self.get_meta("DeploymentModel"):
                deployment_model_node = child
        if system_model_node is None:
            self.raise_and_notify("Failed to find a System Model node within the provided Assembly Model.", ValueError)

        # Get common variables
        assembly_node_name = self.core.get_attribute(assembly_node, "name")
        assembly_node_path = self.core.get_path(assembly_node)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # If a deployment model is specified, determine which nodes belong to which launch files. Otherwise,
        # use a default mapping.
        if deployment_model_node is not None:
            # Call parser function and unpack results
            container_info, deployment_map, launch_file_names, default_launch_file = self.parse_deployment_model(
                deployment_model_node)
        else:
            msg = {"message": "No Deployment model provided. Will assign all nodes to a generated default launch file.",
                   "severity": "warning"}
            self.parent_plugin.send_notification(msg)
            deployment_map = {}
            launch_file_names = []
            default_launch_file = None

        # If no default launch file is specified, create one
        if default_launch_file is None:
            msg = {"message": "No default launch file specified in Assembly model. Will create a default launch file "
                              "for all unassigned nodes.",
                   "severity": "warning"}
            self.parent_plugin.send_notification(msg)
            default_launch_file = "%s_%s.launch" % (assembly_node_name, timestamp)

        # Initialize variables for each launch file
        launch_files_data = {}
        if not (default_launch_file in launch_file_names):
            launch_file_names.append(default_launch_file)
            container_info[default_launch_file] = {}

        for file_name in launch_file_names:
            launch_files_data[file_name] = {"file_text": [],
                                            "include": [],
                                            "generate": [],
                                            "custom_statements": [],
                                            "args": [],
                                            "parameter_set": set([])}

        # For the given model, want to explore blocks at all levels of system hierarchy recursively
        # This is done over two loops. First over the Assembly model,
        # then over the higher level Experiment model (if available).
        child_nodes = self.core.load_children(assembly_node)
        params_dict = {}
        inferred_params_dict = {}
        node_path_to_launch_info_map = {}
        next_node = ''
        while len(child_nodes) > 0:
            prev_node = next_node
            next_node = child_nodes.pop()
            meta_type = self.core.get_meta_type(next_node)

            # For "Block" types, if it is a ROS Node, parse the relevant info for generating launch file
            # Otherwise, add any child blocks contained within this block for consideration
            if meta_type == self.get_meta("Block"):
                # Blocks marked as "IsImplementation" have multiple implementation alternatives
                # For these nodes, only process them if they are also marked "IsActive"
                if self.core.get_attribute(next_node, "IsImplementation"):
                    if not self.core.get_attribute(next_node, "IsActive"):
                        continue

                # Determine "role" of this block and parse accordingly
                role = self.core.get_attribute(next_node, "Role").lower()
                if role in parseable_block_roles:
                    # Get required info for launch file generation and store in map
                    launch_info = self.get_node_launch_info(next_node)
                    if launch_info is None:
                        continue
                    else:
                        node_path_to_launch_info_map[self.core.get_path(next_node)] = launch_info

                    # Check if this node contains an LEC model
                    lec_model_node = None
                    for child in self.core.load_sub_tree(next_node):
                        if self.core.get_meta_type(child) == self.get_meta("LEC_Model"):
                            lec_model_node = child
                            break

                    # If an LEC model exists, add deployment key to parameters
                    if lec_model_node is not None:
                        deployment_key = self.core.get_attribute(lec_model_node, "DeploymentKey")
                        lecdata = self.core.get_pointer_path(lec_model_node, 'ModelDataLink')
                        
                        self.logger.info('################################################')
                        self.logger.info('************ deployment key {0}'.format(deployment_key))
                        folder = self.get_lec_folder(lecdata)
                        self.logger.info(' ***********folder {0}'.format(folder))
                        self.logger.info('################################################')
                    
                        self.lec_deployment_key[deployment_key]= folder
                        



                        if deployment_key in launch_info["parameters"]:
                            # If parameter with this key already exists, notify user with a warning.
                            msg = {"message": "Node %s at path %s specifies the LEC deployment key \"%s\", but a "
                                              "parameter with this name already exists. Overwriting parameter value "
                                              "with directory of selected LEC model." %
                                              (self.core.get_attribute(lec_model_node, "name"),
                                               self.core.get_path(lec_model_node),
                                               deployment_key),
                                   "severity": "warning"}
                            self.parent_plugin.send_notification(msg)

                        # Add/Update parameter to the launch file for this deployment key
                        launch_info["parameters"][deployment_key] = "$(arg %s)" % deployment_key
                        #launch_info["parameters"][deployment_key] = folder

                        linked_objs = {self.core.get_path(next_node): deployment_key}
                        inferred_params_dict[deployment_key] = {"value": '',
                                                                "linked_objects": linked_objs,
                                                                "required": True}

                # For all other roles, add children to child_nodes
                else:
                    new_children = self.core.load_children(next_node)
                    if new_children is not None:
                        child_nodes.extend(new_children)

            # For 'Params' type nodes, add parameters to system-level parameters dictionary
            elif meta_type == self.get_meta("Params"):
                # Parse parameters and check for duplicates
                new_params = self._parse_parameters(next_node, expected_ancestor=assembly_node)
                for param_name in new_params:
                    if param_name in params_dict:
                        continue
                        # TODO: Previously threw an exception in this case. What is the desired behavior here?
                        # self.raise_and_notify("Found duplicate system parameter (%s) when parsing Params node "
                        #                       "(name: %s.%s, path: %s)." %
                        #                       (param_name,
                        #                        self.core.get_attribute(prev_node, "name"),
                        #                        self.core.get_attribute(next_node, "name"),
                        #                        self.core.get_path(next_node)),
                        #                       ValueError)

                # If no duplicates found, update system params dict
                params_dict.update(new_params)

            # For any other type of block, add children of this block to child_nodes
            else:
                new_children = self.core.load_children(next_node)
                if new_children is not None:
                    child_nodes.extend(new_children)

        # TODO: Implement this. What are the desired semantics for parameters above the assembly model level?
        # # Next, if plugin was invoked on an Experiment model, iterate over all nodes above the assembly model
        # # In this iteration, only concerned with Parameter objects
        # if experiment_node is not None:
        #     child_nodes = self.core.load_children(experiment_node)
        #     while len(child_nodes) > 0:
        #         next_node = child_nodes.pop()
        #         meta_type = self.core.get_meta_type(next_node)
        #
        #         # Skip AssemblyModel node since we have already processed it in loop above.
        #         if meta_type == self.get_meta("AssemblyModel"):
        #             continue
        #
        #         # For 'Params' type nodes, add parameters to system-level parameters dictionary
        #         elif meta_type == self.get_meta("Params"):
        #             # Parse parameters and check for duplicates
        #             new_params = self._parse_parameters(next_node, expected_ancestor=assembly_node)
        #             for param_name in new_params:
        #                 if param_name in params_dict:
        #                     self.raise_and_notify("Found duplicate system parameter (%s) when parsing Params node "
        #                                           "(name: %s, path: %s)." %
        #                                           (param_name,
        #                                            self.core.get_attribute(next_node, "name"),
        #                                            self.core.get_path(next_node)),
        #                                           ValueError)
        #
        #             # If no duplicates found, update system params dict
        #             params_dict.update(new_params)
        #
        #         # For any other type of block, add children of this block to child_nodes.
        #         else:
        #             new_children = self.core.load_children(next_node)
        #             if new_children is not None:
        #                 child_nodes.extend(new_children)

        # After all blocks have been parsed, add any inferred parameters to the system parameter dictionary.
        # Inferred parameters are a special case where duplicates are allowed, so this is done last.
        for param_name, inf_param_details in iteritems(inferred_params_dict):
            # Add to system params if not already present
            if param_name not in params_dict:
                params_dict[param_name] = inf_param_details
            # Otherwise, merge inferred parameter with existing entry
            else:
                params_dict[param_name]["linked_objects"].update(inf_param_details["linked_objects"])

        # Read all the identified system parameters and add to respective launch file parameter set(s).
        # Perform any specified linking from system-level parameters to component-level parameters.
        # Also, infer which launch file(s) a particular parameter/argument should be added to.
        # This is inferred based on the linked component-level parameters and which launch file those nodes belong to.
        unused_parameters = []
        for sys_param_name, param_details in iteritems(params_dict):
            # Find all component-level nodes that are linked to this system-level parameter and update their launch_info
            for component_node_path, component_param_name in iteritems(param_details["linked_objects"]):
                node_launch_info = node_path_to_launch_info_map[component_node_path]
                node_launch_info["parameters"][component_param_name] = "$(arg %s)" % sys_param_name

                # Check which launch file this node will be assigned to and add it to that parameter set.
                # Set data-structure is used to avoid duplicates.
                assigned_launch_file = deployment_map.get(component_node_path, default_launch_file)
                launch_files_data[assigned_launch_file]["parameter_set"].add(sys_param_name)

            # If this system parameter is not linked to any component parameters, then mark it as 'unused'
            # Possible the user has done some manual workaround for linking. Add parameter to all launch files.
            if len(param_details["linked_objects"]) == 0:
                unused_parameters.append(sys_param_name)

                for launch_file_name in launch_files_data:
                    launch_files_data[launch_file_name]["parameter_set"].add(sys_param_name)

        # Warn user of any unused parameters
        if len(unused_parameters) > 0:
            msg = {"message": "Found unused system parameters: %s." % str(unused_parameters),
                   "severity": "warning"}
            self.parent_plugin.send_notification(msg)

        # Assign each node's launch info from map to the approprate launch file and template category
        for path, launch_info in iteritems(node_path_to_launch_info_map):
            # Check which launch file this node should be assigned to
            assigned_launch_file = deployment_map.get(path, default_launch_file)

            # Add info to appropriate list based on type and the corresponding launch file
            if launch_info["type"] == "file_path":
                launch_files_data[assigned_launch_file]["include"].append(launch_info)
            elif launch_info["type"] == "custom":
                statement = launch_info["custom_statement"]
                launch_files_data[assigned_launch_file]["custom_statements"].append(statement)
            else:
                self.raise_and_notify("Unrecognized ROS launch information type.", RuntimeError)

        # Generate each desired launch file
        artifact_content = {}
        for launch_file_name, launch_file_data in iteritems(launch_files_data):
            # Fetch the parameters associated with this launch file, and split into two groups: required & optional
            optional_params = {}
            required_params = []
            for param_name in launch_file_data["parameter_set"]:
                param_details = params_dict[param_name]
                if param_details["required"]:
                    required_params.append(param_name)
                else:
                    optional_params[param_name] = param_details["value"]

            # Fill out top-level launch template
            launch_template = template_env.get_template("top_level_launch_template.launch")
            generated_launch_text = launch_template.render(model_name=assembly_node_name,
                                                           model_path=assembly_node_path,
                                                           datetime_str=timestamp,
                                                           ros_args=optional_params,
                                                           required_ros_args=required_params,
                                                           include_launch_files=launch_file_data["include"],
                                                           custom_launch_statements=launch_file_data[
                                                               "custom_statements"])

            # Add generated launch file to WebGME artifact-style dictionary
            if not (launch_file_name.endswith(".launch")):
                launch_file_name = launch_file_name + ".launch"
            artifact_content[launch_file_name] = generated_launch_text
            if (self.target_launch_filename == launch_file_name):
                self.target_launch_file = generated_launch_text

        # Add all generated files to artifact and return
        return artifact_content, container_info

    # Given a particular component node, construct 'launch_info' dictionary for this node.
    def get_node_launch_info(self, component_node):
        # Get node name and replace any whitespace with underscores
        node_name = self.core.get_attribute(component_node, "name")
        self.logger.info('parsing node launch info for component  "{0}"'.format(node_name))
        node_name = re.sub('\s+', '_', node_name).strip()

        # Find any children of interest in this node. (eg. ROSInfo, LEC Model, etc.)
        children = self.core.load_children(component_node)
        ros_info_node = None
        for child in children:
            if self.core.get_meta_type(child) == self.get_meta("ROSInfo"):
                ros_info_node = child
                break

        # ROS Node block must have 1 ROS info block to generate launch information
        if ros_info_node is None:
            self.logger.warning("Node block (%s) did not contain a ROSInfo node." % component_node["nodePath"])
            return None

        # Use ROS name, if it has been changed from the default value.
        # Otherwise, use parent node name.
        ros_node_name = self.core.get_attribute(ros_info_node, "name")
        ros_node_name_default = self.core.get_attribute(self.core.get_meta_type(ros_info_node), "name")
        if ros_node_name == ros_node_name_default:
            ros_node_name = node_name
        else:
            ros_node_name = re.sub('\s+', '_', ros_node_name).strip()

        # For launching a ROS node, look for the following options in order:
        #   1) Custom or Generated Launch Info statement
        #   2) Launch file location specified as ROS-parseable path
        generated_launch_info = self.core.get_attribute(ros_info_node, "launchInfo")
        launch_file_path = self.core.get_attribute(ros_info_node, "LaunchFileLocation")

        # FIXME: Remove "custom" support after models that use it have been updated
        # Check for Custom launch file snippet
        custom_statement = self.core.get_attribute(ros_info_node, "Custom")
        if (custom_statement is not None) and (len(custom_statement) > 0):
            launch_info = {"type": "custom",
                           "custom_statement": custom_statement,
                           "parameters": self._parse_arguments(ros_info_node)}

        # Check for generated or custom launch info
        if (generated_launch_info is not None) and (len(generated_launch_info) > 0):
            # Launch info may be a complete launch file (eg. starts with <launch>) or
            # a launch file snippet (eg. multiple root-level elements like <arg> and <node>).

            # Identify if this is a launch file snippet, or a complete file
            # Snippets may have multiple root elements, and will not necessarily parse correctly.
            # Wrap in a single dummy-root to prevent ParseError on snippets
            wrapped_launch_info = "<data>" + generated_launch_info + "</data>"
            root = ET.fromstring(wrapped_launch_info)
            # If root has only a single child of type "launch", then this is a complete file
            if (len(root) == 1) and (root[0].tag == "launch"):
                launch_info = {"type": "file_path",
                               "name": ros_node_name,
                               "file_path": launch_file_path,
                               "parameters": self._parse_arguments(ros_info_node)}
            # Otherwise, if root has one or more children, assume this is a launch file snippet
            elif len(root) >= 1:
                launch_info = {"type": "custom",
                               "custom_statement": generated_launch_info,
                               "parameters": self._parse_arguments(ros_info_node)}
            # Otherwise, raise exception
            else:
                self.raise_and_notify("Invalid launch info in ROS Node (name: %s, path %s)." %
                                      (ros_node_name, str(self.core.get_path(component_node))), ValueError)

        # Else, check if a launch file location specified as ROS-parsable path was provided
        elif (launch_file_path is not None) and (len(launch_file_path) > 0):
            launch_info = {"type": "file_path",
                           "name": ros_node_name,
                           "file_path": launch_file_path,
                           "parameters": self._parse_arguments(ros_info_node)}

        # Raise exception if no valid launch info found
        else:
            self.raise_and_notify("Could not find any valid launch info in ROS Node (name: %s, path: %s)" %
                                  (ros_node_name, str(self.core.get_path(component_node))), ValueError)

        return launch_info

    # Builds a dictionary of parameters from the provided ROS Info node. In particular, looks for any special cases
    # which need to be handled at the system-level (eg. Setting topic parameters based on currently assigned port topic)
    def _parse_arguments(self, ros_info_node):
        # Iterate over all ROS Argument nodes and check for any reference links or special cases
        param_dict = {}
        for arg_node in self.core.load_children(ros_info_node):
            # Skip any nodes which are not ROS Arguments
            if self.core.get_meta_type(arg_node) != self.get_meta("ROSArgument"):
                continue

            # Get generic info
            topic_parameter_name = self.core.get_attribute(arg_node, "name")

            # Check if this argument is linked to a particular port. If so, store topic name in appropriate param
            linked_port_path = self.core.get_pointer_path(arg_node, "linked_port")
            if linked_port_path is not None:
                linked_port_node = self.core.load_by_path(self.root_node, linked_port_path)
                linked_port_topic = self.core.get_attribute(linked_port_node, "Topic")
                param_dict[topic_parameter_name] = linked_port_topic

            # Otherwise, pass argument value only if it has been overridden from the base object value.
            else:
                # Get the updated argument value (or default if value is not set)
                new_val = self.core.get_attribute(arg_node, "value")
                if not new_val:
                    new_val = self.core.get_attribute(arg_node, "default")

                # Find the base object and if the new value is different than the base value, add to parameters dict
                base_node = self._find_first_non_meta_base(arg_node)
                if base_node is not None:
                    if new_val != self.core.get_attribute(base_node, "default"):
                        param_dict[topic_parameter_name] = new_val

        return param_dict

    # Function to parse and aggregate all parameter nodes from a given Params block.
    # Returns a parameter dictionary with the following structure:
    #
    #
    # {"<system_parameter_name>":
    #   {"value": <parameter_value>,
    #    "required": <boolean>,
    #    "linked_objects": {<component_node_path>: <component_param_name>, ...}
    #   }, ...
    # }
    def _parse_parameters(self, params_node, expected_ancestor=None):
        params_dict = {}

        # Parse each child block of this parameters block
        for param_node in self.core.load_children(params_node):
            # Ignore non-parameter blocks (eg. annotation)
            if self.core.get_meta_type(param_node) != self.get_meta("parameter"):
                continue

            # Get system-level parameter name and value, then add to params dict
            param_name = self.core.get_attribute(param_node, "name")
            param_value = self.core.get_attribute(param_node, "value")
            # If parameter value is an empty string, mark parameter as required.
            param_required = True
            if len(str(param_value)) > 0:
                param_required = False
            params_dict[param_name] = {"value": param_value, "linked_objects": {}, "required": param_required}

            # Explore all lower-level objects linked to this parameter
            for linked_path in self.core.get_member_paths(param_node, "linked_objects"):
                linked_node = self.core.load_by_path(self.root_node, linked_path)
                linked_param_name = self.core.get_attribute(linked_node, "name")

                # Verify valid META type
                if self.core.get_meta_type(linked_node) != self.get_meta("ROSArgument"):
                    self.raise_and_notify("Found node with unexpected Meta type in 'linked_parameter' set of "
                                          "parameter block (name: %s, path: %s)." %
                                          (str(self.core.get_attribute(linked_node, "name")), str(linked_path)),
                                          TypeError)

                # Verify that the linked 'ROSArgument' does NOT also have a 'linked_port' reference.
                # Cannot have both links set at one time
                if self.core.get_pointer_path(linked_node, "linked_port") is not None:
                    self.raise_and_notify("Parameter object (name: %s, path: %s) contains a link to a ROS Argument "
                                          "(name: %s, path: %s), but this argument object already has a 'linked_port'."
                                          "Cannot have both link types set simultaneously." %
                                          (str(param_name), str(self.core.get_path(param_node)),
                                           str(self.core.get_attribute(linked_node, "name")), str(linked_path)),
                                          TypeError)

                # Want to store the path to the component block which contains this argument,
                # not the path to the ROS Argument itself. Find & verify appropriate path.
                parent_component_node = self.core.get_parent(self.core.get_parent(linked_node))

                # Verify path exists
                if self.core.get_meta_type(parent_component_node) != self.get_meta("Block"):
                    self.raise_and_notify("Could not find component with expected meta type 'Block' as a parent or "
                                          "grandparent of ROS Argument object (name: %s, path: %s)." %
                                          (str(self.core.get_attribute(linked_node, "name")), str(linked_path)),
                                          TypeError)

                # If this Component is marked as "isImplementation", verify it is also marked as "isActive".
                # Otherwise, skip this linked parameter.
                if self.core.get_attribute(parent_component_node, "IsImplementation"):
                    if not self.core.get_attribute(parent_component_node, "IsActive"):
                        continue

                # Verify component path and params node we are parsing share a common ancestor within the assembly
                # model. This is necessary to prevent any invalid links from a params node in the assembly model to an
                # argument outside of the assembly (eg. in the Block Library).
                if expected_ancestor is not None:
                    shared_ancestor_node = self.core.get_common_parent([param_node, parent_component_node])
                    if not self.node_is_descendant(expected_ancestor, shared_ancestor_node):
                        self.raise_and_notify(
                            "Parameter block (name: %s, path: %s) contains link to a ROS Argument block "
                            "(name: %s, path: %s) which is NOT a member of the same assembly model."
                            "Common ancestor node is: (name: %s, path: %s)." %
                            (str(param_name), str(self.core.get_path(param_node)),
                             str(linked_param_name), str(linked_path),
                             str(self.core.get_attribute(shared_ancestor_node, "name")),
                             str(self.core.get_path(shared_ancestor_node))),
                            TypeError)

                # Store component path in params_dict accordingly
                parent_component_path = self.core.get_path(parent_component_node)
                params_dict[param_name]["linked_objects"][parent_component_path] = linked_param_name

        return params_dict

    # Parse the deployment model to determine which nodes should be placed on which launch file
    # Return a tuple containing:
    #   1) dictionary mapping node path to corresponding launch file.
    #   2) List of all identified launch file names.
    #   3) String indicating which launch file is default, or None if no default specified.
    def parse_deployment_model(self, deployment_node):
        deployment_map = {}
        default_launch_file = None
        launch_file_names = []
        container_info = {}
        for child in self.core.load_children(deployment_node):
            # Ignore any nodes that are not a "LaunchFile" type
            if self.core.get_meta_type(child) == self.get_meta("LaunchFile"):
                launch_file_name = self.core.get_attribute(child, "name")
                launch_file_names.append(launch_file_name)
                on_target = self.core.get_attribute(child,"run_on_target")
                    


                # Read and store each node path referenced as belonging to this launch file
                for linked_node_path in self.core.get_member_paths(child, "DeployedNodes"):
                    deployment_map[linked_node_path] = launch_file_name

                # Check if this launch file is the default
                if self.core.get_attribute(child, "default"):
                    default_launch_file = launch_file_name

                c_info_str = self.core.get_attribute(child, "container_info")
                try:
                    if (c_info_str):
                        c_info_str1 = json.dumps(c_info_str)
                        c_info = json.loads(c_info_str1, strict=False)
                        container_info[launch_file_name] = c_info
                        if (on_target):
                            self.target_launch_filename = launch_file_name
                            self.target_launch_file = c_info
                            self.target_ros_master_ip = self.core.get_attribute(child,"target_ip")
                            self.target_ros_master_port = self.core.get_attribute(child,"ros_master_port")
                            self.local_ros_master_port_mapping = self.core.get_attribute(deployment_node,"ros_master_port_map")
                            
                except Exception as e:
                    container_info[launch_file_name] = {}
                    self.logger.info('problem in parsing container info for launch_file "{0}"'.format(launch_file_name))
                    self.logger.info('parsing error "{0}"'.format(e))

        return container_info, deployment_map, launch_file_names, default_launch_file

    # Checks if the provided 'descendant_node' is truly a descendant of the provided 'ancestor_node'.
    # WebGME API does not provide this. Function here relies on node paths being strings with identical formatting.
    # If the descendant node path starts with the ancestor node path, then return True. Otherwise, return False.
    # Eg. "/a/b/c/d/" is a descendant of "/a/b/". It is not a descendant of "/a/e/".
    # Note, nodes are considered descendants of themselves as well. Eg. "/a/b/" is a descendant of "/a/b/"
    def node_is_descendant(self, ancestor_node, descendant_node):
        ancestor_path = self.core.get_path(ancestor_node).lower()
        descendant_path = self.core.get_path(descendant_node).lower()

        if descendant_path.startswith(ancestor_path):
            return True

        return False

    def _find_first_non_meta_base(self, node):
        """Recursively find the base of 'node' until reaching the first base object which is not a META object."""
        while node is not None:
            base_node = self.core.get_base(node)
            if base_node is None:
                return None
            elif self.core.is_meta_node(base_node):
                return node
            else:
                node = base_node
