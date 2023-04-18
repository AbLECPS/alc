import networkx as nx
from typing import List
import matplotlib.pyplot as plt
from .utils import select_option, tree_layout
from .node import NodeStructure, Node
from .parameter import ParameterStructure
from .model_search import search_models


SOLUTION_TYPE = "Solution"
GOAL_TYPE = "Goal"
STRATEGY_TYPE = "Strategy"
SUPPORTING_NODE_TYPES = [SOLUTION_TYPE, GOAL_TYPE, STRATEGY_TYPE]


class Pattern(object):
    def __init__(self,
                 name: str,
                 node_structure: NodeStructure,
                 parameter_structure: ParameterStructure):
        self.name = name
        self.node_structure = node_structure  # Structure of assurance argument pattern and instantiated argument
        self.parameter_structure = parameter_structure  # Structure with hierachy of formal parameters and graph of instantiated parameters

    def assign_parameters(self, model_set, base_node: Node = None, print_reason = False):
        # If a base_node was provided, initialize top-level nodes from base node
        # Otherwise, find top level node outlines and initialize node info with null parents
        if base_node is not None:
            par_info = []
            for formal_param in self.parameter_structure.top_level_formal_parameters():
                param_inst = None
                for var in base_node.parameters:
                    if var.formal_base.type == formal_param.type:
                        param_inst = self.parameter_structure.instantiate_parameter(formal_param, None, var.value, var.path)
                        break

                if param_inst is None:
                    raise ValueError("Provide base node %s did not have any matching formal parameter %s." %
                                     (str(base_node), str(formal_param)))

                # Add successor variable groups to variable info list
                for successor_group in self.parameter_structure.hierarchy.successors(formal_param):
                    par_info.append((successor_group, param_inst))
        else:
            par_info = [(var_group, None) for var_group in self.parameter_structure.top_level_formal_parameters()]

        # Iterate over graph to instantiate remaining variable groups
        for formal_param, parent in par_info:
            # Query models to find objects that satisfy constraints
            parent_model = None
            if parent is not None:
                parent_model = parent.value
            matches = search_models(model_set, formal_param.type, parent_model, shortest_paths=True)

            # Determine parameter multiplicity
            multiplicity = (1, 1)
            if parent is not None:
                hierarchy_edge = self.parameter_structure.hierarchy.edges[parent.formal_base, formal_param]["object"]
                multiplicity = hierarchy_edge.cardinality

            # Prompt user for desired variable assignment based on matched models
            if parent is not None:
                print("\nFor variable group %s with parent %s" % (str(formal_param), str(parent)))
            else:
                print("\nFor variable group %s" % str(formal_param))

            if len(matches) == 0:
                print("Could not find any solution. Will leave unassigned.")
                continue

            options = ["{0}".format(match["Name"]) for match in matches]
            if (print_reason):
                options = ["%s, Reason: %s" % (match["Name"], match["Path"]) for match in matches]
            
            selections = select_option(options, multiplicity=multiplicity, addl_opts_avail=True)

            # If user requested more options, query models again without `shortest_paths` option.
            if selections == -1:
                matches = search_models(model_set, formal_param.type, parent_model, shortest_paths=False)
                options = ["{0}".format(match["Name"]) for match in matches]
                if (print_reason):
                    options = ["%s, Reason: %s" % (match["Name"], match["Path"]) for match in matches]
                selections = select_option(options, multiplicity=multiplicity)

            # Instantiate variable for each selection and add successor variable groups to par_info
            for selection in selections:
                param_inst = self.parameter_structure.instantiate_parameter(
                    formal_param,
                    parent,
                    matches[selection]["Name"],
                    matches[selection]["Path"])

                # Add successor variables to variable info list
                for successor_group in self.parameter_structure.hierarchy.successors(formal_param):
                    par_info.append((successor_group, param_inst))

    def find_variable_ancestor(self, starting_node, variable_group):
        """Find first ancestor Node from the starting_node which contains a Variable from the variable_group"""
        if starting_node is None:
            return None
        else:
            current_node = starting_node
        while True:
            # For all variables connected to this node, check if variable is part of the selected group
            for var in current_node.parameters:
                if var.formal_base.id == variable_group.id:
                    return var

            # Get parent of current_node and continue iteration
            current_node = self.node_structure.argument.parent(current_node)

            # Raise error if we run out of nodes before finding variable
            if current_node is None:
                raise ValueError("Did not find parent node with variable in group %s" % variable_group.id)

    def instantiate_nodes(self):
        # Iterate through pattern to initialize all nodes in a top-down approach
        node_info = [(node_outline, None) for node_outline in self.node_structure.top_level_node_outlines()]
        for node_outline, parent in node_info:
            # For each variable group associated with this node outline, find parent variable group
            variable_assignments = {}
            skip_node = False
            for group_id in node_outline.formal_parameters:
                var_group = self.parameter_structure.formal_params[group_id]
                parent_var_group = self.parameter_structure.hierarchy.parent(var_group)

                # Make sure this variable group has been instantiated at least once.
                if len(var_group.instances) == 0:
                    print("Variable group %s associated with Node %s has not been instantiated. Skipping node." %
                          (str(var_group), str(node_outline)))
                    skip_node = True
                    break

                if parent_var_group is None:
                    # If this is a top-level group (i.e. no parent variable group), should be exactly 1 instance.
                    # Add it to variable_assignments.
                    if len(var_group.instances) != 1:
                        raise ValueError("Expected a single instance of Variable Group %s but found %d" %
                                         (str(var_group), len(var_group.instances)))
                    variable_assignments[group_id] = var_group.instances
                else:
                    # Otherwise, find parent variable and store all children parameters of this parent
                    # Parent variable should either be from a previously instantiated node
                    # or part of current node's variable sets.
                    if parent_var_group.id in variable_assignments:
                        parent_vars = variable_assignments[parent_var_group.id]
                        child_vars = []
                        for parent_var in parent_vars:
                            child_vars.extend(self.parameter_structure.graph.successors(parent_var))
                    else:
                        parent_var = self.find_variable_ancestor(parent, parent_var_group)
                        child_vars = self.parameter_structure.graph.successors(parent_var)

                    # Add all child variables of correct type to variable assignment set.
                    variable_assignments[group_id] = []
                    for child_var in child_vars:
                        if child_var.formal_base.id == group_id:
                            variable_assignments[group_id].append(child_var)

            # If 'skip_node' flag was set while parsing any variable group, then skip this node
            if skip_node:
                continue

            # FIXME: If a node outline has more than one variable group, currently require they all have the same
            #        number of variable instances. This could become restrictive in the future.
            var_cnt = None
            for var_set in variable_assignments.values():
                if var_cnt is None:
                    var_cnt = len(var_set)
                elif len(var_set) != var_cnt:
                    raise ValueError("For Node %s, require each variable group to have same number of instances." %
                                     str(node_outline))

            # Group paired variables into sets from variable_assignments. Done by transposing list of lists.
            variable_sets = list(zip(*variable_assignments.values()))

            # Instantiate nodes from this node outline. One instance for each set of variables.
            # For boilerplate nodes with no variables, insert mock var_set.
            if len(variable_sets) == 0:
                variable_sets = [[]]
            for var_set in variable_sets:
                node_inst = self.node_structure.instantiate_node(node_outline, parent, var_set)

                # Add successor node outlines to node_info
                for successor_outline in self.node_structure.structure.successors(node_outline):
                    node_info.append((successor_outline, node_inst))

    def instantiate(self, model_set, base_node=None, print_reason=False):
        """Function to instantiate argument from pattern based on design models describing the system.
        See paper titled 'Automated Method for Assurance Case Construction from System Design Models'
        by Hartsell et al. for a complete explanation of instantiation process."""

        # Perform variable assignment based on design models
        self.assign_parameters(model_set, base_node=base_node, print_reason=print_reason)

        # Instantiate nodes from pattern outline
        self.instantiate_nodes()

        # TODO: Add support for inferred types (e.g. automatically inferred assumptions).

        # TODO: How to handle linking to other arguments (e.g. for assumptions)?

        # TODO: Correctly handle different edge types (e.g. multiplicity, optionality, choice, etc.)

        # TODO: Add some level of variable list semantics. (e.g. list all variables in 'B' in a single node instance)

    def find_undeveloped_nodes(self) -> List[Node]:
        """Find goal nodes which are not supported by any solution, sub-goal, or strategy node."""
        u_nodes = []
        for node in self.node_structure.argument:
            if node.outline.type == GOAL_TYPE:
                undeveloped = True
                for child in self.node_structure.argument.successors(node):
                    if child.outline.type in SUPPORTING_NODE_TYPES:
                        undeveloped = False
                        break
                if undeveloped:
                    u_nodes.append(node)
        return u_nodes

    def _draw_tree(self, tree, tree_type):
        fig = plt.figure(tree_type)
        pos = tree_layout(tree)
        nx.draw(tree, pos=pos, with_labels=True)
        fig.suptitle("%s for %s" % (tree_type, self.name))
        plt.show()

    def draw_structure_graphs(self):
        self._draw_tree(self.node_structure.structure, "Pattern Structure")
        self._draw_tree(self.parameter_structure.hierarchy, "Parameter Hierarchy")

    def draw_instance_graphs(self):
        self._draw_tree(self.parameter_structure.graph, "Parameter Graph")
        self._draw_tree(self.node_structure.argument, "Argument")
