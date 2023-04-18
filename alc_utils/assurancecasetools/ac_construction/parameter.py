from typing import Dict
from .tree import Tree


class Parameter(object):
    def __init__(self, formal_base, instance_id, value, path):
        self.formal_base = formal_base
        self.instance_id = instance_id
        self.value = value
        self.path = path

    def __str__(self):
        return "%s.%s" % (self.formal_base.id, self.instance_id)


class FormalParameter(object):
    def __init__(self, _id, _type):
        self.id = _id
        self.type = _type
        self.instances = []

    def __str__(self):
        return "%s::%s" % (self.id, self.type)

    def create_instance(self, value, path):
        new_var = Parameter(self, len(self.instances), value, path)
        self.instances.append(new_var)
        return new_var


class ParameterStructure(object):
    """Describes both the formal parameter hierarchy and instantiated parameter graph for an Assurance Case Pattern."""
    def __init__(self,
                 formal_params: Dict[str, FormalParameter],
                 hierarchy: Tree):
        self.formal_params = formal_params
        self.hierarchy = hierarchy
        self.graph = Tree()

    def top_level_formal_parameters(self):
        return self.hierarchy.find_root_nodes()

    def instantiate_parameter(self, formal_parameter, parent, value, path):
        # Verify group exists in this structure, instantiate variable, and add to variable graph
        if formal_parameter not in self.formal_params.values():
            raise IOError("Variable group %s does not exist in this VariableStructure." % str(formal_parameter))
        var_inst = formal_parameter.create_instance(value, path)
        self.graph.add_node(var_inst)

        # Add edge from parent to child
        if parent is not None:
            # Verify this variable group is child of parent in variable hierarchy first.
            hierarchy_edge = self.hierarchy.edges[parent.formal_base, formal_parameter]
            self.graph.add_edge(parent, var_inst)

        return var_inst

    def print_parameter_assignments(self):
        print("\n\nVariable assignments:")
        for param in self.graph.nodes:
            assignment = param.value
            if assignment is None:
                assignment = "UNASSIGNED"
            print("%s: %s" % (param, assignment))
