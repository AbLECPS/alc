from typing import Dict
import copy
from .tree import Tree, Edge


class NodeOutline(object):
    def __init__(self, _id, _type, desc, undeveloped=False, formal_parameters=None):
        self.id = _id
        if formal_parameters is None:
            formal_parameters = []
        self.type = _type
        self.desc = desc
        self.undeveloped = undeveloped
        self.formal_parameters = formal_parameters
        self.instances = []

    def __str__(self):
        return "%s: %s" % (self.type, self.id)

    def create_instance(self, variables):
        node_inst = Node(self, len(self.instances), self.undeveloped, variables)
        self.instances.append(node_inst)
        return node_inst


# TODO: Whether or not a Node is "undeveloped" is a property of the argument, not the node itself. Is this always true?
class Node(object):
    def __init__(self,
                 outline: NodeOutline,
                 instance_id: int,
                 undeveloped=False,
                 parameters=None):
        # Save arguments
        self.outline = outline
        self.instance_id = instance_id
        if parameters is None:
            parameters = []
        self.undeveloped = undeveloped
        self.parameters = parameters

        # Instantiate node description based on variable values
        self.desc = self.fill_desc_placeholders()

    def __str__(self):
        return "%s.%s" % (self.outline, self.instance_id)
        #return "%s" % (self.outline.id[0])

    def fill_desc_placeholders(self):
        """
        Function to fill all variable placeholders (e.g. {A} for variable group A) in the Node description.
        """
        # Iterate over all variables associated with this node
        desc = copy.copy(self.outline.desc)
        for var in self.parameters:
            # If variable has not been assigned yet, move to the next one
            if var.value is None:
                continue

            # Otherwise, replace placeholder in node description string with variable value.
            placeholder = "{%s}" % var.formal_base.id
            if placeholder not in desc:
                raise ValueError("Placeholder for variable group %s not found in Node %s description." %
                                 (var.formal_base.id, str(self)))
            desc = desc.replace(placeholder, str(var.value[0]))

        return desc


class NodeStructure(object):
    """Describes the pattern structure (NodeOutlines) and instantiated argument graph for an Assurance Case Pattern."""
    def __init__(self,
                 outlines: Dict[str, NodeOutline],
                 structure: Tree):
        self.outlines = outlines
        self.structure = structure
        self.argument = Tree()

    def top_level_node_outlines(self):
        return self.structure.find_root_nodes()

    def instantiate_node(self, node_outline, parent, parameters):
        # Instantiate node from outline and add to argument graph
        if node_outline not in self.outlines.values():
            raise IOError("NodeOutline %s does not exist in this NodeStructure." % str(node_outline))
        node_inst = node_outline.create_instance(parameters)
        self.argument.add_node(node_inst)

        # Verify node_outline is a valid child of parent, then add edge from parent to node_inst
        if parent is not None:
            outline_edge = self.structure.edges[parent.outline, node_outline]["object"]
            self.argument.add_edge(parent, node_inst, object=Edge(outline_edge.type))

        return node_inst

    def print_argument_nodes(self):
        print("\n\nArgument Nodes:")
        for node in self.argument.nodes:
            print("%s: %s" % (node, node.desc))
