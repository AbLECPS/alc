import json
import sys
import copy
from typing import Dict, Union
from ast import literal_eval
import glob
import os

from .pattern import Pattern
from .tree import Tree, Edge
from .parameter import FormalParameter, ParameterStructure
from .node import NodeOutline, NodeStructure


_MODULE_DIR = os.path.realpath(os.path.dirname(__file__))
_DEFAULT_PATTERN_LIB_PATH = os.path.join(_MODULE_DIR, "patterns/")


def _tree_to_dict(graph):
    """Serialize a VertexStructure or NodeStructure Tree into a dictionary. Omit instance-specific data."""
    nodes = [node.__dict__ for node in graph.nodes()]
    for obj in nodes:
        if "instances" in obj:
            del obj["instances"]

    edges = []
    for source, dest in graph.edges:
        data = graph.edges[source, dest]["object"]
        obj = {"source": source.id,
               "dest": dest.id,
               "type": data.type,
               "multiplicity": str(data.multiplicity)}
        edges.append(obj)

    return {"vertices": nodes, "edges": edges}


def _tree_from_dict(graph_dict, vertex_class, argument_keys, vertices_key="vertices", edges_key="edges"):
    """Create a Tree object from a serialized dictionary representation."""
    tree = Tree()
    vertices = {}
    for vertex_dict in graph_dict[vertices_key]:
        args = [vertex_dict[key] for key in argument_keys]
        vertex = vertex_class(*args)
        vertices[vertex.id] = vertex
        tree.add_node(vertex)

    for edge_dict in graph_dict[edges_key]:
        # Special case for sys.maxsize
        if "inf" in edge_dict["multiplicity"]:
            multiplicity = []
            for x in literal_eval(edge_dict["multiplicity"].replace("inf", "-1")):
                if x < 0:
                    multiplicity.append(sys.maxsize)
                else:
                    multiplicity.append(x)
            multiplicity = tuple(multiplicity)
        else:
            multiplicity = literal_eval(edge_dict["multiplicity"])

        edge = Edge(edge_dict["type"], multiplicity)
        src = vertices[edge_dict["source"]]
        dest = vertices[edge_dict["dest"]]
        tree.add_edge(src, dest, object=edge)

    return tree, vertices


class PatternDefinition(object):
    """Contains the definition of Pattern nodes/variables and their structure.
    Provides functions for Importing/Exporting pattern definition from/to file."""
    def __init__(self,
                 name: str = None,
                 parameter_structure: ParameterStructure = None,
                 node_structure: NodeStructure = None):
        self._name: str = name
        self._parameter_structure = parameter_structure
        self._node_structure = node_structure

    @property
    def name(self):
        return self._name

    @property
    def parameter_structure(self):
        return self._parameter_structure

    @property
    def node_structure(self):
        return self._node_structure

    def __str__(self):
        if self._name is None:
            return super().__str__()
        else:
            return self._name

    def to_dict(self):
        """Serialize pattern definition into a dictionary. Omit any instance-specific data."""
        param_dict = _tree_to_dict(self._parameter_structure.hierarchy)
        node_dict = _tree_to_dict(self._node_structure.structure)
        result = {"name": self._name,
                  "parameters": {"formal_parameters": param_dict["vertices"], "edges": param_dict["edges"]},
                  "nodes": {"outlines": node_dict["vertices"], "edges": node_dict["edges"]}}
        return result

    def from_dict(self, pattern_dict):
        """Build pattern definition from a serialized dictionary representation."""
        var_hierarchy, var_groups = _tree_from_dict(
            pattern_dict["parameters"],
            FormalParameter,
            ["id", "type"],
            vertices_key="formal_parameters")

        node_structure, node_outlines = _tree_from_dict(
            pattern_dict["nodes"],
            NodeOutline,
            ["id", "type", "desc", "undeveloped", "formal_parameters"],
            vertices_key="outlines")

        self._name = pattern_dict["name"]
        self._parameter_structure = ParameterStructure(var_groups, var_hierarchy)
        self._node_structure = NodeStructure(node_outlines, node_structure)

    def export_to_file(self, directory, filename=None, **json_kwargs):
        """Serialize pattern definition and export to JSON file."""
        if (self._name is None) or (self._node_structure is None) or (self._parameter_structure is None):
            raise ValueError("Cannot export uninitialized PatternDefinition.")

        if filename is None:
            filename = "%s.json" % self._name
        path = os.path.join(directory, filename)

        kwargs = {"indent": 2}
        kwargs.update(json_kwargs)
        with open(path, 'w') as fp:
            json.dump(self.to_dict(), fp, **kwargs)

    def import_from_file(self, filepath):
        """Read JSON file and deserialize to initialize PatternDefinition."""
        with open(filepath, 'r') as fp:
            self.from_dict(json.load(fp))

    def instantiate(self):
        return Pattern(self._name,
                       copy.deepcopy(self._node_structure),
                       copy.deepcopy(self._parameter_structure))


class PatternLibrary(object):
    """Class for loading patterns from description files and searching attributes of these patterns.
    Also acts as Factory for creating instances of each pattern."""

    def __init__(self):
        self._pattern_definitions: Union[Dict[str, PatternDefinition], None] = None

    def load_default_lib(self):
        self.load_library(_DEFAULT_PATTERN_LIB_PATH)

    def load_library(self, path):
        self._pattern_definitions = {}
        for pattern_file in glob.glob(os.path.join(path, "*.json")):
            pd = PatternDefinition()
            pd.import_from_file(pattern_file)
            self._pattern_definitions[pd.name] = pd

    def find_by_root_variable_type(self, var_type):
        matching_patterns = []
        for pattern in self._pattern_definitions.values():
            for root_var_group in pattern.parameter_structure.hierarchy.find_root_nodes():
                if root_var_group.type == var_type:
                    matching_patterns.append(pattern)
        return matching_patterns

    def find_by_name(self, name):
        if name in self._pattern_definitions:
            return self._pattern_definitions[name]
        else:
            return None

    def list_patterns(self):
        return list(self._pattern_definitions.values())
