import networkx as nx
from .utils import node_to_tuple


class ModelInterface(object):
    # Ignore 'base' pointers to avoid getting lost in the meta
    # Currently ignore 'src' and 'dst' pointers to avoid traversing connector objects
    IGNORED_POINTER_NAMES = ["base", "src", "dst"]

    def __init__(self, plugin_base, modeling_root):
        """
        Implements various networkx functions required by assurance case construction module on a WebGME model set.
        """
        self.core = plugin_base.core
        self.get_meta = plugin_base.get_meta
        self.root_node = plugin_base.root_node
        self.modeling_root = modeling_root

        self.graph = self._build_nx_graph()

    def _build_nx_graph(self):
        """Builds a NetworkX DiGraph from WebGME model database."""
        graph = nx.DiGraph()

        # Get all model objects starting from modeling_root
        subtree = self.core.load_sub_tree(self.modeling_root)

        # Add all nodes in the models to the graph
        for node in subtree:
            meta_node = self.core.get_meta_type(node)
            meta_type_name = self.core.get_attribute(meta_node, "name")
            graph.add_node(self._node_to_tuple(node), type=meta_type_name)

        # Add the containment hierarchy as edges in the graph
        for node in subtree:
            for child in self.core.load_children(node):
                graph.add_edge(self._node_to_tuple(node), self._node_to_tuple(child), type="contain")

        # For each node, add all relevant pointers and set relationships as edges in the graph
        for node in subtree:
            # Pointers
            for pointer_name in self.core.get_pointer_names(node):
                if pointer_name in self.IGNORED_POINTER_NAMES:
                    continue
                neighbor_path = self.core.get_pointer_path(node, pointer_name)
                if neighbor_path is not None:
                    neighbor = self.core.load_by_path(self.root_node, neighbor_path)
                    # Don't add edges to neighbors that aren't in the subtree
                    if neighbor["nodePath"].startswith(self.modeling_root["nodePath"]):
                        graph.add_edge(self._node_to_tuple(node), self._node_to_tuple(neighbor), type=pointer_name)

            # Sets
            for set_name in self.core.get_set_names(node):
                for neighbor in self.core.load_members(node, set_name):
                    # Don't add edges to neighbors that aren't in the subtree
                    if neighbor["nodePath"].startswith(self.modeling_root["nodePath"]):
                        graph.add_edge(self._node_to_tuple(node), self._node_to_tuple(neighbor), type=set_name)

        return graph

    def _node_to_tuple(self, node):
        return node_to_tuple(node, name=self.core.get_attribute(node, "name"))

    def nodes(self, *args, **kwargs):
        return self.graph.nodes(*args, **kwargs)

    def in_edges(self, *args, **kwargs):
        return self.graph.in_edges(*args, **kwargs)

    def out_edges(self, *args, **kwargs):
        return self.graph.out_edges(*args, **kwargs)

    # FIXME: Code below traverses WebGME models 'on-the-fly' but was extremely slow. Not sure if it is of use.
    # def nodes(self, **kwargs):
    #     """
    #     Return all valid nodes in the WebGME models.
    #     This is currently restricted to models under the 'Modeling' section of ALC.
    #
    #     :returns List[Tuple(node: WebGME node, node_attributes: Dict)]
    #     """
    #     nodes = []
    #     for node in self.core.load_sub_tree(self.modeling_root):
    #         meta_node = self.core.get_meta_type(node)
    #         meta_type_name = self.core.get_attribute(meta_node, "name")
    #         node["name"] = self.core.get_attribute(node, "name")
    #         nodes.append((node, {"type": meta_type_name}))
    #     return nodes
    #
    # def out_edges(self, node, **kwargs):
    #     """
    #     Return all edges outbound from a given node.
    #     'Edges' here includes any outgoing pointers as well as any children (i.e. a 'containment' edge).
    #
    #     :returns List[Tuple(src: WebGME node, dst: WebGME node, relationship_type: str)]
    #     """
    #     out_edges = []
    #
    #     # Find all children of this node
    #     for child in self.core.load_children(node):
    #         out_edges.append((node, child, 'contain'))
    #
    #     # Find all outgoing pointers
    #     for pointer_name in self.core.get_pointer_names(node):
    #         if pointer_name in self.IGNORED_POINTER_NAMES:
    #             continue
    #         neighbor_path = self.core.get_pointer_path(node, pointer_name)
    #         if neighbor_path is not None:
    #             neighbor = self.core.load_by_path(self.root_node, neighbor_path)
    #             out_edges.append((node, neighbor, pointer_name))
    #
    #     return out_edges
    #
    # def in_edges(self, node, **kwargs):
    #     """
    #     Return all edges inbound to a given node.
    #     'Edges' here includes pointers as well as the parent node (i.e. a 'containment' edge).
    #
    #     :returns List[Tuple(src: WebGME node, dst: WebGME node, relationship_type: str)]
    #     """
    #     in_edges = []
    #
    #     # Find the parent node, if any
    #     parent = self.core.get_parent(node)
    #     if parent is not None:
    #         in_edges.append((parent, node, 'contain'))
    #
    #     # Find all incoming pointers
    #     for pointer_name in self.core.get_collection_names(node):
    #         if pointer_name in self.IGNORED_POINTER_NAMES:
    #             continue
    #         for neighbor_path in self.core.get_collection_paths(node, pointer_name):
    #             neighbor = self.core.load_by_path(self.root_node, neighbor_path)
    #             in_edges.append((neighbor, node, pointer_name))
    #
    #     return in_edges
