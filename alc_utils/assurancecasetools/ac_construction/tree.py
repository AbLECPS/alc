import networkx as nx


class Edge(object):
    def __init__(self, _type, cardinality=1):
        self.type = _type
        self.cardinality = cardinality

    def __str__(self):
        if self.cardinality != 1:
            return "%s:%s" % (str(self.type), str(self.cardinality))
        else:
            return str(self.type)


class Tree(nx.DiGraph):
    """
    Implementation of a Pseduo-Tree data structure based on networkx DiGraph.
    All nodes must have exactly one parent, except for the root node which has no parent.
    Edges between nodes may contain data objects.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.root_node = None

    def add_edge(self, u_of_edge, v_of_edge, **attr):
        # Don't allow edges to be added to nodes that already have a parent
        if self.in_degree(v_of_edge) != 0:
            raise ValueError("Cannot add another input edge to node %s" % str(v_of_edge))
        super().add_edge(u_of_edge, v_of_edge, **attr)

    def add_edges_from(self, ebunch_to_add, **attr):
        # Don't allow edges to be added to nodes that already have a parent
        for e in ebunch_to_add:
            ne = len(e)
            if ne == 3:
                u, v, dd = e
                dd.update(attr)
            elif ne == 2:
                u, v = e
                dd = attr
            else:
                raise TypeError("Edges must be provided as a 2-tuple or 3-tuple.")
            self.add_edge(u, v, **dd)

    def parent(self, node):
        """Find parent of node and ensure only one (or None) such parent exists."""
        predecessors = list(self.predecessors(node))

        if len(predecessors) > 1:
            raise ValueError("Node %s has multiple parents." % predecessors)
        elif len(predecessors) == 1:
            return predecessors[0]
        else:
            return None

    def depth(self, node):
        """Determine tree depth of node"""
        depth = 0
        parent = self.parent(node)
        while parent is not None:
            depth += 1
            parent = self.parent(parent)
        return depth

    def find_root_nodes(self):
        """Find root node(s) of tree(s)."""
        root_nodes = []
        for node, in_deg in self.in_degree:
            if in_deg == 0:
                root_nodes.append(node)

        if len(root_nodes) == 0:
            raise ValueError("No root node exists in tree.")
        return root_nodes

    def validate(self):
        """Validate graph satisfies tree constraints"""
        # Check for any cycles
        try:
            nx.find_cycle(self)
            raise ValueError("Found cycle in tree structure.")
        except nx.NetworkXNoCycle:
            pass

        # Validate all nodes have at most one parent
        for _, in_deg in self.in_degree:
            if in_deg > 1:
                raise ValueError("Found node with multiple parents in tree.")

    def serialize_edges(self):
        """'Serializes' Edge objects into string type. Useful for exporting to GraphML"""
        for u, v in self.edges():
            if "object" in self.edges[u, v]:
                self.edges[u, v]["object"] = str(self.edges[u, v]["object"])
