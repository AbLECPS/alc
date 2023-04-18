from typing import List, Optional, Tuple
import networkx as nx
import matplotlib.pyplot as plt

from .node import Node
from .pattern import Pattern
from .tree import Tree
from .utils import tree_layout


class AssuranceCase(object):
    """Composition of multiple argument fragments.
    Fragments may be either instantiated Patterns or hand-made arguments."""

    def __init__(self):
        self.graph = Tree()
        self.pattern_info: List[Tuple[Pattern, Node]] = []

    def add_pattern(self,
                    pattern: Pattern,
                    parent_node: Optional[Node] = None):
        # Verify parent exists in graph, if not None
        if parent_node is not None:
            if parent_node not in self.graph.nodes:
                raise IOError("Parent node %s does not exist in assurance case." % str(parent_node))

        # Compose fragment with existing assurance case, then add edge from parent to fragment root(s).
        self.pattern_info.append((pattern, parent_node))
        self.graph = nx.algorithms.operators.compose(self.graph, pattern.node_structure.argument)
        if parent_node is not None:
            for root in pattern.node_structure.argument.find_root_nodes():
                self.graph.add_edge(parent_node, root, type="reference")

    def draw(self):
        pos = tree_layout(self.graph)
        nx.draw(self.graph, pos=pos, with_labels=True)
        plt.show()
