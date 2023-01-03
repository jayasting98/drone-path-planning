from typing import Dict

from drone_path_planning.graphs.edge_set import EdgeSet
from drone_path_planning.graphs.node_set import NodeSet


class Graph:
    def __init__(self, node_sets: Dict[str, NodeSet], edge_sets: Dict[str, EdgeSet]) -> None:
        self.node_sets = node_sets
        self.edge_sets = edge_sets
