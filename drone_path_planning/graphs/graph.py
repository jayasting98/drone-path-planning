from typing import Dict
from typing import NamedTuple

from drone_path_planning.graphs.edge_set import EdgeSet
from drone_path_planning.graphs.node_set import NodeSet


class Graph(NamedTuple):
    node_sets: Dict[str, NodeSet]
    edge_sets: Dict[str, EdgeSet]
