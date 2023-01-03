import dataclasses
from typing import Dict
from typing import List

from drone_path_planning.graphs.edge_set import EdgeSet
from drone_path_planning.graphs.node_set import NodeSet


OutputHeadSpec = Dict[str, int]
OutputComponentSetSpec = List[OutputHeadSpec]


@dataclasses.dataclass
class OutputGraphSpec:
    node_sets: Dict[str, OutputComponentSetSpec]
    edge_sets: Dict[str, OutputComponentSetSpec]
