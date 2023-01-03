from __future__ import annotations
from typing import Callable
from typing import TypeVar

from drone_path_planning.graphs.common import Feature
from drone_path_planning.graphs.common import OutputFeature
from drone_path_planning.graphs.component_set import ComponentSet


T = TypeVar('T', Feature, OutputFeature)
U = TypeVar('U', Feature, OutputFeature)


class NodeSet(ComponentSet[T]):
    def map(self, features_map_function: Callable[[T], U] = lambda x: x) -> NodeSet[U]:
        mapped_features = features_map_function(self.features)
        mapped_component_set = NodeSet(mapped_features)
        return mapped_component_set
