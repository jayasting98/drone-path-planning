from __future__ import annotations
from typing import Callable
from typing import Generic
from typing import TypeVar

from drone_path_planning.graphs.common import Feature
from drone_path_planning.graphs.common import OutputFeature


T = TypeVar('T', Feature, OutputFeature)
U = TypeVar('U', Feature, OutputFeature)


class ComponentSet(Generic[T]):
    def __init__(self, features: T) -> None:
        super().__init__()
        self.features: T = features

    def map(self, features_map_function: Callable[[T], U] = lambda x: x) -> ComponentSet[U]:
        mapped_features = features_map_function(self.features)
        mapped_component_set = ComponentSet(mapped_features)
        return mapped_component_set
