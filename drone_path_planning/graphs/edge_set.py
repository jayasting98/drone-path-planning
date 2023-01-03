from __future__ import annotations
from typing import Callable
from typing import TypeVar

import tensorflow as tf

from drone_path_planning.graphs.common import Feature
from drone_path_planning.graphs.common import OutputFeature
from drone_path_planning.graphs.component_set import ComponentSet


T = TypeVar('T', Feature, OutputFeature)
U = TypeVar('U', Feature, OutputFeature)


class EdgeSet(ComponentSet[T]):
    def __init__(self, features: T, senders: tf.Tensor, receivers: tf.Tensor, sender_set: str, receiver_set: str) -> None:
        super().__init__(features)
        self.senders = senders
        self.receivers = receivers
        self.sender_set = sender_set
        self.receiver_set = receiver_set

    def map(self, features_map_function: Callable[[T], U] = lambda x: x) -> EdgeSet[U]:
        mapped_features = features_map_function(self.features)
        mapped_component_set = EdgeSet(mapped_features, self.senders, self.receivers, self.sender_set, self.receiver_set)
        return mapped_component_set
