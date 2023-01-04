import abc

import tensorflow as tf

from drone_path_planning.graphs import Graph
from drone_path_planning.layers.graph_decoders import GraphDecoder
from drone_path_planning.layers.graph_encoders import GraphEncoder
from drone_path_planning.layers.graph_processors import GraphProcessor


class EncodeProcessDecode(tf.keras.layers.Layer):
    @property
    @abc.abstractmethod
    def encoder(self) -> GraphEncoder:
        pass

    @property
    @abc.abstractmethod
    def processor(self) -> GraphProcessor:
        pass

    @property
    @abc.abstractmethod
    def decoder(self) -> GraphDecoder:
        pass

    def call(self, graph: Graph):
        encoded_graph = self.encoder(graph)
        processed_graph = self.processor(encoded_graph)
        decoded_graph = self.decoder(processed_graph)
        return decoded_graph
