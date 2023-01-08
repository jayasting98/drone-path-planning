import abc
from typing import Dict
from typing import Sequence
from typing import TypeVar

import tensorflow as tf

from drone_path_planning.graphs import ComponentSet
from drone_path_planning.graphs import Graph
from drone_path_planning.graphs import NodeSet
from drone_path_planning.graphs import EdgeSet
from drone_path_planning.graphs import OutputComponentSetSpec
from drone_path_planning.graphs import OutputGraphSpec


SetDecoder = Sequence[tf.keras.layers.Layer]
SetDecoderMap = Dict[str, SetDecoder]


T = TypeVar('T', bound=ComponentSet)


class GraphDecoder(tf.keras.layers.Layer):
    def __init__(
        self,
        output_specs: OutputGraphSpec,
        latent_size: int,
        num_hidden_layers: int,
        *args,
        activation=None,
        use_bias: bool = True,
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        should_layer_normalize: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._latent_size = latent_size
        self._num_hidden_layers = num_hidden_layers
        self._activation = activation
        self._use_bias = use_bias
        self._kernel_regularizer = kernel_regularizer
        self._bias_regularizer = bias_regularizer
        self._activity_regularizer = activity_regularizer
        self._should_layer_normalize = should_layer_normalize
        self._node_output_spec_map = output_specs.node_sets
        self._edge_output_spec_map = output_specs.edge_sets
        self._node_decoders = self._create_set_decoders(output_specs.node_sets)
        self._edge_decoders = self._create_set_decoders(output_specs.edge_sets)

    def call(self, graph: Graph):
        decoded_node_sets = self._decode_node_sets(graph.node_sets)
        decoded_edge_sets = self._decode_edge_sets(graph.edge_sets)
        decoded_graph = graph._replace(node_sets=decoded_node_sets, edge_sets=decoded_edge_sets)
        return decoded_graph

    @abc.abstractmethod
    def _create_decoder(self, output_size: int) -> tf.keras.layers.Layer:
        raise NotImplementedError()

    def _create_set_decoders(self, output_spec_map: Dict[str, OutputComponentSetSpec]) -> SetDecoderMap:
        set_decoders: SetDecoderMap = dict()
        for set_name, set_output_specs in output_spec_map.items():
            set_decoder = self._create_set_decoder(set_output_specs)
            set_decoders[set_name] = set_decoder
        return set_decoders

    def _decode_node_sets(self, node_sets: Dict[str, NodeSet]) -> Dict[str, NodeSet]:
        decoded_node_sets = self._decode_sets(self._node_decoders, self._node_output_spec_map, node_sets)
        return decoded_node_sets

    def _decode_edge_sets(self, edge_sets: Dict[str, EdgeSet]) -> Dict[str, EdgeSet]:
        decoded_edge_sets = self._decode_sets(self._edge_decoders, self._edge_output_spec_map, edge_sets)
        return decoded_edge_sets

    def _decode_sets(self, set_decoders: SetDecoderMap, output_spec_map: Dict[str, OutputComponentSetSpec], encoded_sets: Dict[str, T]) -> Dict[str, T]:
        decoded_sets: Dict[str, T] = dict()
        for set_name, set_output_specs in output_spec_map.items():
            set_decoder = set_decoders[set_name]
            encoded_set = encoded_sets[set_name]
            decode = lambda x: self._decode_set(set_decoder, set_output_specs, x)
            decoded_set = encoded_set.map(decode)
            decoded_sets[set_name] = decoded_set
        return decoded_sets

    def _create_set_decoder(self, set_output_specs: OutputComponentSetSpec) -> SetDecoder:
        set_decoder: SetDecoder = list()
        for set_output_head_specs in set_output_specs:
            set_output_head_size = sum(set_output_head_specs.values())
            set_output_head_decoder = self._create_decoder(set_output_head_size)
            set_decoder.append(set_output_head_decoder)
        return set_decoder

    def _decode_set(self, set_decoder: SetDecoder, set_output_specs: OutputComponentSetSpec, encoded_set_features):
        decoded_features = dict()
        for set_output_head_decoder, set_output_head_specs in zip(set_decoder, set_output_specs):
            output_head_features = set_output_head_decoder(encoded_set_features)
            set_output_head_names = set_output_head_specs.keys()
            set_output_head_sizes = set_output_head_specs.values()
            outputs = tf.split(output_head_features, list(set_output_head_sizes), axis=-1)
            for output_name, output in zip(set_output_head_names, outputs):
                decoded_features[output_name] = output
        return decoded_features
