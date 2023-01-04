from typing import Dict
from typing import TypeVar

import tensorflow as tf

from drone_path_planning.graphs import ComponentSet
from drone_path_planning.graphs import Graph
from drone_path_planning.layers.basic_layers import MultiLayerPerceptron


T = TypeVar('T', bound=ComponentSet)


class GraphEncoder(tf.keras.layers.Layer):
  def __init__(
      self,
      latent_size: int,
      num_hidden_layers: int,
      *args,
      activation=None,
      use_bias: bool = True,
      kernel_regularizer=None,
      bias_regularizer=None,
      activity_regularizer=None,
      should_layer_normalize=False,
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

  def build(self, input_shape):
    self._node_encoders: Dict[str, tf.keras.layers.Layer] = {set_name: self._create_encoder() for set_name in input_shape.node_sets}
    self._edge_encoders: Dict[str, tf.keras.layers.Layer] = {set_name: self._create_encoder() for set_name in input_shape.edge_sets}

  def call(self, graph: Graph):
    encoded_node_sets = self._encode_sets(self._node_encoders, graph.node_sets)
    encoded_edge_sets = self._encode_sets(self._edge_encoders, graph.edge_sets)
    encoded_graph = graph._replace(node_sets=encoded_node_sets, edge_sets=encoded_edge_sets)
    return encoded_graph

  def _create_encoder(self) -> tf.keras.layers.Layer:
    encoder = MultiLayerPerceptron(
        self._latent_size,
        self._latent_size,
        self._num_hidden_layers,
        activation=self._activation,
        use_bias=self._use_bias,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        activity_regularizer=self._activity_regularizer,
        should_layer_normalize=self._should_layer_normalize,
    )
    return encoder

  def _encode_sets(self, encoders: Dict[str, tf.keras.layers.Layer], raw_sets: Dict[str, T]) -> Dict[str, T]:
    encoded_sets: Dict[str, T] = dict()
    for set_name in raw_sets:
        encoder = encoders[set_name]
        raw_set = raw_sets[set_name]
        encoded_set = raw_set.map(encoder)
        encoded_sets[set_name] = encoded_set
    return encoded_sets
