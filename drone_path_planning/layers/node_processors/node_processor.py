import abc

import tensorflow as tf


class NodeProcessor(tf.keras.layers.Layer):
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
    ) -> None:
        super().__init__(*args, **kwargs)
        self._node_layer = self._create_node_layer(
            latent_size,
            num_hidden_layers,
            activation=activation,
            use_bias=use_bias,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            should_layer_normalize=should_layer_normalize,
        )

    def call(self, inputs, *args, **kwargs):
        node_features, messages = inputs
        features_list = [node_features, *messages]
        features = tf.concat(features_list, axis=-1)
        new_node_features = self._node_layer(features)
        return new_node_features

    @abc.abstractmethod
    def _create_node_layer(
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
    ) -> tf.keras.layers.Layer:
        raise NotImplementedError()
