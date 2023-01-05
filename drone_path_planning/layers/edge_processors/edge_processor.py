import abc

import tensorflow as tf


class EdgeProcessor(tf.keras.layers.Layer):
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
        self._edge_layer = self._create_edge_layer(
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
        sender_features, receiver_features, edge_set = inputs
        edge_features = edge_set.features
        sender_nodes = edge_set.senders
        receiver_nodes = edge_set.receivers
        gathered_sender_features = tf.gather(sender_features, sender_nodes, axis=-2, batch_dims=-1)
        gathered_receiver_features = tf.gather(receiver_features, receiver_nodes, axis=-2, batch_dims=-1)
        features_list = [gathered_sender_features, gathered_receiver_features, edge_features]
        features = tf.concat(features_list, axis=-1)
        new_edge_features = self._edge_layer(features)
        return new_edge_features

    @abc.abstractmethod
    def _create_edge_layer(
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
