import tensorflow as tf


class MultiLayerPerceptron(tf.keras.layers.Layer):
    def __init__(
        self,
        output_size: int,
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
    ) -> None:
        super().__init__(*args, **kwargs)
        self._hidden_layers = [tf.keras.layers.Dense(
            latent_size,
            activation=tf.nn.relu,
            use_bias=use_bias,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
        ) for _ in range(num_hidden_layers)]
        self._output_layer = tf.keras.layers.Dense(
            output_size,
            activation=activation,
            use_bias=use_bias,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
        )
        self._should_layer_normalize = should_layer_normalize
        if should_layer_normalize:
            self._layer_normalization_layer = tf.keras.layers.LayerNormalization()

    def call(self, x, *args, **kwargs):
        for hidden_layer in self._hidden_layers:
            x = hidden_layer(x)
        x = self._output_layer(x)
        if self._should_layer_normalize:
            x = self._layer_normalization_layer(x)
        return x
