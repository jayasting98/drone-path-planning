import tensorflow as tf


def find_direction(angular_displacement: tf.Tensor) -> tf.Tensor:
    direction = tf.concat(
        [
            tf.math.cos(angular_displacement),
            tf.math.sin(angular_displacement),
            tf.zeros(tf.shape(angular_displacement)),
        ],
        axis=-1,
    )
    return direction
