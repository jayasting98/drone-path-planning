import enum
from typing import Dict
from typing import NamedTuple

import tensorflow as tf


class StepType(enum.Enum):
    FIRST: tf.Tensor = tf.constant(0)
    MID: tf.Tensor = tf.constant(1)
    LAST: tf.Tensor = tf.constant(2)


class TimeStep(NamedTuple):
    step_type: StepType
    reward: tf.Tensor
    observation: Dict[str, tf.Tensor]
