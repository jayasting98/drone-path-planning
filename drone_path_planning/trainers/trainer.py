import abc
from typing import List

import tensorflow as tf


class Trainer:
    @abc.abstractmethod
    def initialize(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def train(self):
        raise NotImplementedError()

    def _create_training_callbacks(
        self,
        num_epochs: int,
        num_steps_per_epoch: int,
        save_dir: str,
    ) -> List[tf.keras.callbacks.Callback]:
        callbacks = []
        progbar_logger_callback = tf.keras.callbacks.ProgbarLogger(
            count_mode='steps',
            stateful_metrics=['epsilon', 'num_episodes'],
        )
        progbar_logger_callback_params = {
            'verbose': 1,
            'epochs': num_epochs,
            'steps': num_steps_per_epoch,
        }
        progbar_logger_callback.set_params(progbar_logger_callback_params)
        callbacks.append(progbar_logger_callback)
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            save_dir,
        )
        callbacks.append(model_checkpoint_callback)
        return callbacks
