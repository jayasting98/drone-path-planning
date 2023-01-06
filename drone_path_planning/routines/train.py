from typing import Any
from typing import Dict
from typing import List

import tensorflow as tf

from drone_path_planning.agents import DeepQNetworkAgent
from drone_path_planning.environments import Environment
from drone_path_planning.environments import StepType
from drone_path_planning.utilities.training_helpers import ReplayBuffer
from drone_path_planning.utilities.training_helpers import Transition


INITIAL_LEARNING_RATE = 1e-5
NUM_STEPS_PER_FULL_LEARNING_RATE_DECAY = 524288
LEARNING_RATE_DECAY_RATE = 0.9999956082
NUM_ITERATIONS: int = 2097152
MAX_NUM_STEPS_PER_EPISODE: int = 256
NUM_EVAL_EPISODES: int = 16
NUM_STEPS_PER_EPOCH: int = MAX_NUM_STEPS_PER_EPISODE * 8
NUM_EPOCHS: int = NUM_ITERATIONS // NUM_STEPS_PER_EPOCH
REPLAY_BUFFER_SIZE: int = MAX_NUM_STEPS_PER_EPISODE * 64


def _create_training_callbacks(num_epochs: int, num_steps_per_epoch: int) -> List[tf.keras.callbacks.Callback]:
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
  return callbacks


def _validate(
    agent: DeepQNetworkAgent,
    environment: Environment,
    callback_list_callback: tf.keras.callbacks.CallbackList,
    num_eval_episodes: int,
    max_num_steps_per_episode: int,
):
    test_logs = dict()
    callback_list_callback.on_test_begin()
    for i in range(num_eval_episodes):
        callback_list_callback.on_test_batch_begin(i)
        batch_logs = agent.evaluate_step(environment, max_num_steps_per_episode)
        callback_list_callback.on_test_batch_end(float(i), logs=batch_logs)
        for key, value in batch_logs.items():
            if key not in test_logs:
                test_logs[key] = []
            test_logs[key].append(value)
    test_logs = {key: tf.reduce_mean(tf.stack(values)) for key, values in test_logs.items()}
    callback_list_callback.on_test_end(logs=test_logs)
    return test_logs


def _train(
    agent: DeepQNetworkAgent,
    training_environment: Environment,
    replay_buffer: ReplayBuffer,
    callback_list_callback: tf.keras.callbacks.CallbackList,
    num_epochs: int,
    num_steps_per_epoch: int,
    max_num_steps_per_episode: int,
    num_eval_episodes: int,
    validation_environment: Environment = None
):
    train_step = tf.function(agent.train_step)
    collect_step = tf.function(agent.collect_step)
    time_step = training_environment.reset()
    k = 0
    num_episodes = 0
    training_logs: Dict[str, tf.Tensor]
    callback_list_callback.on_train_begin()
    for i in range(num_epochs):
        epoch_logs: Dict[str, tf.Tensor]
        callback_list_callback.on_epoch_begin(i)
        for j in range(num_steps_per_epoch):
            action = collect_step(time_step)
            next_time_step = training_environment.step(action)
            transition = Transition(time_step, action, next_time_step)
            replay_buffer.append(transition)
            k += 1
            if k < max_num_steps_per_episode and time_step.step_type != StepType.LAST:
                time_step = next_time_step
            else:
                time_step = training_environment.reset()
                k = 0
                num_episodes += 1
            training_transition = replay_buffer.sample()
            callback_list_callback.on_train_batch_begin(j)
            batch_logs = train_step(training_transition)
            batch_logs['num_episodes'] = num_episodes
            callback_list_callback.on_train_batch_end(float(j), logs=batch_logs)
            agent.update_epsilon()
            agent.update_target_model()
            replay_buffer.remove_oldest()
            epoch_logs = batch_logs
        if validation_environment is not None:
            val_logs = _validate(
                agent,
                validation_environment,
                callback_list_callback,
                num_eval_episodes,
                max_num_steps_per_episode,
            )
            epoch_logs.update({f'val_{key}': value for key, value in val_logs.items()})
        callback_list_callback.on_epoch_end(float(i), logs=epoch_logs)
        training_logs = epoch_logs
    callback_list_callback.on_train_end(logs=training_logs)


def train(scenario: Dict[str, Any]):
    agent = scenario['agent'](**scenario['agent_parameters'])
    training_environment = scenario['training_environment'](**scenario['training_environment_parameters'])
    validation_environment = scenario['validation_environment'](**scenario['validation_environment_parameters'])
    replay_buffer = ReplayBuffer()
    replay_buffer.fill(tf.function(agent.collect_step), training_environment, REPLAY_BUFFER_SIZE - 1, MAX_NUM_STEPS_PER_EPISODE)
    exponential_decay_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        INITIAL_LEARNING_RATE,
        NUM_STEPS_PER_FULL_LEARNING_RATE_DECAY,
        LEARNING_RATE_DECAY_RATE,
    )
    adam_optimizer = tf.keras.optimizers.Adam(learning_rate=exponential_decay_schedule)
    agent.compile(optimizer=adam_optimizer)
    callbacks = _create_training_callbacks(
        NUM_EPOCHS,
        NUM_STEPS_PER_EPOCH,
    )
    callback_list_callback = tf.keras.callbacks.CallbackList(
        callbacks=callbacks,
        model=agent,
    )
    _train(
        agent,
        training_environment,
        replay_buffer,
        callback_list_callback,
        NUM_EPOCHS,
        NUM_STEPS_PER_EPOCH,
        MAX_NUM_STEPS_PER_EPISODE,
        NUM_EVAL_EPISODES,
        validation_environment=validation_environment,
    )
