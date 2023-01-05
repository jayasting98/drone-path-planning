import abc

import tensorflow as tf

from drone_path_planning.environments.time_step import TimeStep


class Environment:
  def reset(self):
    self.current_time_step = self._reset()
    return self.current_time_step

  def step(self, action: tf.Tensor) -> TimeStep:
    self.current_time_step = self._step(action)
    return self._current_time_step

  @property
  def current_time_step(self) -> TimeStep:
    if not hasattr(self, '_current_time_step'):
      self._current_time_step = self.reset()
    return self._current_time_step

  @current_time_step.setter
  def current_time_step(self, value):
    self._current_time_step = value

  @abc.abstractmethod
  def render(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def _reset(self) -> TimeStep:
    raise NotImplementedError()

  @abc.abstractmethod
  def _step(self, action: tf.Tensor) -> TimeStep:
    raise NotImplementedError()