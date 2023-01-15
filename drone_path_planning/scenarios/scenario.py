import abc

import tensorflow as tf

from drone_path_planning.trainers import Trainer


class Scenario:
    @abc.abstractmethod
    def create_trainer(self, save_dir: str) -> Trainer:
        raise NotImplementedError()

