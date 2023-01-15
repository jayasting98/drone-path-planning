import argparse

from drone_path_planning.scenarios import Scenario


def train(scenario: Scenario, args: argparse.Namespace):
    trainer = scenario.create_trainer()
    trainer.initialize()
    trainer.train()
