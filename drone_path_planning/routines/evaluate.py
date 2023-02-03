import argparse

from drone_path_planning.scenarios import Scenario


def evaluate(scenario: Scenario, args: argparse.Namespace):
    evaluator = scenario.create_evaluator(args.save_dir, args.plot_data_dir, args.logs_dir)
    evaluator.initialize()
    evaluator.evaluate()
