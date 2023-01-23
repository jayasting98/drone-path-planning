import argparse
from typing import Dict
from typing import Type

from drone_path_planning.routines import ROUTINES
from drone_path_planning.scenarios import Scenario
from drone_path_planning.scenarios import SingleChaserSingleMovingTargetScenario


_PROGRAM_DESCRIPTION = 'Use reinforcement learning to train a drone to plan its path.'


_SCENARIOS: Dict[str, Type[Scenario]] = {
    'single-chaser_single-moving-target': SingleChaserSingleMovingTargetScenario,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=_PROGRAM_DESCRIPTION,
    )
    parser.add_argument('routine', choices=ROUTINES)
    parser.add_argument('scenario', choices=_SCENARIOS)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--logs_dir')
    args = parser.parse_args()
    return args


def main():
    args: argparse.Namespace = _parse_args()
    routine = ROUTINES[args.routine]
    scenario = _SCENARIOS[args.scenario]()
    routine(scenario, args)


if __name__ == '__main__':
    main()
