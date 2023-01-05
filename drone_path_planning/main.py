import argparse

from drone_path_planning.routines import ROUTINES


_PROGRAM_NAME = 'Drone Path Planning Reinforcement Learning Trainer'
_PROGRAM_DESCRIPTION = 'Use reinforcement learning to train a drone to plan its path.'


_SCENARIO_PARAMETERS = {
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=_PROGRAM_NAME,
        description=_PROGRAM_DESCRIPTION,
    )
    parser.add_argument('routine', choices=ROUTINES)
    parser.add_argument('scenario', choices=_SCENARIO_PARAMETERS)
    args = parser.parse_args()
    return args


def main():
    args: argparse.Namespace = _parse_args()
    routine = ROUTINES[args.routine]
    scenario = _SCENARIO_PARAMETERS[args.scenario]
    agent = scenario['agent'](**scenario['agent_parameters'])
    training_environment = scenario['training_environment'](**scenario['training_environment_parameters'])
    validation_environment = scenario['validation_environment'](**scenario['validation_environment_parameters'])
    routine(agent, training_environment, validation_environment)


if __name__ == '__main__':
    main()
