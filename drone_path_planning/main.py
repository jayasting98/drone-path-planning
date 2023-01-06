import argparse

from drone_path_planning.agents import SingleChaserSingleTargetAgent
from drone_path_planning.environments import SingleChaserSingleMovingTargetEnvironment
from drone_path_planning.graphs import OutputGraphSpec
from drone_path_planning.routines import ROUTINES
from drone_path_planning.utilities.constants import ANTI_CLOCKWISE
from drone_path_planning.utilities.constants import BACKWARD
from drone_path_planning.utilities.constants import CLOCKWISE
from drone_path_planning.utilities.constants import FORWARD
from drone_path_planning.utilities.constants import REST


_PROGRAM_DESCRIPTION = 'Use reinforcement learning to train a drone to plan its path.'


_SCENARIO_PARAMETERS = {
    'single-chaser_single-moving-target': {
        'agent': SingleChaserSingleTargetAgent,
        'agent_parameters': dict(
            output_specs=OutputGraphSpec(
                node_sets={
                    'self': [
                        {
                            REST: 1,
                            FORWARD: 1,
                            BACKWARD: 1,
                            ANTI_CLOCKWISE: 1,
                            CLOCKWISE: 1,
                        }
                    ],
                },
                edge_sets=dict(),
            ),
            latent_size=128,
            num_hidden_layers=2,
            num_message_passing_steps=1,
            tau=0.08,
        ),
        'training_environment': SingleChaserSingleMovingTargetEnvironment,
        'training_environment_parameters': dict(),
        'validation_environment': SingleChaserSingleMovingTargetEnvironment,
        'validation_environment_parameters': dict(),
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
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
