from typing import Dict
from typing import Type

from drone_path_planning.scenarios.scenario import Scenario
from drone_path_planning.scenarios.single_chaser_single_moving_target_scenario import SingleChaserSingleMovingTargetScenario


SCENARIOS: Dict[str, Type[Scenario]] = {
    'single-chaser_single-moving-target': SingleChaserSingleMovingTargetScenario,
}
