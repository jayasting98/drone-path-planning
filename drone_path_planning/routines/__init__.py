from typing import Dict

from drone_path_planning.routines.evaluate_routine import EvaluateRoutine
from drone_path_planning.routines.routine import Routine
from drone_path_planning.routines.train_routine import TrainRoutine


ROUTINES: Dict[str, Routine] = {
    'evaluate': EvaluateRoutine(),
    'train': TrainRoutine(),
}
