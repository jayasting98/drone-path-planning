import bisect
import os
import pickle
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Tuple

from matplotlib import animation
from matplotlib import artist
from matplotlib import axes
from matplotlib import figure
import matplotlib.pyplot as plt
import tensorflow as tf

from drone_path_planning.plotters.plotter import Plotter
from drone_path_planning.utilities.constants import CHASER_DIRECTIONS
from drone_path_planning.utilities.constants import CHASER_DISPLACEMENTS
from drone_path_planning.utilities.constants import TARGET_DIRECTIONS
from drone_path_planning.utilities.constants import TARGET_DISPLACEMENTS


_TRAJECTORIES: str = 'trajectories'
_CENTER: str = 'center'
_HALF_WIDTH: str = 'half_width'
_ANIMATION_TITLE_TEMPLATE = 'Run {run:d} Step {step:d}'


class ChaserTargetPlotter(Plotter):
    def __init__(
        self,
        min_width: float,
        animation_filename: str,
        animation_figsize: Tuple[float, float],
        animation_arrow_length: float,
        animation_ms_per_frame: int,
    ):
        self._min_width = min_width
        self._animation_filename = animation_filename
        self._animation_figsize = animation_figsize
        self._animation_arrow_length = animation_arrow_length
        self._animation_ms_per_frame = animation_ms_per_frame

    def load_data(self, plot_data_dir: str):
        plot_data_filepath = os.path.join(plot_data_dir, 'plot_data.pkl')
        with open(plot_data_filepath, 'rb') as fp:
            self._raw_plot_data: List[List[Dict[str, tf.Tensor]]] = pickle.load(fp)

    def process_data(self):
        self._plot_data = dict()
        self._plot_data[_TRAJECTORIES] = list(map(self._process_trajectory, self._raw_plot_data))

    def _process_trajectory(self, trajectory: List[Dict[str, tf.Tensor]]) -> List[Dict[str, tf.Tensor]]:
        processed_trajectory = list(map(self._process_environment_state, trajectory))
        return processed_trajectory

    def _process_environment_state(self, environment_state: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        processed_environment_state = {**environment_state}
        chaser_displacements = environment_state[CHASER_DISPLACEMENTS]
        target_displacements = environment_state[TARGET_DISPLACEMENTS]
        all_displacements = tf.concat([chaser_displacements, target_displacements], axis=0)
        center = tf.math.reduce_mean(all_displacements, axis=0)
        maximum_distance_from_center = tf.math.reduce_max(tf.linalg.norm(all_displacements - center, axis=-1))
        half_width = tf.math.maximum(maximum_distance_from_center, self._min_width / 2)
        processed_environment_state[_CENTER] = center
        processed_environment_state[_HALF_WIDTH] = half_width
        return processed_environment_state

    def plot(self, plots_dir: str):
        animation_filepath = os.path.join(plots_dir, self._animation_filename)
        self._plot_animation(animation_filepath)

    def _plot_animation(self, animation_filepath: str):
        trajectories = self._plot_data[_TRAJECTORIES]
        num_trajectories = len(trajectories)
        if num_trajectories < 1:
            return
        trajectory_lengths = [len(trajectory) for trajectory in trajectories]
        trajectory_length_prefix_sums = [trajectory_lengths[0]]
        for i in range(1, num_trajectories):
            trajectory_length_prefix_sums.append(trajectory_length_prefix_sums[i - 1] + trajectory_lengths[i])
        fig = plt.figure(figsize=self._animation_figsize)
        ax = fig.add_subplot(projection='3d')
        update = self._create_animation_frame_updater(fig, ax, trajectory_length_prefix_sums, trajectories)
        num_frames = trajectory_length_prefix_sums[-1]
        func_animation = animation.FuncAnimation(fig, update, frames=num_frames, interval=self._animation_ms_per_frame)
        func_animation.save(animation_filepath)

    def _create_animation_frame_updater(
        self,
        fig: figure.Figure,
        ax: axes.Axes,
        trajectory_length_prefix_sums: List[int],
        trajectories: List[List[Dict[str, tf.Tensor]]],
    ) -> Callable[[int], Iterable[artist.Artist]]:
        def update(frame_index: int) -> Iterable[artist.Artist]:
            trajectory_index = bisect.bisect_right(trajectory_length_prefix_sums, frame_index)
            trajectory_length_prefix_sum = 0 if trajectory_index < 1 else trajectory_length_prefix_sums[trajectory_index - 1]
            environment_states = trajectories[trajectory_index]
            environment_state_index = frame_index - trajectory_length_prefix_sum
            environment_state = environment_states[environment_state_index]
            ax.clear()
            chaser_displacements = environment_state[CHASER_DISPLACEMENTS]
            chaser_directions = environment_state[CHASER_DIRECTIONS]
            target_displacements = environment_state[TARGET_DISPLACEMENTS]
            target_directions = environment_state[TARGET_DIRECTIONS]
            center = environment_state[_CENTER]
            half_width = environment_state[_HALF_WIDTH]
            ax.set_xlim3d(left=(center[0] - half_width), right=(center[0] + half_width))
            ax.set_ylim3d(bottom=(center[1] - half_width), top=(center[1] + half_width))
            ax.set_zlim3d(bottom=(center[2]), top=(center[2] + 2 * half_width))
            title = _ANIMATION_TITLE_TEMPLATE.format(run=trajectory_index, step=environment_state_index)
            ax.set_title(title)
            chaser_quiver = ax.quiver(
                chaser_displacements[:, 0],
                chaser_displacements[:, 1],
                chaser_displacements[:, 2],
                chaser_directions[:, 0],
                chaser_directions[:, 1],
                chaser_directions[:, 2],
                length=self._animation_arrow_length,
                normalize=True,
                colors=[(0.0, 0.0, 1.0, 0.9)],
            )
            target_quiver = ax.quiver(
                target_displacements[:, 0],
                target_displacements[:, 1],
                target_displacements[:, 2],
                target_directions[:, 0],
                target_directions[:, 1],
                target_directions[:, 2],
                length=self._animation_arrow_length,
                normalize=True,
                colors=[(1.0, 0.0, 0.0, 0.9)],
            )
            return fig,
        return update
