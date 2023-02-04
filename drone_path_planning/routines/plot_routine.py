import argparse
import os

from drone_path_planning.routines.routine import Routine
from drone_path_planning.scenarios import SCENARIOS


class PlotRoutine(Routine):
    def setup_parser(self, parser: argparse.ArgumentParser):
        parser.add_argument('scenario', choices=SCENARIOS)
        parser.add_argument('--plot_data_dir', required=True)
        parser.add_argument('--plots_dir', required=True)

    def run(self, args: argparse.Namespace):
        scenario = SCENARIOS[args.scenario]()
        plotter = scenario.create_plotter()
        plotter.load_data(args.plot_data_dir)
        plotter.process_data()
        os.makedirs(args.plots_dir, exist_ok=True)
        plotter.plot(args.plots_dir)
