import abc


class Plotter:
    @abc.abstractmethod
    def load_data(self, plot_data_dir: str):
        raise NotImplementedError()

    @abc.abstractmethod
    def process_data(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def plot(self, plots_dir: str):
        raise NotImplementedError()
