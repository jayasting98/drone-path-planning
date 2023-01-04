import abc


class ComponentSet:
    @abc.abstractmethod
    def replace(self, **kwargs):
        pass

    def map(self, features_map_function=None):
        mapped_features = self._features if features_map_function is None else features_map_function(self._features)
        mapped_component_set = self.replace(features=mapped_features)
        return mapped_component_set

    @property
    @abc.abstractmethod
    def _features(self):
        pass
