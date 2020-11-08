from abc import ABCMeta, abstractmethod

import numpy as np


class DataSetI(ABCMeta):
    """
    Common interface for data set used by various Generative models
    """
    @property
    @abstractmethod
    def n_data_points(cls) -> int:
        ...

    @abstractmethod
    def latent_batch(self, size: int, latent_dim: int) -> np.array:
        ...

    @abstractmethod
    def real_batch(self, size: int) -> np.array:
        ...

    @abstractmethod
    def batch(self, samples: np.array, size: int) -> tuple:
        ...
