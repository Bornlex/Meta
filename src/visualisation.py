import numpy
import torch
import numpy as np
from typing import List, Tuple, Union
from matplotlib import pyplot as plt


class TrainingStore:
    def __init__(self):
        self._losses: List[float] = []
        self._meta_losses: List[float] = []

    def add(
        self,
        loss: float,
    ):
        self._losses.append(loss)

    def add_meta(self, predictor_loss: float, meta_loss: float):
        self._losses.append(predictor_loss)
        self._meta_losses.append(meta_loss)

    @staticmethod
    def _to_series(data: List[List[float]]):
        return [list(zip(*data))[i] for i in range(len(data[0]))]

    def plot(self):
        """
        Display important metrics about the training.
        :return:
        """
        fig, ax = plt.subplots(1, 2, figsize=(10, 8))

        ax[0].plot(self._losses)
        ax[0].set_title('loss')

        plt.show()

    def plot_meta(self):
        fig, ax = plt.subplots(1, 2, figsize=(10, 8))

        ax[0].plot(self._losses)
        ax[0].set_title('loss')

        ax[1].plot(self._meta_losses)
        ax[1].set_title('meta loss')

        plt.show()


class WeightsAnalyser:
    def __init__(self):
        self._names = []
        self._weights = []
        self._distributions: List[Tuple[float, float]] = []

    def _compute_distribution(self):
        for w in self._weights:
            self._distributions.append((w.mean(), w.std()))

    def add(self, weights: Tuple[Union[torch.Tensor, np.ndarray], str]):
        assert isinstance(weights[0], (torch.Tensor, np.ndarray))
        self._names.append(weights[1])
        if isinstance(weights[0], torch.Tensor):
            self._weights.append(weights[0].cpu().detach().numpy())
        else:
            self._weights.append(weights[0])

    def plot(self):
        self._compute_distribution()

        fig, ax = plt.subplots(len(self._weights), 2, figsize=(10, 8))
        for i in range(len(self._weights)):
            first_axis = ax[0] if len(self._weights) == 1 else ax[i, 0]
            second_axis = ax[1] if len(self._weights) == 1 else ax[i, 1]
            first_axis.hist(self._weights[i].flatten(), bins=20)
            first_axis.set_title(f'weights {self._names[i]}')
            second_axis.table(
                cellText=[
                    ['mean', 'std', 'min', 'max'],
                    [self._distributions[i][0], self._distributions[i][1], self._weights[i].min(), self._weights[i].max()]
                ],
                cellLoc='center',
                loc='center'
            )
            second_axis.set_title(f'distribution {self._names[i]}')

        plt.show()
