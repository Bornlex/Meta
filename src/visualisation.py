from typing import List
from matplotlib import pyplot as plt


class TrainingStore:
    def __init__(self):
        self._losses: List[float] = []
        self._meta_losses: List[float] = []
        self._weights: List[List[float]] = []
        self._bias: List[List[float]] = []
        self._weights_grad: List[List[float]] = []
        self._bias_grad: List[List[float]] = []

    def add(
        self,
        loss: float,
        weights: List[float],
        bias: List[float],
        weights_grad: List[float] = None,
        bias_grad: List[float] = None
    ):
        self._losses.append(loss)
        self._weights.append(weights)
        self._bias.append(bias)
        if weights_grad is not None:
            self._weights_grad.append(weights_grad)
        if bias_grad is not None:
            self._bias_grad.append(bias_grad)

    def add_meta(self, predictor_loss: float, meta_loss: float, weights: List[float], bias: List[float]):
        self._losses.append(predictor_loss)
        self._meta_losses.append(meta_loss)
        self._weights.append(weights)
        self._bias.append(bias)

    @staticmethod
    def _to_series(data: List[List[float]]):
        return [list(zip(*data))[i] for i in range(len(data[0]))]

    def plot(self):
        """
        Display important metrics about the training.
        :return:
        """
        fig, ax = plt.subplots(2, 2, figsize=(15, 10))

        ax[0, 0].plot(self._losses)
        ax[0, 0].set_title('loss')

        for series in self._to_series(self._weights):
            ax[0, 1].plot(series)
        ax[0, 1].set_title('weights')

        for series in self._to_series(self._bias):
            ax[1, 0].plot(series)
        ax[1, 0].set_title('bias')

        if len(self._weights_grad) > 0:
            for series in self._to_series(self._weights_grad):
                ax[1, 1].plot(series)
            ax[1, 1].set_title('weights gradients')

        plt.show()

    def plot_meta(self):
        fig, ax = plt.subplots(2, 2, figsize=(15, 10))

        ax[0, 0].plot(self._losses)
        ax[0, 0].set_title('loss')

        ax[0, 1].plot(self._meta_losses)
        ax[0, 1].set_title('meta loss')

        for series in self._to_series(self._weights):
            ax[1, 0].plot(series)
        ax[1, 0].set_title('weights')

        for series in self._to_series(self._bias):
            ax[1, 1].plot(series)
        ax[1, 1].set_title('bias')

        plt.show()
