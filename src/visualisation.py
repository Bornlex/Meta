from typing import List
from matplotlib import pyplot as plt


class TrainingStore:
    def __init__(self):
        self._losses: List[float] = []
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
        self._weights_grad.append(weights_grad)
        self._bias_grad.append(bias_grad)

    def plot(self):
        """
        Display important metrics about the training.
        :return:
        """
        fig, ax = plt.subplots(2, 2, figsize=(15, 10))

        ax[0, 0].plot(self._losses)
        ax[0, 0].set_title('loss')

        l, s = len(self._weights), len(self._weights[0])
        for i in range(s):
            series = [self._weights[j][i] for j in range(l)]
            ax[0, 1].plot(series)
        ax[0, 1].set_title('weights')

        l, s = len(self._bias), len(self._bias[0])
        for i in range(s):
            series = [self._bias[j][i] for j in range(l)]
            ax[1, 0].plot(series)
        ax[1, 0].set_title('bias')

        plt.show()
