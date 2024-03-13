from typing import List
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
