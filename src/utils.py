import os
import sys

import numpy as np
import torch


class Console:
    def __init__(self, epochs: int):
        self.__epochs = epochs
        self.__epoch = 0

    def p(self, loss: float):
        self.__epoch += 1
        percentage = self.__epoch / self.__epochs * 100
        sys.stdout.write('\r')
        sys.stdout.flush()
        sys.stdout.write(f'[{percentage:.0f}%][{loss:.4f}]: {" ".join(["." for _ in range(self.__epoch)])}')
        sys.stdout.flush()
        if self.__epoch == self.__epochs:
            sys.stdout.write('\n')


def get_number_parameters(model: torch.nn.Module) -> int:
    return sum([torch.numel(p) for p in model.parameters()])


def get_device():
    return (
        "cuda" if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


def to_one_hot(x: np.ndarray):
    """
    Converts a vector of labels to a matrix of one-hot vectors.

    :param x: the vector of labels
    :return: a matrix of one-hot vectors
    """
    n = x.shape[0]
    y = np.zeros((n, np.max(x) + 1))
    y[np.arange(n), x] = 1
    return y


DEVICE = get_device()
