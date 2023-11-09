import os
import sys
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
