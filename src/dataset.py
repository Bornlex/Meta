import torch
import numpy as np


class Dataset:
    def __init__(self, xs: np.ndarray, ys: np.ndarray, device, excluded: int = 2, k: float = .8):
        """
        Prepares the IRIS dataset. Performs the following steps:
        1. remove the last class (with label == 2)
        2. split train and test datasets
        3. shuffle the dataset
        4. create an iterator

        :param xs: the examples
        :param ys: the labels
        :param k: the fraction of the dataset to use for training
        """
        self.__device = device

        self.__input_size = xs.shape[1]
        if len(ys.shape) == 1:
            ys = np.expand_dims(ys, axis=-1)

        self.__output_size = ys.shape[1]

        self._xs = xs[ys[:, 0] != excluded]
        self._ys = ys[ys[:, 0] != excluded]
        self._meta_xs = xs[ys[:, 0] == 2]
        self._meta_ys = ys[ys[:, 0] == 2]

        order = np.random.permutation(self._xs.shape[0])
        self._xs = self._xs[order]
        self._ys = self._ys[order]

        train_n: int = int(self._xs.shape[0] * k)
        self._xs = self._xs[:train_n]
        self._ys = self._ys[:train_n]
        self._test_xs = self._xs[train_n:]
        self._test_ys = self._ys[train_n:]

        self._xs = torch.from_numpy(self._xs).to(self.__device, dtype=torch.float32)
        self._ys = torch.from_numpy(self._ys).to(self.__device, dtype=torch.float32)
        self._test_xs = torch.from_numpy(self._test_xs).to(self.__device, dtype=torch.float32)
        self._test_ys = torch.from_numpy(self._test_ys).to(self.__device, dtype=torch.float32)
        self._meta_xs = torch.from_numpy(self._meta_xs).to(self.__device, dtype=torch.float32)
        self._meta_ys = torch.from_numpy(self._meta_ys).to(self.__device, dtype=torch.float32)

        self.__index = 0

        self.__training = True

    @property
    def input_size(self) -> int:
        return self.__input_size

    @property
    def output_size(self) -> int:
        return self.__output_size

    def test(self):
        self.__training = False

    def train(self):
        self.__training = True

    def start_over(self):
        self.__index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.__training:
            xs = self._xs
            ys = self._ys
        else:
            xs = self._test_xs
            ys = self._test_ys

        if self.__index >= xs.shape[0]:
            raise StopIteration

        a, b = xs[self.__index:self.__index + 1], ys[self.__index:self.__index + 1]
        self.__index += 1
        return a, b
