import torch
import numpy as np

from src import utils


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

        excluded_indices = ys == excluded
        ys = utils.to_one_hot(ys)

        self.__input_size = xs.shape[1]
        self.__output_size = ys.shape[1]

        self._meta_xs = xs[excluded_indices]
        self._meta_ys = ys[excluded_indices]
        xs = xs[~excluded_indices]
        ys = ys[~excluded_indices]

        order = np.random.permutation(xs.shape[0])
        xs = xs[order]
        ys = ys[order]

        train_n: int = int(xs.shape[0] * k)
        self._xs = xs[:train_n]
        self._ys = ys[:train_n]
        self._test_xs = xs[train_n:]
        self._test_ys = ys[train_n:]

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


if __name__ == '__main__':
    from sklearn import datasets

    iris = datasets.load_iris()
    X, Y = iris['data'], iris['target']
    data = Dataset(X, Y, utils.DEVICE)
    print(data.input_size, data.output_size)
