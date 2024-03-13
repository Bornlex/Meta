import torch
import numpy as np
from torchvision import datasets, transforms

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
        self.__meta = False

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

    def meta(self):
        self.__meta = True

    def start_over(self):
        self.__index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.__meta:
            xs = self._meta_xs
            ys = self._meta_ys
        else:
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


class MNIST:
    def __init__(self, batch_size: int, device):
        self._device = device
        self._transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.Lambda(lambda x: torch.flatten(x)),
        ])
        self._train_set = datasets.MNIST('~/.pytorch/mnist/', download=True, train=True, transform=self._transform)
        self._train_loader = torch.utils.data.DataLoader(self._train_set, batch_size=batch_size, shuffle=True)
        self._test_set = datasets.MNIST('~/.pytorch/mnist/', download=True, train=False, transform=self._transform)
        self._test_loader = torch.utils.data.DataLoader(self._test_set, batch_size=batch_size, shuffle=True)

    @property
    def input_size(self) -> int:
        return 28 * 28

    @property
    def output_size(self) -> int:
        return 10

    def train(self):
        x, y = next(iter(self._train_loader))
        y = torch.eye(self.output_size)[y].to(self._device)
        return x.to(self._device), y

    def test(self):
        x, y = next(iter(self._test_loader))
        y = torch.eye(self.output_size)[y].to(self._device)
        return x.to(self._device), y


def kolmogorov_dataloader(n_batches, batch_size, length):
    """
    Each sequence returned contains 2 channels. The first channel is a sequence of random numbers. The second channel
    contains the delimiter.

    :param n_batches: the number of batches to yield
    :param batch_size: the size of each batch
    :param length: the length of the sequence
    :return: a generator
    """
    for b in range(n_batches):
        sequence = np.random.randint(0, 10, (batch_size, length, 2))
        sequence[:, :, 1] = 0
        sequence[:, -1, 1] = 1
        inp = torch.from_numpy(sequence).float()
        out = inp.clone()

        yield inp, out


def kolmogorov_dataloader_simple(n_batches, length):
    """
    This function is a copy of the previous function but with simpler tasks.
    It involves increasing or decreasing sequences.

    Also, the batch size if fixed to 8.
    """
    seed = np.random.randint(0, 10)

    for b in range(n_batches):
        sequence = np.zeros((8, length, 2))
        sequence[0, :, 0] = [seed + i for i in range(length)]
        sequence[1, :, 0] = [seed + 2 * i for i in range(length)]
        sequence[2, :, 0] = [seed + i ** 2 for i in range(length)]
        sequence[3, :, 0] = [seed - i for i in range(length)]
        sequence[4, :, 0] = [seed - 4 * i for i in range(length)]
        sequence[5, :, 0] = [seed - 2 * i for i in range(length)]
        sequence[6, :, 0] = [seed * i for i in range(length)]
        sequence[7, :, 0] = [2 * seed - i for i in range(length)]
        sequence[:, -1, 1] = 1
        inp = torch.from_numpy(sequence).float()
        out = inp.clone()

        yield inp, out


if __name__ == '__main__':
    from sklearn import datasets

    iris = datasets.load_iris()
    X, Y = iris['data'], iris['target']
    data = Dataset(X, Y, utils.DEVICE)
    print(data.input_size, data.output_size)

    for i, o in kolmogorov_dataloader_simple(2, 10):
        print(i, o)
