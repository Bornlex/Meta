import sys
import torch
from torch import nn
from sklearn import datasets

from src import utils
from src import dataset


_DEVICE = utils.get_device()


class Meta(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._output_size = output_size

        self._rnn = nn.RNN(self._input_size, self._hidden_size, None, batch_first=True)
        self._hidden_state = torch.randn(1, self._input_size, self._hidden_size).to(_DEVICE)
        self._fc = nn.Linear(self._hidden_size, self._output_size)

    def forward(self, x):
        """
        Updates the knowledge of the meta-learner after seeing another example.

        WARNING: x contains both the data and the label associated with it, they are concatenated.

        :param x: an example from the training set
        :return: the predicted weights of the prediction network
        """
        out, self._hidden_state = self._rnn(x, self._hidden_state)
        out = self._fc(out)
        return out


class Predictor(nn.Module):
    """
    A very simple model to make sure that the overall system works.
    y = ax + b
    """
    def __init__(self, input_size, output_size, weights: torch.Tensor, bias: torch.Tensor):
        super().__init__()
        self._input_size = input_size
        self._output_size = output_size
        self._fc = nn.Linear(self._input_size, self._output_size)
        self._fc.weight = nn.Parameter(weights.T)
        self._fc.bias = nn.Parameter(bias)

    def forward(self, x: torch.Tensor):
        """
        Make a linear prediction from an input.
        y = a * x + b

        :param x: the input
        :return: the prediction
        """
        return self._fc(x)


def load_dataset():
    iris = datasets.load_iris()
    return iris['data'], iris['target']


def train_predictor_alone(dataset: dataset.Dataset, epochs: int = 10, lr: float = 0.01):
    linear_model = Predictor(
        dataset.input_size,
        dataset.output_size,
        torch.randn(dataset.input_size, dataset.output_size),
        torch.randn(dataset.output_size)
    )
    linear_model.to(_DEVICE)

    numel = utils.get_number_parameters(linear_model)
    print(f'[parameters]: {numel}')

    optimizer = torch.optim.SGD(linear_model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print('[training]...')
    c = utils.Console(epochs)
    for e in range(epochs):
        loss = None
        for x, y in dataset:
            output = linear_model(x)
            loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        c.p(loss.item())
        dataset.start_over()

    return linear_model


def train(meta_model, dataset: dataset.Dataset, epochs: int = 10):
    """
    The training procedure. It follows the following steps:
    1. samples an example and its label from the training set
    2. passes it to the meta-network that outputs the weights
    3. copies the weights inside the predictor
    4. makes a prediction using the predictor
    5. computes the loss
    6. copies the gradient to the output of the meta-network
    7. backpropagates

    :return: both models trained
    """
    for e in range(epochs):
        for x, y in dataset:
            pass


if __name__ == '__main__':
    X, Y = load_dataset()
    data = dataset.Dataset(X, Y, _DEVICE)
    train_predictor_alone(data)
