import torch
from torch import nn

from src.utils import DEVICE


class Meta(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._output_size = output_size

        self._rnn = nn.RNN(self._input_size, self._hidden_size, None, batch_first=True)
        self._hidden_state = torch.randn(1, self._input_size, self._hidden_size).to(DEVICE)
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
        self._soft = nn.Softmax(dim=1)

        self._fc.weight = nn.Parameter(weights.T)
        self._fc.bias = nn.Parameter(bias)

    def forward(self, x: torch.Tensor):
        """
        Make a linear prediction from an input.
        y = a * x + b

        :param x: the input
        :return: the prediction
        """
        x = self._fc(x)
        return self._soft(x)
