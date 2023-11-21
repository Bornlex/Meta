import torch
from torch import nn

from src import utils


class Predictor(nn.Module):
    """
    A very simple model to make sure that the overall system works.
    y = softmax(ax + b)
    """
    def __init__(self, input_size, output_size, weights: torch.Tensor, bias: torch.Tensor):
        super().__init__()
        self._input_size = input_size
        self._output_size = output_size
        self._fc = nn.Linear(self._input_size, self._output_size)
        self._soft = nn.Softmax(dim=1)

        self._fc.weight = nn.Parameter(weights.T)
        self._fc.bias = nn.Parameter(bias)

    @property
    def numel(self):
        return utils.get_number_parameters(self)

    @property
    def weights_grad(self):
        return self._fc.weight.grad.clone()

    @property
    def bias_grad(self):
        return self._fc.bias.grad.clone()

    def forward(self, x: torch.Tensor):
        """
        Make a linear prediction from an input.
        y = softmax(a * x + b)

        :param x: the input
        :return: the prediction
        """
        x = self._fc(x)
        return self._soft(x)


class MetaLoss(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, predicted: torch.Tensor, gradients: torch.Tensor):
        return predicted


class Meta(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        A meta-learner that learns to learn.

        Considering a simple predictor model: y = ax + b.
        :param input_size: the size of the x
        :param hidden_size: the size of the hidden state
        :param output_size: the size of y
        """
        super().__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._output_size = output_size

        self._model = Predictor(
            self._input_size,
            self._output_size,
            torch.randn(self._input_size, self._output_size),
            torch.randn(self._output_size)
        )
        self._model.to(utils.DEVICE)
        self._predictor_loss = nn.MSELoss()

        meta_learning_input_size = self._input_size + self._output_size
        meta_learning_output_size = self._model.numel

        self._rnn = nn.RNN(meta_learning_input_size, self._hidden_size, batch_first=True)
        self._hidden_state = None
        self._fc = nn.Linear(self._hidden_size, meta_learning_output_size)
        self._tanh = nn.Tanh()

        self._optimizer = torch.optim.SGD(self.parameters(), lr=.1)

    @property
    def predictor(self):
        return self._model

    def copy_weights_to_predictor(self, weights: torch.Tensor):
        """
        Copies the weights to the predictor.
        :param weights: the weights that have been predicted
        :return: None
        """
        already_allocated = 0
        for p in self._model.parameters():
            p.data = weights[:, already_allocated:already_allocated + p.numel()].data.reshape(p.shape)
            already_allocated += p.numel()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        Predicts the weights of the predictor.

        :param x: an example
        :param y: its label
        :return: the predicted weights
        """
        input_tensor = torch.cat((x, y), -1)
        out, self._hidden_state = self._rnn(input_tensor, self._hidden_state)
        out = self._fc(out)
        return self._tanh(out)

    def step(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Updates the knowledge of the meta-learner after seeing one more example.

        :param x: an example
        :param y: its label
        :return: None
        """
        prediction = self._model(x)
        predictor_loss = self._predictor_loss(prediction, y)
        predictor_loss.backward()

        concatenated_gradients = torch.cat((
            self._model.weights_grad.reshape((1, self._model.weights_grad.numel())),
            self._model.bias_grad.reshape((1, self._model.bias_grad.numel()))
        ), -1)

        self._fc.weight.grad = self._model.weights_grad
        self._fc.bias.grad = self._model.bias_grad
        self._optimizer.step()

        return predictor_loss


if __name__ == '__main__':
    meta = Meta(2, 3, 2)
    print(meta(torch.randn(1, 1, 2)))
