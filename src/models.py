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
    def weights(self):
        return self._fc.weight.data.clone()

    @property
    def bias(self):
        return self._fc.bias.data.clone()

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
    """
    This loss function is made only for propagating the gradient
    from the predictor to the meta-learner.
    """
    def __init__(self):
        super().__init__()

    def forward(self, predicted: torch.Tensor, gradients: torch.Tensor):
        """
        Custom loss function in order to connect the loss of the predictor and
        the weights predicted by the meta-learner.
        We want that the derivative of this function is the gradient of the loss of the
        predictor with respect to the output of the meta-learner.

        So if we want a function whose derivative is 'a', the simplest is:
        f(x) = a * x

        :param predicted: the weights as predicted by the meta-learner
        :param gradients: the gradients of the predictor with respect to its parameters
        :return:
        """
        return predicted * gradients


class Meta(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, lr: float = .1, momentum: float = .9):
        """
        A meta-learner that learns to learn.

        Considering a simple predictor model: y = ax + b.
        :param input_size: the size of the x
        :param hidden_size: the size of the hidden state
        :param output_size: the size of y
        :param lr: the learning rate
        """
        super().__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._output_size = output_size
        self._lr = lr
        self._momentum = momentum

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

        self._loss = MetaLoss()
        self._optimizer = torch.optim.SGD(
            [p for n, p in self.named_parameters() if '_model' not in n], lr=self._lr, momentum=self._momentum
        )

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

    def step(self, x: torch.Tensor, y: torch.Tensor, update: bool = True) -> [torch.Tensor, torch.Tensor]:
        """
        Updates the knowledge of the meta-learner after seeing one more example.

        :param x: an example
        :param y: its label
        :param update: whether to update the meta-learner or not
        :return: both the meta-loss and the predictor-loss
        """
        predictor_weights = self(x, y)
        self.copy_weights_to_predictor(predictor_weights)

        if not update:
            return None, None

        prediction = self._model(x)
        predictor_loss = self._predictor_loss(prediction, y)
        predictor_loss.backward()

        concatenated_gradients = torch.cat((
            self._model.weights_grad.reshape((1, self._model.weights_grad.numel())),
            self._model.bias_grad.reshape((1, self._model.bias_grad.numel()))
        ), -1)

        meta_loss = self._loss(predictor_weights, concatenated_gradients)
        self._optimizer.step()

        return meta_loss, predictor_loss


if __name__ == '__main__':
    meta = Meta(2, 3, 2)
    print(meta(torch.randn(1, 1, 2)))
