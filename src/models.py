import torch
from torch import nn

from src import utils


torch.manual_seed(42)
torch.autograd.set_detect_anomaly(True)


class Predictor(nn.Module):
    """
    A very simple model to make sure that the overall system works.
    y = softmax(ax + b)
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int, weights: torch.Tensor, bias: torch.Tensor):
        super().__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._output_size = output_size
        self._fc1 = nn.Linear(self._input_size, self._hidden_size)
        self._fc2 = nn.Linear(self._hidden_size, self._output_size)
        self._relu = nn.ReLU()
        self._soft = nn.Softmax(dim=1)

        #self._fc1.weight = nn.Parameter(weights.T)
        #self._fc1.bias = nn.Parameter(bias)

    @property
    def numel(self):
        return utils.get_number_parameters(self)

    @property
    def weights(self):
        return self._fc1.weight.data.clone()

    @property
    def bias(self):
        return self._fc1.bias.data.clone()

    @property
    def weights_grad(self):
        return self._fc1.weight.grad.clone()

    @property
    def bias_grad(self):
        return self._fc1.bias.grad.clone()

    def forward(self, x: torch.Tensor):
        """
        Make a linear prediction from an input.
        y = softmax(a * x + b)

        :param x: the input
        :return: the prediction
        """
        x = self._fc1(x)
        x = self._relu(x)
        x = self._fc2(x)
        return self._soft(x)


class MetaLoss(nn.Module):
    """
    This loss function is made only for propagating the gradient
    from the predictor to the meta-learner.
    """
    def __init__(self):
        super().__init__()

    def forward(
        self,
        current_predictor_loss: torch.Tensor,
        previous_predictor_loss: torch.Tensor,
        weights_update: torch.Tensor
    ):
        return torch.mean(
            (current_predictor_loss - previous_predictor_loss) * weights_update
        )


class PseudoLoss(nn.Module):
    """
    This loss function is made only for propagating the gradient
    from the predictor to the meta-learner.
    """
    def __init__(self):
        super().__init__()

    def forward(self, network_output: torch.Tensor):
        return network_output.sum()


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
            2 * self._input_size,
            self._output_size,
            torch.randn(self._input_size, self._output_size),
            torch.randn(self._output_size)
        )
        self._model.to(utils.DEVICE)
        self._predictor_loss = nn.MSELoss()

        meta_learning_input_size = self._input_size + self._output_size
        meta_learning_output_size = self._model.numel

        self._rnn = nn.RNN(meta_learning_input_size, self._hidden_size, batch_first=True)
        self._hidden_state = torch.zeros(1, self._hidden_size, device=utils.DEVICE)
        self._fc = nn.Linear(self._hidden_size, meta_learning_output_size)
        self._tanh = nn.Tanh()

        self._loss = MetaLoss()
        self._pseudo_loss = PseudoLoss()
        self._optimizer = torch.optim.SGD(
            [p for n, p in self.named_parameters() if '_model' not in n], lr=self._lr, momentum=self._momentum
        )
        self._losses = []
        self._gradients = []

    @staticmethod
    def _detach(var: torch.Tensor):
        v = torch.autograd.Variable(var.data, requires_grad=True)
        v.retain_grad()
        return v

    @property
    def predictor(self):
        return self._model

    def update_predictor_weights(self, weights: torch.Tensor):
        """
        Update the weights of the predictor.
        :param weights: the weights update that has been predicted
        :return: None
        """
        already_allocated = 0
        for n, p in self._model.named_parameters():
            p.data += weights[:, already_allocated:already_allocated + p.numel()].data.reshape(p.shape)
            already_allocated += p.numel()

    def get_gradients_from_predictor(self) -> torch.Tensor:
        """
        Get the gradients of the predictor's weights.
        :return:
        """
        return torch.cat([p.grad.flatten() for p in self._model.parameters()])

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        Predicts the variation of the predictor's weights.

        :param x: an example
        :param y: its label
        :return: the predicted weights
        """
        input_tensor = torch.cat((x, y), -1)
        out, self._hidden_state = self._rnn(input_tensor, self._hidden_state)
        self._hidden_state = self._detach(self._hidden_state)
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
        weights_update = self(x, y)
        self.update_predictor_weights(weights_update)

        if not update:
            return None, None

        prediction = self._model(x)
        predictor_loss = self._predictor_loss(prediction, y)
        predictor_loss.backward(retain_graph=True)

        self._losses.append(predictor_loss.detach())

        if len(self._losses) <= 1:
            return None, predictor_loss

        self._optimizer.zero_grad()
        pseudo_loss = self._pseudo_loss(weights_update)
        gradients = self.get_gradients_from_predictor()
        weights_update.grad = gradients.reshape(weights_update.shape)
        pseudo_loss.backward()
        self._optimizer.step()

        return pseudo_loss, predictor_loss


if __name__ == '__main__':
    meta = Meta(2, 3, 2)
    print(meta(torch.randn(1, 1, 2)))
