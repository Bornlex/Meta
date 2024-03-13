import torch
from torch import nn

from src import utils
from src import layers


torch.manual_seed(42)
torch.autograd.set_detect_anomaly(True)


class Predictor(nn.Module):
    """
    A very simple model to make sure that the overall system works.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._output_size = output_size
        self._fc1 = nn.Linear(self._input_size, self._hidden_size)
        self._fc2 = nn.Linear(self._hidden_size, self._hidden_size // 2)
        self._fc3 = nn.Linear(self._hidden_size // 2, self._output_size)

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
        Make a prediction from an input.

        :param x: the input
        :return: the prediction
        """
        x = torch.relu(self._fc1(x))
        x = torch.relu(self._fc2(x))
        x = torch.log_softmax(self._fc3(x), dim=1)

        return x


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
        )
        self._model.to(utils.DEVICE)
        self._predictor_loss = nn.NLLLoss()

        meta_learning_input_size = self._input_size + self._output_size
        meta_learning_output_size = self._model.numel

        self._implicit = layers.Implicit(meta_learning_input_size, meta_learning_input_size)
        self._rnn = nn.RNN(meta_learning_input_size, self._hidden_size, batch_first=True)
        self._hidden_state = torch.zeros(1, self._hidden_size, device=utils.DEVICE)
        self._fc = nn.Linear(self._hidden_size, meta_learning_output_size)
        self._tanh = nn.Tanh()

        self._loss = nn.MSELoss()
        self._optimizer = torch.optim.SGD(
            [p for n, p in self.named_parameters() if '_model' not in n], lr=self._lr, momentum=self._momentum
        )
        self._losses = []
        self._gradients = []

    def flatten_parameters(self):
        p_shapes = []
        flat_parameters = []
        for n, p in self.named_parameters():
            if '_model' in n:
                continue
            p_shapes.append(p.size())
            flat_parameters.append(p.flatten())
        return torch.cat(flat_parameters)

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
            p.data = weights[:, already_allocated:already_allocated + p.numel()].data.reshape(p.shape)
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
        weights = self(x, y)
        if weights.shape[0] > 1:
            weights = torch.mean(weights, dim=0, keepdim=True)
        self.update_predictor_weights(weights)

        if not update:
            return None, None

        prediction = self._model(x)
        predictor_loss = self._predictor_loss(prediction, torch.argmax(y, -1))
        predictor_loss.backward(retain_graph=True)

        self._losses.append(predictor_loss.detach())
        gradients = self.get_gradients_from_predictor()

        if len(self._losses) <= 1:
            return None, predictor_loss

        self._optimizer.zero_grad()
        weights.backward(gradient=gradients.reshape(weights.shape))
        self._optimizer.step()

        return torch.sum(gradients), predictor_loss
