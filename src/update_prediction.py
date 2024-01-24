import torch
from torch import nn

from src import ode
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


class Meta(ode.ODEF):
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

        self._loss = nn.MSELoss()
        self._pseudo_loss = PseudoLoss()
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
        gradients = self.get_gradients_from_predictor()

        if len(self._losses) <= 1:
            return None, predictor_loss

        self._optimizer.zero_grad()
        loss = self._loss(weights_update, torch.reshape(gradients, weights_update.shape))
        loss.backward()
        self._optimizer.step()

        return loss, predictor_loss


if __name__ == '__main__':
    import torch
    from torch import nn
    import numpy as np
    from sklearn import datasets

    from src import utils
    from src import update_prediction
    from src import dataset
    from src import visualisation


    def load_dataset():
        iris = datasets.load_iris()
        return iris['data'], iris['target']


    def train_predictor_alone(
            ds: dataset.Dataset,
            epochs: int = 10,
            lr: float = .1,
            momentum: float = .9,
            store: visualisation.TrainingStore = None,
            batch_size: int = 8
    ):
        linear_model = models.Predictor(
            ds.input_size,
            16,
            ds.output_size,
            torch.randn(ds.input_size, ds.output_size),
            torch.randn(ds.output_size)
        )
        linear_model.to(utils.DEVICE)

        numel = utils.get_number_parameters(linear_model)
        print(f'[parameters]: {numel}')

        optimizer = torch.optim.SGD(linear_model.parameters(), lr=lr, momentum=momentum)
        criterion = nn.MSELoss()

        print('[training]...')
        c = utils.Console(epochs)
        for e in range(epochs):
            loss = None
            xs = []
            ys = []
            for x, y in ds:
                xs.append(x)
                ys.append(y)

                if len(xs) == batch_size:
                    optimizer.zero_grad()
                    xs = torch.reshape(torch.stack(xs), (batch_size, ds.input_size))
                    ys = torch.reshape(torch.stack(ys), (batch_size, ds.output_size))
                    output = linear_model(xs)
                    loss = criterion(output, ys)
                    loss.backward()
                    optimizer.step()
                    store.add(
                        loss.item(),
                        linear_model.weights.tolist(),
                        linear_model.bias.tolist(),
                        linear_model.weights_grad.tolist(),
                        linear_model.bias_grad.tolist()
                    )

                    xs = []
                    ys = []

            c.p(loss.item())
            ds.start_over()

        print('[testing]...')
        ds.test()
        i = 0
        accuracy = 0.0  # mean between [0, 1]
        for x, y in ds:
            output = linear_model(x)
            good = int(torch.argmax(output, -1) == torch.argmax(y, -1))
            if i == 0:
                accuracy = good
                i += 1
                continue
            accuracy = (i / (i + 1)) * accuracy + good / (i + 1)
            i += 1
        print(f'[accuracy]: {accuracy * 100:.1f}%')

        print('[meta]...')
        ds.meta()
        meta_accuracy = 0.0
        first_shot = True
        for x, y in ds:
            output = linear_model(x)
            if first_shot:
                loss = criterion(output, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                first_shot = False
                continue

            good = int(torch.argmax(output, -1) == torch.argmax(y, -1))
            if i == 0:
                meta_accuracy = good
                i += 1
                continue
            meta_accuracy = (i / (i + 1)) * meta_accuracy + good / (i + 1)
            i += 1
        print(f'[meta accuracy]: {meta_accuracy * 100:.1f}%')
        metrics_store.plot()

        return linear_model


    def train(
            ds: dataset.Dataset,
            epochs: int = 10,
            lr: float = .1,
            momentum: float = .9,
            store: visualisation.TrainingStore = None
    ):
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
        meta = models.Meta(
            input_size=ds.input_size,
            hidden_size=16,
            output_size=ds.output_size,
            lr=lr,
            momentum=momentum
        )
        meta.to(utils.DEVICE)

        print('[training]...')
        c = utils.Console(epochs)
        for e in range(epochs):
            meta_loss, predictor_loss = None, None
            for x, y in ds:
                predictor_losses = []
                meta_losses = []
                for _ in range(1):
                    meta_loss, predictor_loss = meta.step(x, y)
                    if meta_loss is not None:
                        meta_losses.append(meta_loss.item())
                    predictor_losses.append(predictor_loss.item())
                store.add_meta(
                    np.mean(predictor_losses),
                    np.mean(meta_losses),
                    meta.predictor.weights.tolist(),
                    meta.predictor.bias.tolist(),
                )

            c.p(torch.sum(predictor_loss).item())
            ds.start_over()

        print('[testing]...')
        ds.test()
        i = 0
        accuracy = 0.0  # mean between [0, 1]
        for x, y in ds:
            output = meta.predictor(x)
            good = int(torch.argmax(output, -1) == torch.argmax(y, -1))
            if i == 0:
                accuracy = good
                i += 1
                continue
            accuracy = (i / (i + 1)) * accuracy + good / (i + 1)
            i += 1
        print(f'[accuracy]: {accuracy * 100:.1f}%')

        print('[meta]...')
        ds.meta()
        meta_accuracy = 0.0
        first_shot = True
        for x, y in ds:
            if first_shot:
                meta.step(x, y, update=False)
                first_shot = False
                continue

            output = meta.predictor(x)
            good = int(torch.argmax(output, -1) == torch.argmax(y, -1))
            if i == 0:
                meta_accuracy = good
                i += 1
                continue
            meta_accuracy = (i / (i + 1)) * meta_accuracy + good / (i + 1)
            i += 1
        print(f'[meta accuracy]: {meta_accuracy * 100:.1f}%')
        metrics_store.plot_meta()

        return meta


    X, Y = load_dataset()
    data = dataset.Dataset(X, Y, utils.DEVICE)
    metrics_store = visualisation.TrainingStore()
    train(data, lr=.05, momentum=.9, store=metrics_store)
    # train_predictor_alone(data, epochs=20, lr=.01, momentum=.8, store=metrics_store, batch_size=16)

