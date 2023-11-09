import torch
from torch import nn
from sklearn import datasets

from src import utils
from src import models
from src import dataset


def load_dataset():
    iris = datasets.load_iris()
    return iris['data'], iris['target']


def train_predictor_alone(dataset: dataset.Dataset, epochs: int = 10, lr: float = .1):
    linear_model = models.Predictor(
        dataset.input_size,
        dataset.output_size,
        torch.randn(dataset.input_size, dataset.output_size),
        torch.randn(dataset.output_size)
    )
    linear_model.to(utils.DEVICE)

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

    print('[testing]...')
    dataset.test()
    i = 0
    accuracy = 0.0  # mean between [0, 1]
    for x, y in dataset:
        output = linear_model(x)
        good = int(torch.argmax(output, -1) == torch.argmax(y, -1))
        if i == 0:
            accuracy = good
            i += 1
            continue
        accuracy = (i / (i + 1)) * accuracy + good / (i + 1)
        i += 1
    print(f'[accuracy]: {accuracy * 100:.1f}%')

    return linear_model


def train(dataset: dataset.Dataset, epochs: int = 10, lr: float = .1):
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
    meta = models.Meta()


if __name__ == '__main__':
    X, Y = load_dataset()
    data = dataset.Dataset(X, Y, utils.DEVICE)
    print(data.input_size, data.output_size)
