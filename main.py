import torch
from torch import nn
import numpy as np
from sklearn import datasets

from src import utils
from src import models
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


if __name__ == '__main__':
    X, Y = load_dataset()
    data = dataset.Dataset(X, Y, utils.DEVICE)
    metrics_store = visualisation.TrainingStore()
    train(data, lr=.05, momentum=.9, store=metrics_store)
    # train_predictor_alone(data, epochs=20, lr=.01, momentum=.8, store=metrics_store, batch_size=16)
