import numpy.random
import torch
from torch import nn
import numpy as np

from src import utils
from src import meta
from src import dataset
from src import visualisation


numpy.random.seed(42)


def train_predictor_alone(
        ds: dataset.MNIST,
        epochs: int = 10,
        iterations: int = 100,
        lr: float = .1,
        momentum: float = .9,
        store: visualisation.TrainingStore = None,
):
    linear_model = meta.Predictor(
        ds.input_size,
        16,
        ds.output_size,
    )
    linear_model.to(utils.DEVICE)

    numel = utils.get_number_parameters(linear_model)
    print(f'[parameters]: {numel}')

    optimizer = torch.optim.SGD(linear_model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.NLLLoss()

    print('[training]...')
    c = utils.Console(epochs)
    for e in range(epochs):
        loss = None
        for _ in range(iterations):
            xs, ys = ds.train()
            optimizer.zero_grad()
            output = linear_model(xs)
            loss = criterion(output, torch.argmax(ys, -1))
            loss.backward()
            optimizer.step()
            store.add(
                loss.item(),
            )

        c.p(loss.item())

    print('[testing]...')
    i = 0
    accuracy = 0.0
    for _ in range(iterations):
        xs, ys = ds.test()
        output = linear_model(xs)
        good = torch.mean((torch.argmax(output, -1) == torch.argmax(ys, -1)).float())
        if i == 0:
            accuracy = good
            i += 1
            continue
        accuracy = (i / (i + 1)) * accuracy + good / (i + 1)
        i += 1
    print(f'[accuracy]: {accuracy * 100:.1f}%')

    # metrics_store.plot()

    return linear_model


def train(
        ds: dataset.MNIST,
        epochs: int = 10,
        iterations: int = 100,
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
    ml = meta.Meta(
        input_size=ds.input_size,
        hidden_size=16,
        output_size=ds.output_size,
        lr=lr,
        momentum=momentum
    )
    ml.to(utils.DEVICE)

    print('[training]...')
    c = utils.Console(epochs)
    for e in range(epochs):
        meta_loss, predictor_loss = None, None
        for _ in range(iterations):
            xs, ys = ds.train()
            predictor_losses = []
            meta_losses = []
            for _ in range(1):
                meta_loss, predictor_loss = ml.step(xs, ys)
                if meta_loss is not None:
                    meta_losses.append(meta_loss.item())
                predictor_losses.append(predictor_loss.item())
            store.add_meta(
                np.mean(predictor_losses),
                np.mean(meta_losses),
            )

        c.p(torch.sum(predictor_loss).item())

    print('[testing]...')
    i = 0
    accuracy = 0.0
    for _ in range(iterations):
        xs, ys = ds.test()
        output = ml.predictor(xs)
        good = torch.mean((torch.argmax(output, -1) == torch.argmax(ys, -1)).float())
        if i == 0:
            accuracy = good
            i += 1
            continue
        accuracy = (i / (i + 1)) * accuracy + good / (i + 1)
        i += 1
    print(f'[accuracy]: {accuracy * 100:.1f}%')

    # metrics_store.plot_meta()

    return ml


if __name__ == '__main__':
    epochs = 10
    batch_size = 8
    lr = .01
    momentum = .8
    iterations = 50

    data = dataset.MNIST(batch_size, utils.DEVICE)
    metrics_store = visualisation.TrainingStore()

    meta_model = train(data, epochs=epochs, iterations=iterations, lr=lr, momentum=momentum, store=metrics_store)
    linear_model = train_predictor_alone(data, epochs=epochs, iterations=iterations, lr=lr, momentum=momentum, store=metrics_store)

    wa = visualisation.WeightsAnalyser()
    wa.add((meta_model.flatten_predictor_parameters(), 'meta'))
    wa.add((utils.get_parameters(linear_model), 'regular'))
    wa.plot()
