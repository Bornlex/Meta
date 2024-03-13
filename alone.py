import torch
from torch import nn
from matplotlib import pyplot as plt
from torchvision import datasets, transforms


if torch.backends.mps.is_available():
    print("MPS is available!")
    device = torch.device("mps")
else:
    print("MPS not available, defaulting to CPU")
    device = torch.device("cpu")


LOG = 1


def flatten_parameters(m):
    p_shapes = []
    flat_parameters = []
    for n, p in m.named_parameters():
        p_shapes.append(p.size())
        flat_parameters.append(p.flatten())
    return torch.cat(flat_parameters)


def update_weights(m, weights):
    already_allocated = 0
    for n, p in m.named_parameters():
        p.data = weights[already_allocated:already_allocated + p.numel()].data.reshape(p.shape)
        already_allocated += p.numel()


def fixed_point(model, images, labels, criterion, optimizer, epsilon=1e-4):
    delta = 1e8
    losses = []
    deltas = []
    current_parameters = None
    previous_parameters = flatten_parameters(model)
    counter = 0
    if LOG:
        print(f'finding the fixed point...')
    while delta > epsilon:
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        counter += 1
        l = loss.item()
        losses.append(l)
        current_parameters = flatten_parameters(model)
        # delta = torch.norm(current_parameters - previous_parameters).item()
        delta = losses[-1] - losses[-2] if len(losses) > 1 else 1e8
        deltas.append(delta)
        previous_parameters = current_parameters
        if counter % 100 == 0 and LOG:
            print(f'[{counter}] loss: {l}, delta: {delta}')

    if LOG:
        print(f'fixed point found in {counter} iterations')
    return current_parameters, losses, deltas


def train(model, data, criterion, optimizer, epochs: int = 10):
    print(f'training...')
    losses = []
    for e in range(epochs):
        loss = None
        for x, y in data:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f'epoch {e + 1}/{epochs}, loss: {loss.item()}')
    return losses


def train_fp(model, data, criterion, optimizer, epochs: int = 10):
    print(f'training...')
    global LOG
    LOG = 0
    losses = []
    counter = 1
    for e in range(epochs):
        loss = None
        for x, y in data:
            x, y = x.to(device), y.to(device)
            parameters, _, _ = fixed_point(model, x, y, criterion, optimizer)
            interpolated_parameters = (counter - 1) / counter * flatten_parameters(model) + parameters / counter
            update_weights(model, interpolated_parameters)
            output = model(x)
            loss = criterion(output, y)
            losses.append(loss.item())
        print(f'epoch {e + 1}/{epochs}, loss: {loss.item()}')
    return losses


transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize the tensors
])

trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader_fp = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)
trainloader_regular = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True)

testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=True)


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.shape[0], -1)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.log_softmax(self.fc3(x), dim=1)

        return x


network_fp = Network().to(device)
network = Network().to(device)
cri = nn.NLLLoss()
opt = torch.optim.SGD(network.parameters(), lr=0.003)
eps = 1e-3

"""
First example.
"""
images, labels = next(iter(trainloader_fp))
images = images.view(images.shape[0], -1)
# fp1, _, _ = fixed_point(network_fp, images, labels, cri, opt, eps)

losses_fp = train_fp(network_fp, trainloader_regular, cri, opt, 10)
losses = train(network, trainloader_regular, cri, opt, 10)


fig, ax = plt.subplots(2)
ax[0].plot(losses_fp, label='fixed point')
ax[1].plot(losses, label='regular')
plt.show()
