import torch
from torch import nn


class Implicit(nn.Module):
    def __init__(self, input_features, output_features, tol=1e-4, max_iter=50):
        super().__init__()
        self.linear = nn.Linear(input_features, output_features, bias=False)
        self.tol = tol
        self.max_iter = max_iter

    def forward(self, x):
        with (torch.no_grad()):
            z = torch.tanh(x)
            iterations = 0
            while iterations < self.max_iter:
                z_linear = self.linear(z) + x
                g = z - torch.tanh(z_linear)
                err = torch.norm(g)
                if err < self.tol:
                    break

                j = torch.eye(z.shape[1])[None, :, :] - (1 / torch.cosh(z_linear) ** 2)[:, :, None] * self.linear.weight[None, :, :]
                z = z - torch.solve(g[:, :, None], j)[0][:, :, 0]
                self.iterations += 1

        z = torch.tanh(self.linear(z) + x)
        z.register_hook(lambda grad: torch.solve(grad[:, :, None], j.transpose(1, 2))[0][:, :, 0])
        return z
