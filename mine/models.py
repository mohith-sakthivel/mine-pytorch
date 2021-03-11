import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    MLP with gaussian or deterministic outputs
    """

    def __init__(self, input_dim, output_dim, layers, stochastic=False,
                 act=lambda: nn.ReLU(inplace=True), init_std_bias=0.0, min_std=1e-5):
        super().__init__()
        self._stochastic = stochastic
        self._init_std_bias = init_std_bias
        self._min_std = min_std

        net_layers = nn.ModuleList()
        inp = input_dim
        for layer_dim in layers:
            net_layers.append(nn.Linear(inp, layer_dim))
            net_layers.append(act())
            inp = layer_dim
        outp = 2*output_dim if self._stochastic else output_dim
        net_layers.append(nn.Linear(inp, outp))
        self._net = nn.Sequential(*net_layers)

    def is_stochastic(self):
        return self._stochastic

    def forward(self, x):
        x = self._net(x)
        if self._stochastic:
            # parameterize outputs as a sample from a gaussian
            mean, std = torch.chunk(x, chunks=2, dim=1)
            std = nn.functional.softplus(
                std+self._init_std_bias) + self._min_std
            return mean, std
        return x


class StatisticsNet(nn.Module):
    """
    Network for estimating mutual information between two random variables
    """

    def __init__(self, x_dim, z_dim):
        super().__init__()
        self._layers = nn.ModuleList()
        self._layers.append(nn.Linear(x_dim+z_dim, 512))
        self._layers.append(nn.Linear(512, 512))
        self._out_layer = nn.Linear(512, 1)

    def forward(self, x, z):
        x = torch.cat([x, z], dim=1)
        x = x + 0.3 * torch.randn_like(x)
        for hid_layer in self._layers:
            x = F.elu(hid_layer(x), inplace=True)
            x = x + 0.5 * torch.randn_like(x)
        return self._out_layer(x)