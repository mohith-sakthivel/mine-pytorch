import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, output_dim, stochastic=False,
                 init_std_bias=-5.0, min_std=1e-8):
        super().__init__()
        self._stochastic = stochastic
        if stochastic:
            self._init_std_bias = init_std_bias
            self._min_std = min_std
            self.latent_out = 2 * output_dim
        else:
            self.latent_out = output_dim
        self._net = None

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


class MLP(FeatureExtractor):
    """
    MLP with gaussian or deterministic outputs
    """

    def __init__(self, input_dim, output_dim, layers, act=nn.ReLU, **kwargs):
        super().__init__(input_dim, output_dim, **kwargs)

        net_layers = nn.ModuleList()
        inp = input_dim
        for layer_dim in layers:
            net_layers.append(nn.Linear(inp, layer_dim))
            net_layers.append(act())
            inp = layer_dim

        net_layers.append(nn.Linear(inp, self.latent_out))
        self._net = nn.Sequential(*net_layers)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = int(v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d,
                           nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGGNet(FeatureExtractor):

    cfgs = {
        'vgg_9': [32, 'M', 64, 'M', 128, 128, 'M', 256, 256, 'M'],
        'vgg_11': [32, 'M', 64, 'M', 128, 128, 'M', 256, 256, 'M', 256, 256, 'M']
    }

    def __init__(self, input_dim, output_dim, bn=False, arch='vgg_11', **kwargs):
        super().__init__(input_dim, output_dim, **kwargs)

        self._arch = arch
        self._base_layers = make_layers(self.cfgs[arch], bn)

        with torch.no_grad():
            dum_input = torch.zeros((1,) + input_dim, dtype=torch.float32)
            flat_shape = self._base_layers(dum_input).shape.numel()

        self._net = nn.Sequential(
            self._base_layers,
            nn.Flatten(start_dim=1),
            nn.Linear(flat_shape, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.latent_out)
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


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
        self._initialize_weights()

    def _initialize_weights(self):
        for (name, param) in self._layers.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param, nonlinearity='relu')
            elif 'bias' in name:
                nn.init.zeros_(param)

        for (name, param) in self._out_layer.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param, nonlinearity='linear')
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x, z):
        x = torch.cat([x, z], dim=1)
        x = x + 0.3 * torch.randn_like(x)
        for hid_layer in self._layers:
            x = F.elu(hid_layer(x))
            x = x + 0.5 * torch.randn_like(x)
        return self._out_layer(x)
