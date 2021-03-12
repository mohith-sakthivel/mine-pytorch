import math
import argparse
import itertools

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import torch.nn.functional as F

import torchvision.transforms as transforms
import torchvision.datasets as datasets

import pytorch_lightning as pl

from mine.models import MLP, StatisticsNet
from mine.utils import PolyakAveraging


class Classifer(pl.LightningModule):
    def __init__(self, base_net, K, lr=1e-4, base_net_args={}, use_polyak=True):
        """
        Multi-label classifier with cross entropy loss
        """
        super().__init__()
        self.save_hyperparameters()
        self._base_net = base_net(
            input_dim=28*28, output_dim=K, **base_net_args)
        self._logits = nn.Linear(K, 10)
        self._polyak_list = [self._base_net, self._logits]
        self._initialize_weights()

    def _initialize_weights(self):
        for (name, param) in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
            else:
                raise ValueError

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        items.pop("loss", None)
        return items

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(), lr=self.hparams.lr, betas=(0.5, 0.999))
        scheduler = {
            'scheduler': optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97),
            'interval': 'epoch',
            'frequency': 2
        }
        return [optimizer], [scheduler]

    def polyak_parameters(self):
        return itertools.chain(*[module.parameters() for module in self._polyak_list])

    def configure_callbacks(self):
        callbacks = []
        if self.hparams.use_polyak:
            callbacks.append(PolyakAveraging())
        return callbacks

    def _get_embedding(self, x):
        x = self._base_net(x)
        if self._base_net.is_stochastic():
            mean, std = x
            x = dist.Independent(dist.Normal(mean, std), 1).rsample()
        return x

    def forward(self, x):
        x = self._get_embedding(x)
        return self._logits(x)

    def get_eval_stats(self, batch, batch_idx):
        stats = {}
        x, y = batch
        x = x.view(x.shape[0], -1)
        y_pred = self(x)
        y_pred = torch.argmax(y_pred, dim=1)
        stats['error'] = torch.sum(y != y_pred)/len(y)*100
        return stats

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.shape[0], -1)
        z = self._get_embedding(x)
        logits = self._logits(z)
        loss = F.cross_entropy(logits, y)
        self.log('cross_entorpy', loss, on_epoch=True,
                 on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        stats = self.get_eval_stats(batch, batch_idx)
        self.log('test_error_during_train', stats['error'])

    def test_step(self, batch, batch_idx):
        stats = self.get_eval_stats(batch, batch_idx)
        self.log('test_error', stats['error'])
        return {'test_error', stats['error']}


class MINE_Classifier(Classifer):
    def __init__(self, base_net, K, beta=1e-3, **kwargs):
        super().__init__(base_net, K, **kwargs)
        self._T = StatisticsNet(28*28, K)
        self.automatic_optimization = False
        self._decay = 0.5
        self.save_hyperparameters('beta')

    def configure_optimizers(self):
        optimizers, schedulers = super().configure_optimizers()
        mine_opt = [optim.Adam(self._T.parameters(),
                               lr=self.hparams.lr, betas=(0.5, 0.999))]
        return optimizers + mine_opt, schedulers

    def _get_train_embedding(self, x):
        x = self._base_net(x)
        x_dist = None
        if self._base_net.is_stochastic():
            mean, std = x
            x_dist = dist.Independent(dist.Normal(mean, std), 1)
            x = x_dist.rsample()
        return x, x_dist

    def _reduce_bias(self, biased_value, t_margin):
        exp_t = torch.exp(t_margin).mean(dim=0)
        if hasattr(self, '_exp_t_ma'):
            self._exp_t_ma = \
                self._decay * self._exp_t_ma + (1-self._decay) * exp_t
        else:
            self._exp_t_ma = exp_t
        correction = exp_t/self._exp_t_ma
        return correction.detach() * biased_value

    def _get_mi_bound(self, T, x, z):
        t_joint = T(x, z).mean(dim=0)
        z_margin = z[torch.randperm(x.shape[0])]
        t_margin = T(x, z_margin)
        log_exp_t = torch.logsumexp(t_margin, dim=0) - math.log(x.shape[0])
        # log_exp_t = self._reduce_bias(log_exp_t, t_margin)
        mi = t_joint - log_exp_t
        return mi

    def model_train_step(self, x, y, opt):
        """ Train classifier """
        opt.zero_grad()
        # calculate loss
        z, z_dist = self._get_train_embedding(x)
        self._cache = {'z': z.detach()}  # cache z for MINE loss calculation
        mi_xz = self._get_mi_bound(self._T, x, z)
        logits = self._logits(z)
        cross_entropy = F.cross_entropy(logits, y)
        loss = cross_entropy + self.hparams.beta * mi_xz
        # log train stats
        if z_dist is not None:
            self.log('z_post_entropy', z_dist.entropy().mean(),
                     on_epoch=True, on_step=False, prog_bar=True)
        self.log('cross_entorpy', cross_entropy,
                 on_epoch=True, on_step=False, prog_bar=True)
        self.log('mi_xz', mi_xz, on_epoch=True, on_step=False, prog_bar=True)
        self.log('total_loss', loss, on_epoch=True,
                 on_step=False, prog_bar=True)
        # step optimizer
        self.manual_backward(loss)
        opt.step()

    def mine_train_step(self, x, z, opt):
        opt.zero_grad()
        # calculate loss
        loss = -self._get_mi_bound(self._T, x, z.detach())
        # log stats
        self.log('stat_net_loss', loss, on_epoch=True,
                 on_step=False, prog_bar=True)
        # step optimizer
        self.manual_backward(loss)
        opt.step()

    def training_step(self, batch, batch_idx):
        model_opt, mine_opt = self.optimizers()
        x, y = batch
        x = x.view(x.shape[0], -1)

        self.model_train_step(x, y, model_opt)
        self.mine_train_step(x, self._cache['z'], mine_opt)


def run(experiment, args):
    pl.seed_everything(args['seed'])
    if experiment == 'deter':
        Model = Classifer
    elif experiment == 'mine':
        Model = MINE_Classifier

    data_transforms = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(0.5, 0.5)])

    train_dataset = datasets.MNIST(
        './data', train=True, download=True, transform=data_transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args['batch_size'],
                                               shuffle=True,
                                               num_workers=args['workers'])

    test_dataset = datasets.MNIST(
        './data', train=False, download=True, transform=data_transforms)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args['batch_size'],
                                              shuffle=False,
                                              num_workers=args['workers'])

    model = Model(MLP, **args['model_args'])

    trainer = pl.Trainer(gpus=args['gpus'],
                         max_epochs=args['epochs'],
                         default_root_dir=args['log_dir']+'/'+experiment,
                         check_val_every_n_epoch=5,
                         num_sanity_val_steps=0)

    _ = trainer.fit(model, train_dataloader=train_loader)
    _ = trainer.test(test_dataloaders=test_loader)


def get_default_args(experiment):
    """
    Returns default experiment arguments
    """

    args = {
        'seed': 0,
        # Trainer args
        'gpus': 1,
        'epochs': 200,
        'log_dir': './logs',
        # Dataset args
        'batch_size': 100,
        'workers': 4,
        # Model args
        'model_args': {
            'lr': 1e-4,
            'use_polyak': False,
        }
    }

    if experiment == 'deter':
        args['model_args']['K'] = 1024
        args['model_args']['base_net_args'] = {
            'layers': [784, 1024], 'stochastic': False}

    elif experiment == 'mine':
        args['model_args']['K'] = 256
        args['model_args']['beta'] = 1e-2
        args['model_args']['base_net_args'] = {
            'layers': [784, 1024, 1024], 'stochastic': True}

    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Information BottleNeck with MINE')
    parser.add_argument('--seed', action='store', type=int)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--deter', action='store_const',
                       const=True, default=False)
    group.add_argument('--mine', action='store_const',
                       const=True, default=False)
    parser.add_argument('--beta', action='store', type=float,
                        help='information bottleneck ')
    args = parser.parse_args()

    if args.deter:
        experiment = 'deter'
    elif args.mine:
        experiment = 'mine'

    exp_args = get_default_args(experiment)
    for key, value in args.__dict__.items():
        exp_args[key] = value

    if args.beta is not None:
        exp_args['model_args']['beta'] = args.beta

    run(experiment, exp_args)
