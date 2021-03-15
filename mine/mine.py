import math
import time
import tqdm
import pathlib
import argparse
import itertools

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import torch.nn.functional as F
from torch.autograd import Function

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from mine.models import MLP, StatisticsNet
from mine.utils import PolyakAveraging, Logger


class UnbiasedLogMeanExp(Function):

    epsilon = 1e-6

    @staticmethod
    def forward(ctx, i, ema):
        ctx.save_for_backward(i, ema)
        mean_numel = torch.tensor(i.shape[0])
        output = i.logsumexp(dim=0).subtract(torch.log(mean_numel))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        i, ema = ctx.saved_tensors
        grad_i = grad_ema = None
        mean_numel = torch.tensor(i.shape[0])
        grad_i = grad_output * \
            (i.exp() / ((ema+UnbiasedLogMeanExp.epsilon)*mean_numel))
        return grad_i, grad_ema

# class UnbiasedWrapper(Function):

#     epsilon = 1e-6

#     @staticmethod
#     def forward(ctx, i, ema):
#         ctx.save_for_backward(i, ema)
#         return i

#     @staticmethod
#     def backward(ctx, grad_output):
#         i, ema = ctx.saved_tensors
#         grad_i = grad_ema = None
#         ratio = i.exp().mean(dim=0) / (ema)
#         grad_i = grad_output * ratio.clip(0.5, 2)
#         return grad_i, grad_ema


class Classifer(nn.Module):
    def __init__(self, base_net, K, lr=1e-4, base_net_args={}, use_polyak=True, logdir='.'):
        """
        Multi-label classifier with cross entropy loss
        """
        super().__init__()
        self._K = K
        self._lr = lr
        self._use_polyak = use_polyak
        self.logdir = logdir
        self._device = 'cpu'

        self._base_net = base_net(
            input_dim=28*28, output_dim=K, **base_net_args)
        self._logits = nn.Linear(K, 10)
        self._polyak_list = [self._base_net, self._logits]
        self._initialize_weights()
        self._configure_callbacks()

    def _initialize_weights(self):
        for (name, param) in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
            else:
                raise ValueError

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(), lr=self._lr, betas=(0.5, 0.999))
        scheduler = {
            'scheduler': optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97),
            'frequency': 2
        }
        self.optimizers, self.schedulers = ([optimizer], [scheduler])

    @property
    def device(self):
        return self._logits.weight.device

    def _configure_callbacks(self):
        self._callbacks = []
        if self._use_polyak:
            self._callbacks.append(PolyakAveraging())

    def invoke_callback(self, hook):
        for callback in self._callbacks:
            if hasattr(callback, hook):
                func = getattr(callback, hook)
                func(self)

    def polyak_parameters(self):
        return itertools.chain(*[module.parameters() for module in self._polyak_list])

    def _unpack_batch(self, batch):
        batch = [item.to(self.device) for item in batch]
        return batch

    def _get_embedding(self, x):
        x = self._base_net(x)
        if self._base_net.is_stochastic():
            mean, std = x
            x = dist.Independent(dist.Normal(mean, std), 1).rsample()
        return x

    def forward(self, x):
        x = self._get_embedding(x.to(self.device))
        return self._logits(x)

    def _get_eval_stats(self, batch, batch_idx):
        stats = {}
        x, y = self._unpack_batch(batch)
        x = x.view(x.shape[0], -1)
        y_pred = self(x)
        y_pred = torch.argmax(y_pred, dim=1)
        stats['error'] = torch.sum(y != y_pred)/len(y)*100
        return stats

    def training_step(self, batch, batch_idx, logger):
        opt = self.optimizers[0]
        opt.zero_grad()
        x, y = self._unpack_batch(batch)
        x = x.view(x.shape[0], -1)
        z = self._get_embedding(x)
        logits = self._logits(z)
        loss = F.cross_entropy(logits, y)
        stats = {'loss': loss.detach().cpu().numpy()}
        logger.scalar(stats['loss'], 'cross_ent',
                      accumulator='train', progbar=True)
        loss.backward()
        opt.step()
        return stats

    def validation_step(self, batch, batch_idx, logger):
        stats = self._get_eval_stats(batch, batch_idx)
        logger.scalar(stats['error'], 'error', accumulator='validation')

    def test_step(self, batch, batch_idx, logger):
        stats = self._get_eval_stats(batch, batch_idx)
        logger.scalar(stats['error'], 'error', accumulator='test')


class MINE_Classifier(Classifer):
    def __init__(self, base_net, K, beta=1e-3, **kwargs):
        super().__init__(base_net, K, **kwargs)
        self._T = StatisticsNet(28*28, K)
        self._decay = 0.99
        self._beta = beta
        self._unbiased = True
        self._exp_t_ma = {}

    def configure_optimizers(self):
        super().configure_optimizers()
        mine_opt = optim.Adam(self._T.parameters(),
                              lr=self._lr, betas=(0.5, 0.999))

        self.optimizers.append(mine_opt)

    def _get_train_embedding(self, x):
        x = self._base_net(x)
        x_dist = None
        if self._base_net.is_stochastic():
            mean, std = x
            x_dist = dist.Independent(dist.Normal(mean, std), 1)
            x = x_dist.rsample()
        return x, x_dist

    def _update_ema(self, t_margin, called_from):
        with torch.no_grad():
            exp_t = t_margin.exp().mean(dim=0)
            if called_from in self._exp_t_ma:
                self._exp_t_ma[called_from] = self._decay * self._exp_t_ma[called_from] + (1-self._decay) * exp_t
            else:
                self._exp_t_ma[called_from] = exp_t
            logger.scalar(exp_t, 'exp_t_'+called_from, step=ee+ss/600, accumulator=None)
            logger.scalar(self._exp_t_ma[called_from], 'exp_t_ema_'+called_from, step=ee + ss/600, accumulator=None)
            logger.scalar(exp_t/self._exp_t_ma[called_from], 'ratio_'+called_from, step=ee + ss/600, accumulator=None)

    def _get_mi_bound(self, T, x, z, called_from):
        t_joint = T(x, z).mean(dim=0)
        z_margin = z[torch.randperm(x.shape[0])]
        t_margin = T(x, z_margin)
        if self._unbiased and ee>5:
            _ = self._update_ema(t_margin, called_from)
            if ee > 10:
                # t_margin = UnbiasedWrapper.apply(t_margin, self._exp_t_ma[called_from])
                log_exp_t = UnbiasedLogMeanExp.apply(t_margin, self._exp_t_ma[called_from])
            else:
                log_exp_t = t_margin.logsumexp(dim=0).subtract(math.log(x.shape[0]))
        else:
            log_exp_t = t_margin.logsumexp(dim=0).subtract(math.log(x.shape[0]))
        mi = t_joint - log_exp_t
        return mi, log_exp_t

    def model_train_step(self, x, y, opt, logger):
        """ Train classifier """
        opt.zero_grad()
        # calculate loss
        z, z_dist = self._get_train_embedding(x)

        # ************************************************************************
        logger.scalar(z_dist.mean, 'z_mean', accumulator='train')
        logger.scalar(z_dist.stddev, 'z_std', accumulator='train')
        # ************************************************************************
        
        self._cache = {'z': z.detach()}  # cache z for MINE loss calculation
        mi_xz, log_exp_t = self._get_mi_bound(self._T, x, z, 'model')

        # ************************************************************************
        logger.scalar(log_exp_t, 'log_exp_t', accumulator='train')
        # ************************************************************************

        logits = self._logits(z)
        cross_entropy = F.cross_entropy(logits, y)
        loss = cross_entropy + self._beta * mi_xz
        # log train stats
        if z_dist is not None:
            logger.scalar(z_dist.entropy().mean(), 'z_post_ent',
                          accumulator='train', progbar=True)
        logger.scalar(cross_entropy, 'cross_ent',
                      accumulator='train', progbar=True)
        logger.scalar(mi_xz, 'mi_xz', accumulator='train', progbar=True)
        logger.scalar(loss, 'total_loss',  accumulator='train', progbar=False)
        # step optimizer
        loss.backward()
        opt.step()

    def mine_train_step(self, x, z, opt, logger):
        opt.zero_grad()
        # calculate loss
        loss, log_exp_t = self._get_mi_bound(self._T, x, z.detach(), 'mine')
        loss = -loss
        # log stats
        logger.scalar(loss, 'mi_est_loss',
                      accumulator='train', progbar=True)
        # step optimizer
        loss.backward()
        opt.step()

    def training_step(self, batch, batch_idx, logger):
        model_opt, mine_opt = self.optimizers
        x, y = self._unpack_batch(batch)
        x = x.view(x.shape[0], -1)

        self.model_train_step(x, y, model_opt, logger)
        self.mine_train_step(x, self._cache['z'], mine_opt, logger)


def run(model_id, args):
    if args['seed'] is not None:
        torch.manual_seed(args['seed'])
    if model_id == 'deter':
        Model = Classifer
    elif model_id == 'mine':
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

    logdir = pathlib.Path(args['logdir'])
    time_stamp = time.strftime("%d-%m-%Y_%H:%M:%S")
    logdir = logdir.joinpath(model_id, '_'.join(
        [args['exp_name'], 's{}'.format(args['seed']), time_stamp]))
    global logger
    logger = Logger(log_dir=logdir)
    args['model_args']['logdir'] = logdir

    model = Model(MLP, **args['model_args'])
    model.to(args['device'])
    model.configure_optimizers()

    global ee, ss

    # Training loop
    model.invoke_callback('on_train_start')
    for epoch in tqdm.trange(1, args['epochs']+1, disable=True):
        ee = epoch

        model.train(True)
        for batch_idx, batch in enumerate(tqdm.tqdm(train_loader,
                                                    desc='{}/{} Epochs'.format(
                                                        epoch, args['epochs']),
                                                    unit=' batches',
                                                    postfix=logger.get_progbar_desc(),
                                                    leave=False)):
            ss = batch_idx
            batch_stats = model.training_step(batch, batch_idx, logger)
            model.invoke_callback('on_train_batch_end')
        train_stats = logger.scalar_queue_flush('train', epoch)
        # _ = logger.scalar_queue_flush('mi_debug', epoch)

        for sch in model.schedulers:
            if epoch % sch['frequency'] == 0:
                sch['scheduler'].step()

        # Run validation step
        if (args['validation_freq'] is not None and
                epoch % args['validation_freq'] == 0):
            model.eval()
            # testset used in validation step for observation/study purpose
            for batch_idx, batch in enumerate(test_loader):
                model.validation_step(batch, batch_idx, logger)
            _ = logger.scalar_queue_flush('validation', epoch)

    model.invoke_callback('on_train_end')

    # Test model
    model.eval()
    for batch_idx, batch in enumerate(test_loader):
        model.test_step(batch, batch_idx, logger)
    test_out = logger.scalar_queue_flush('test', epoch)

    print('Model Test Error: {:.4f}%'.format(test_out['error']))


def get_default_args(model_id):
    """
    Returns default experiment arguments
    """

    args = {
        'exp_name': 'mine_unbiased_beta_0.99_debug',
        'seed': 0,
        # Trainer args
        'device': 'cuda',
        'epochs': 200,
        'logdir': './logs',
        'validation_freq': 5,
        # Dataset args
        'batch_size': 100,
        'workers': 4,
        # Model args
        'model_args': {
            'lr': 1e-4,
            'use_polyak': True,
        }
    }

    if model_id == 'deter':
        args['model_args']['K'] = 1024
        args['model_args']['base_net_args'] = {
            'layers': [784, 1024], 'stochastic': False}

    elif model_id == 'mine':
        args['model_args']['K'] = 256
        args['model_args']['beta'] = 1e-3
        args['model_args']['base_net_args'] = {
            'layers': [784, 1024, 1024], 'stochastic': True}

    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Information BottleNeck with MINE')

    parser.add_argument('--exp_name', action='store', type=str,
                        help='Experiment Name')
    parser.add_argument('--seed', action='store', type=int)
    parser.add_argument('--logdir', action='store', type=str,
                        help='Directory to log results')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--deter', action='store_const',
                       const=True, default=False)
    group.add_argument('--mine', action='store_const',
                       const=True, default=False)

    parser.add_argument('--beta', action='store', type=float,
                        help='information bottleneck')
    args = parser.parse_args()

    if args.deter:
        model_id = 'deter'
    elif args.mine:
        model_id = 'mine'

    exp_args = get_default_args(model_id)
    for key, value in args.__dict__.items():
        if value is not None:
            exp_args[key] = value

    if args.beta is not None:
        exp_args['model_args']['beta'] = args.beta

    run(model_id, exp_args)
