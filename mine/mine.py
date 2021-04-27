import math

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Function

from mine.models import StatisticsNet


class UnbiasedLogMeanExp(Function):
    """
    Calculates and uses gradients with reduced bias
    """

    epsilon = 1e-6

    @staticmethod
    def forward(ctx, i, ema):
        ctx.save_for_backward(i, ema)
        mean_numel = torch.tensor(i.shape[0], dtype=torch.float)
        output = i.logsumexp(dim=0).subtract(torch.log(mean_numel))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        i, ema = ctx.saved_tensors
        grad_i = grad_ema = None
        mean_numel = torch.tensor(i.shape[0], dtype=torch.float)
        grad_i = grad_output * \
            (i.exp() / ((ema+UnbiasedLogMeanExp.epsilon)*mean_numel))
        return grad_i, grad_ema


class MINE_Base(nn.Module):

    def __init__(self, input_dim, K, est_lr=2e-4, variant='unbiased'):
        super().__init__()
        self._T = StatisticsNet(input_dim, K)
        self._est_lr = est_lr
        self.variant = variant
        self._current_epoch = 0

    def _configure_optimizers(self):
        opt = optim.Adam(self._T.parameters(),
                         lr=self._est_lr, betas=(0.5, 0.999))
        sch = None
        return opt, sch

    def step_epoch(self):
        self._current_epoch += 1


class MINE_DV(MINE_Base):
    _ANNEAL_PERIOD = 0
    _EMA_ANNEAL_PERIOD = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._decay = 0.994  # decay for ema (not tuned)
        self._ema = None

    def _update_ema(self, t_margin):
        with torch.no_grad():
            exp_t = t_margin.exp().mean(dim=0)
            if self._ema is not None:
                self._ema = self._decay * self._ema + (1-self._decay) * exp_t
            else:
                self._ema = exp_t

    def get_mi_bound(self, x, z, z_margin=None, update_ema=False):
        t_joint = self._T(x, z).mean(dim=0)
        if z_margin is not None:
            t_margin = self._T(x, z_margin)
        else:
            t_margin = self._T(x, z[torch.randperm(x.shape[0])])
        # maintain an exponential moving average of exp_t under the marginal distribution
        # done to reduce bias in the estimator
        if ((self.variant == 'unbiased' and update_ema) and
                self._current_epoch > self._EMA_ANNEAL_PERIOD):
            self._update_ema(t_margin)
        # Calculate biased or unbiased estimate
        if self.variant == 'unbiased' and self._current_epoch > self._ANNEAL_PERIOD:
            log_exp_t = UnbiasedLogMeanExp.apply(t_margin, self._ema)
        else:
            log_exp_t = t_margin.logsumexp(
                dim=0).subtract(math.log(x.shape[0]))
        # mi lower bound
        return t_joint - log_exp_t


class MINE_f_Div(MINE_Base):

    def get_mi_bound(self, x, z, z_margin=None, update_ema=False):
        t_joint = self._T(x, z).mean(dim=0)
        if z_margin is not None:
            t_margin = self._T(x, z_margin)
        else:
            t_margin = self._T(x, z[torch.randperm(x.shape[0])])

        exp_t = torch.mean(torch.exp(t_margin-1), dim=0)
        # mi lower bound
        return t_joint - exp_t


class NWJ(nn.Module):
    """
    NWJ (Nguyen, Wainwright, and Jordan) estimator
    """

    def __init__(self, input_dim, K, est_lr=2e-4, variant='unbiased'):
        super().__init__()
        self._critic = StatisticsNet(input_dim, K)
        # from mine.models import ConcatCritic, BiLinearCritic
        # self._critic = ConcatCritic(input_dim, K)
        # self._critic = BiLinearCritic(input_dim, K)
        self._est_lr = est_lr
        self._current_epoch = 0
        self.variant = variant

    def _configure_optimizers(self):
        opt = optim.Adam(self._critic.parameters(),
                         lr=self._est_lr, betas=(0.5, 0.999))
        sch = None
        return opt, sch
    
    def step_epoch(self):
        self._current_epoch += 1

    @staticmethod
    def logmeanexp_nondiag(tensor):
        batch_size = tensor.shape[0]
        device = tensor.device
        dim = (0, 1)
        numel = batch_size * (batch_size-1)
        logsumexp = torch.logsumexp(tensor - torch.diag(np.inf * torch.ones(batch_size, device=device)), dim=dim)
        return logsumexp - np.math.log(numel)

    def get_mi_bound(self,  x, z, z_margin=None, update_ema=None):
        joint = self._critic(x, z).mean(dim=0)
        if z_margin is not None:
            margin = self._critic(x, z_margin)
        else:
            margin = self._critic(x, z[torch.randperm(x.shape[0])])
            margin = torch.logsumexp(margin, dim=0) - np.math.log(z.shape[0])
            margin = torch.exp(margin-1)
        return joint - margin


def get_estimator(input_dim, K, args_dict):
    args_dict = args_dict.copy()
    estimator = args_dict.pop('estimator')
    if estimator == 'dv':
        return MINE_DV(input_dim, K, **args_dict)
    elif estimator == 'fdiv':
        return MINE_f_Div(input_dim, K, **args_dict)
    elif estimator == 'nwj':
        return NWJ(input_dim, K, **args_dict)
    else:
        raise ValueError
