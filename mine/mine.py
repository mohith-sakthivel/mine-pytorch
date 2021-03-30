import math

from mine.models import StatisticsNet

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Function


class UnbiasedLogMeanExp(Function):
    """
    Calculates and uses gradients with reduced bias
    """

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


class MINE_Base(nn.Module):

    def __init__(self, K, mine_lr=2e-4, variant='unbiased'):
        super().__init__()
        self._T = StatisticsNet(28*28, K)
        self._mine_lr = mine_lr
        self.variant = variant
        self._current_epoch = 0

    def _configure_optimizers(self):
        opt = optim.Adam(self._T.parameters(),
                         lr=self._mine_lr, betas=(0.5, 0.999))
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


def get_estimator(K, args_dict):
    estimator = args_dict.pop('estimator')
    if estimator == 'dv':
        return MINE_DV(K, **args_dict)
    elif estimator == 'fdiv':
        return MINE_f_Div(K, **args_dict)
    else:
        raise ValueError
