import copy
import warnings
import torch


class PolyakAveraging():
    """
    Calculates exponential moving average of parameter weights during training
    """

    def __init__(self, alpha=0.999):
        self._alpha = alpha
        self._avg_module = None

    def on_train_start(self, module):
        if self._avg_module is None:
            self._avg_module = copy.deepcopy(module)
        else:
            warnings.warn(
                "Existing EMA(Exponential Moving Average) values of the model is being used for new 'Trainer.fit' sequence", RuntimeWarning)

    def on_train_batch_end(self, module):
        device = self._avg_module.device
        for src, dst in zip(module.get_model_parameters(), self._avg_module.get_model_parameters()):
            dst.detach().copy_(self._alpha * dst + (1-self._alpha) * src.detach().to(device))

    def on_train_end(self, module):
        torch.save(module.state_dict, module.logdir.joinpath('train_end.pth'))
        device = module.device
        for src, dst in zip(self._avg_module.get_model_parameters(), module.get_model_parameters()):
            dst.detach().copy_(src.detach().to(device))
        torch.save(module.state_dict, module.logdir.joinpath('ema_weights.pth'))
        self._avg_module = None


class BetaScheduler:
    """
        Schedules beta after an intial warmup period. Value is annealed from 
        the intial value to the steady state value linearly.
    """
    def __init__(self, warm_up, value, anneal_period, init_value=0):
        self._warm_up = warm_up
        self._value = value
        self._anneal_period = anneal_period
        self._init_value = init_value

    def get(self, epoch):
        if epoch <= self._warm_up:
            return self._init_value
        elif epoch > (self._warm_up + self._anneal_period):
            return self._value
        else:
            ratio = (epoch-self._warm_up)/(self._anneal_period)
            return ratio * (self._value - self._init_value) + self._init_value