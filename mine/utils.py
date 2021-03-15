import copy
import warnings
import numpy as np

import torch
from torch.utils.tensorboard.writer import SummaryWriter


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


class Logger():

    def __init__(self, log_dir):
        self.log_dir = log_dir
        self._tb_logger = SummaryWriter(log_dir=self.log_dir)
        self._data = {}
        self._progbar = StatsDescriptor()

    def scalar(self, value, tag, step=None, accumulator=None, progbar=False):
        assert step is None or accumulator is None
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        if accumulator is None:
            self._tb_logger.add_scalar(tag, value, global_step=step)
            _ = self._progbar.add(tag, value) if progbar else None
        else:
            if not accumulator in self._data.keys():
                self._data[accumulator] = {}
            if tag in self._data[accumulator].keys():
                self._data[accumulator][tag].append(value)
            else:
                self._data[accumulator][tag] = [value]
            if progbar and not self._progbar.contains('_'.join([accumulator, tag])):
                self._progbar.add('_'.join([accumulator, tag]), None)

    def scalar_queue_flush(self, accumulator, step=None):
        assert accumulator in self._data.keys()
        out = {}
        if len(self._data[accumulator]) > 0:
            for key, values in self._data[accumulator].items():
                out[key] = np.mean(values)
                self._tb_logger.add_scalar('/'.join([accumulator, key]),
                                           out[key],
                                           global_step=step)
                if self._progbar.contains('_'.join([accumulator, key])):
                    self._progbar.add(key, out[key])

            _ = self._data.pop(accumulator)
        return out

    def scalar_queue_group_flush(self, accumulator, step=None):
        assert accumulator in self._data.keys()
        out = {}
        if len(self._data[accumulator]) > 0:
            for key, values in self._data[accumulator].items():
                out[key] = np.mean(values)
                if self._progbar.contains('_'.join([accumulator, key])):
                    self._progbar.add(key, out[key])
            self._tb_logger.add_scalars(accumulator, out, global_step=step)
            _ = self._data.pop(accumulator)
        return out
    
    def get_progbar_desc(self):
        return self._progbar.get_descriptor()


class StatsDescriptor:
    def __init__(self):
        self._stats = {}

    def get_descriptor(self):
        desc = []
        for key, val in self._stats.items():
            if val is not None:
                desc.append(key + '={:.4f}'.format(val))
        return '   '.join(desc)

    def add(self, key, value):
        self._stats[key] = value
    
    def contains(self, tag):
        return tag in self._stats
