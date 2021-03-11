import copy
import warnings

import torch
from pytorch_lightning.callbacks import Callback


class PolyakAveraging(Callback):

    def __init__(self, alpha=0.999):
        self._alpha = alpha
        self._avg_module = None

    def on_train_start(self, trainer, pl_module):
        if self._avg_module is None:
            self._avg_module = copy.deepcopy(pl_module)
        else:
            warnings.warn(
                "Existing EMA(Exponential Moving Average) values of the model is being used for new 'Trainer.fit' sequence", RuntimeWarning)

    def on_train_batch_end(self, trainer, pl_module, *args):
        device = self._avg_module.device
        for src, dst in zip(pl_module.polyak_parameters(), self._avg_module.polyak_parameters()):
            dst.detach().copy_(self._alpha * dst + (1-self._alpha) * src.detach().to(device))

    def on_train_end(self, trainer, pl_module):
        torch.save(pl_module.state_dict, trainer.log_dir +
                   '/epoch_{}.pth'.format(pl_module.current_epoch))
        device = pl_module .device
        for src, dst in zip(self._avg_module.polyak_parameters(), pl_module.polyak_parameters()):
            dst.detach().copy_(src.detach().to(device))
        torch.save(pl_module.state_dict, trainer.log_dir+'/ema_weights.pth')
        self._avg_module = None
