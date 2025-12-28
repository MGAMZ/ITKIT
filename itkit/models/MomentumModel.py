import copy
import logging

import torch
from mmengine.logging import print_log
from torch import Tensor


class MomentumAvgModel(torch.nn.Module):
    module: torch.nn.Module

    def __init__(self,
                 model: torch.nn.Module,
                 momentum: float = 0.0002,
                 gamma: int = 100,
                 interval: int = 1,
                 device: torch.device|None = None,
                 update_buffers: bool = False) -> None:
        super().__init__()

        # Check distributed environment
        self.is_distributed = hasattr(model, 'module')
        self.is_deepspeed = hasattr(model, 'module') and hasattr(model.module, 'deepspeed')

        # For DeepSpeed, get the full underlying model parameters
        if self.is_deepspeed:
            with model.module.summon_full_params(): # pyright: ignore
                self.module = copy.deepcopy(model.module).requires_grad_(False)
        else:
            target_model = model.module if self.is_distributed else model
            self.module = copy.deepcopy(target_model).requires_grad_(False)

        self.interval = interval
        if device is not None:
            self.module = self.module.to(device)

        self.register_buffer('steps', torch.tensor(0, dtype=torch.long, device=device))

        self.update_buffers = update_buffers
        if update_buffers:
            state_dict = self.module.state_dict()
            self.avg_parameters = {
                k: v for k, v in state_dict.items()
                if v.numel() > 0
            }
        else:
            params = dict(self.module.named_parameters())
            self.avg_parameters = {k: v for k, v in params.items()
                                   if v.numel() > 0}

        # Validate momentum parameter range
        assert 0.0 < momentum < 1.0, f'momentum must be in range (0.0, 1.0) but got {momentum}'
        if momentum > 0.5:
            print_log('The value of momentum in EMA is usually a small number,'
                      'which is different from the conventional notion of '
                      f'momentum but got {momentum}. Please make sure the '
                      f'value is correct.',
                      logger='current',
                      level=logging.WARNING)
        self.momentum = momentum
        assert gamma > 0, f'gamma must be greater than 0, but got {gamma}'
        self.gamma = gamma

    def forward(self, *args, **kwargs):
        """Forward method of the averaged model."""
        return self.module(*args, **kwargs)

    def _get_current_param(self):
        if self.update_buffers:
            return self.module.state_dict()
        else:
            return dict(self.module.named_parameters())

    def update_parameters(self, model: torch.nn.Module) -> None:
        """Update the parameters of the model. This method will execute the
        ``avg_func`` to compute the new parameters and update the model's
        parameters.

        Args:
            model (nn.Module): The model whose parameters will be averaged.
        """
        src_parameters = (
            model.state_dict()
            if self.update_buffers else dict(model.named_parameters()))
        if self.steps == 0:
            for k, p_avg in self.avg_parameters.items():
                p_avg.data.copy_(src_parameters[k].data)
        elif self.steps % self.interval == 0:  # type: ignore
            for k, p_avg in self.avg_parameters.items():
                # NOTE handle deepspeed model shred issue, p_avg may be empty here.
                if p_avg.dtype.is_floating_point and p_avg.shape==src_parameters[k].data.shape:
                    device = p_avg.device
                    self.avg_func(p_avg.data,
                                  src_parameters[k].data.to(device),
                                  self.steps)
        if not self.update_buffers:
            # If not update the buffers,
            # keep the buffers in sync with the source model.
            for b_avg, b_src in zip(self.module.buffers(), model.buffers()):
                b_avg.data.copy_(b_src.data.to(b_avg.device))
        self.steps += 1  # type: ignore

    def avg_func(self, averaged_param: Tensor, source_param: Tensor,
                 steps: int) -> None:
        """Compute the moving average of the parameters using the linear
        momentum strategy.

        Args:
            averaged_param (Tensor): The averaged parameters.
            source_param (Tensor): The source parameters.
            steps (int): The number of times the parameters have been
                updated.
        """
        momentum = max(self.momentum,
                       self.gamma / (self.gamma + self.steps.item()))
        averaged_param.lerp_(source_param, momentum)
