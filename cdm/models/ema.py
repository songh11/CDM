# Copied from another repo, but I can't remember exactly which one.

from collections.abc import Iterable

import torch


def _get_data(param):
    """Get the underlying tensor data, handling DTensor (FSDP2) transparently.

    FSDP2 wraps parameters as DTensor. Direct .data.copy_() between DTensor and
    regular Tensor raises 'mixed Tensor and DTensor' errors. This helper
    returns the local shard tensor so that all copy/arithmetic operations work
    on plain Tensors.
    """
    if hasattr(param, '_local_tensor'):
        return param._local_tensor
    return param.data


class EMAModuleWrapper:
    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        decay: float = 0.9999,
        update_step_interval: int = 1,
        device: torch.device | None = None,
    ):
        parameters = list(parameters)
        self.ema_parameters = [_get_data(p).clone().detach().to(device) for p in parameters]

        self.temp_stored_parameters = None

        self.decay = decay
        self.update_step_interval = update_step_interval
        self.device = device

    def get_current_decay(self, optimization_step) -> float:
        return min((1 + optimization_step) / (10 + optimization_step), self.decay)

    @torch.no_grad()
    def step(self, parameters: Iterable[torch.nn.Parameter], optimization_step):
        parameters = list(parameters)

        one_minus_decay = 1 - self.get_current_decay(optimization_step)

        if (optimization_step + 1) % self.update_step_interval == 0:
            for ema_parameter, parameter in zip(self.ema_parameters, parameters, strict=True):
                if parameter.requires_grad:
                    param_data = _get_data(parameter)
                    if ema_parameter.device == param_data.device:
                        ema_parameter.add_(one_minus_decay * (param_data - ema_parameter))
                    else:
                        parameter_copy = param_data.detach().to(ema_parameter.device)
                        parameter_copy.sub_(ema_parameter)
                        parameter_copy.mul_(one_minus_decay)
                        ema_parameter.add_(parameter_copy)
                        del parameter_copy

    def to(self, device: torch.device = None, dtype: torch.dtype = None) -> None:
        self.device = device
        self.ema_parameters = [
            p.to(device=device, dtype=dtype) if p.is_floating_point() else p.to(device=device)
            for p in self.ema_parameters
        ]

    @torch.no_grad()
    def sync_with_model(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """
        Force the EMA parameters to be a direct copy of the given model parameters.
        This is used to create a snapshot for the rollout policy.
        """
        parameters = list(parameters)
        for ema_parameter, parameter in zip(self.ema_parameters, parameters, strict=True):
            ema_parameter.data.copy_(_get_data(parameter).detach())

    def copy_ema_to(self, parameters: Iterable[torch.nn.Parameter], store_temp: bool = True, grad=False) -> None:
        if store_temp:
            if grad:
                self.temp_stored_parameters = [_get_data(parameter).clone() for parameter in parameters]
            else:
                self.temp_stored_parameters = [_get_data(parameter).detach().clone() for parameter in parameters]

        parameters = list(parameters)
        for ema_parameter, parameter in zip(self.ema_parameters, parameters, strict=True):
            _get_data(parameter).copy_(ema_parameter.to(_get_data(parameter).device).data)

    def copy_temp_to(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        for temp_parameter, parameter in zip(self.temp_stored_parameters, parameters, strict=True):
            _get_data(parameter).copy_(temp_parameter.to(_get_data(parameter).device))

        self.temp_stored_parameters = None

    def load_state_dict(self, state_dict: dict) -> None:
        self.decay = self.decay if self.decay else state_dict.get("decay", self.decay)
        self.ema_parameters = state_dict.get("ema_parameters")
        self.to(self.device)

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "ema_parameters": self.ema_parameters,
        }