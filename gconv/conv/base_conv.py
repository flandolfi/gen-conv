import math
from abc import abstractmethod
from typing import Optional, Union, Sequence, Any
from itertools import product

import torch
from torch import Tensor
from torch.nn import Conv1d, Conv2d, Conv3d, Parameter

from torch_geometric.nn import inits
from torch_geometric.nn.conv import MessagePassing


class BaseGenConv(MessagePassing):
    def __init__(self, in_channels: int,
                 out_channels: Optional[int] = None,
                 pos_channels: Optional[int] = None,
                 bias: bool = True,
                 offsets: Union[int, Sequence[float], Tensor] = 8,
                 learn_offsets: bool = True,
                 temperature: Union[float, str] = 1.,
                 learn_temperature: bool = True,
                 groups: int = 1,
                 offset_initializer: str = 'uniform',
                 weight_initializer: str = 'kaiming_uniform',
                 bias_initializer: str = 'uniform',
                 *args, **kwargs):
        super(BaseGenConv, self).__init__(*args, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.pos_channels = pos_channels or in_channels

        assert self.in_channels % groups == 0, \
            "`in_channels` must be divisible by `groups`"
        assert self.out_channels % groups == 0, \
            "`out_channels` must be divisible by `groups`"

        self.groups = groups

        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.temperature_initializer = float(temperature)

        if isinstance(offsets, int):
            self.num_offsets = offsets
            offsets = Tensor(self.num_offsets, self.pos_channels)
            self.offset_initializer = offset_initializer
        else:
            if isinstance(offsets, (tuple, list)):
                offsets = Tensor(list(product(*[offsets] * self.pos_channels)))

            self.num_offsets = offsets.size(0)
            self.offset_initializer = offsets.clone()

        learn_temperature &= not math.isinf(self.temperature_initializer)
        self.temperature = Parameter(Tensor(1), requires_grad=learn_temperature)

        learn_offsets &= not math.isinf(self.temperature_initializer)
        self.offsets = Parameter(offsets, requires_grad=learn_offsets)

        param_per_offset = self.out_channels * self.in_channels // self.groups
        self.weights = Parameter(Tensor(self.num_offsets, param_per_offset))

        if bias:
            self.bias = Parameter(Tensor(1, self.out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameter(self, param: Parameter, initializer: Union[str, Tensor]):
        if torch.is_tensor(initializer):
            with torch.no_grad():
                param.set_(initializer.clone())
        elif initializer == 'zeros':
            inits.zeros(param)
        elif initializer == 'ones':
            inits.ones(param)
        elif initializer == 'glorot':
            inits.glorot(param)
        elif initializer == 'uniform':
            inits.uniform(self.in_channels, param)
        elif initializer == 'kaiming_uniform':
            inits.kaiming_uniform(param, self.in_channels, a=math.sqrt(5))
        elif initializer == 'orthogonal':
            torch.nn.init.orthogonal_(param)

    def reset_parameters(self):
        self.reset_parameter(self.weights, self.weight_initializer)
        self.reset_parameter(self.bias, self.bias_initializer)
        self.reset_parameter(self.offsets, self.offset_initializer)
        torch.nn.init.constant_(self.temperature, self.temperature_initializer)

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        pass

    @classmethod
    def from_regular_conv(cls, conv: Union[Conv1d, Conv2d, Conv3d], **kwargs):
        ndim = conv.weight.dim() - 2
        offsets = list(product(*[range(k) for k in conv.kernel_size]))
        offsets = torch.Tensor(offsets)
        index = offsets.T.long()

        weights = conv.weight.permute(tuple(range(2, 2 + ndim)) + (0, 1))
        weights = weights[tuple(index)].flatten(start_dim=1)

        if conv.padding == 'same':
            offsets = offsets - offsets.max(0)[0] // 2
        elif isinstance(conv.padding, tuple):
            padding = torch.Tensor([conv.padding])
            offsets = offsets - padding

        out = cls(in_channels=conv.in_channels,
                  out_channels=conv.out_channels,
                  pos_channels=ndim,
                  bias=conv.bias is not None,
                  offsets=offsets,
                  groups=conv.groups,
                  **kwargs)

        with torch.no_grad():
            out.weights.set_(weights)

            if conv.bias is not None:
                out.bias.set_(conv.bias.unsqueeze(0))

        return out
