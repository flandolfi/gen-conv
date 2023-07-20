import math
from typing import Optional

import torch
from torch import Tensor
from torch.nn import Parameter, Module

from torch_geometric.nn import inits
from torch_geometric.utils import scatter
from torch_geometric.typing import OptTensor, Adj


class GenConv(Module):
    def __init__(self, in_channels: int,
                 out_channels: Optional[int] = None,
                 pos_channels: Optional[int] = None,
                 bias: bool = True,
                 num_offsets: int = 16,
                 trainable_offsets: bool = True,
                 metric: str = 'euclidean',
                 depthwise: bool = False,
                 offset_initializer: str = 'uniform',
                 weight_initializer: str = 'kaiming_uniform',
                 bias_initializer: str = 'uniform'):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.pos_channels = pos_channels or in_channels

        self.num_offsets = num_offsets
        self.metric = metric
        self.depthwise = depthwise

        self.offset_initializer = offset_initializer
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer

        self.offset = Parameter(Tensor(self.num_offsets, self.pos_channels),
                                requires_grad=trainable_offsets)

        if self.depthwise:
            self.weight = Parameter(Tensor(self.num_offsets, self.in_channels))
        else:
            self.weight = Parameter(Tensor(self.num_offsets, self.out_channels, self.in_channels))

        if bias:
            self.bias = Parameter(Tensor(self.num_offsets, self.out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameter(self, param: Tensor, initializer: str):
        if initializer == 'zeros':
            inits.zeros(param)
        elif initializer == 'glorot':
            inits.glorot(param)
        elif initializer == 'uniform':
            inits.uniform(self.in_channels, param)
        elif initializer == 'kaiming_uniform':
            inits.kaiming_uniform(param, self.in_channels, a=math.sqrt(5))
        elif initializer == 'orthogonal':
            torch.nn.init.orthogonal_(param)

    def reset_parameters(self):
        self.reset_parameter(self.weight, self.weight_initializer)
        self.reset_parameter(self.bias, self.bias_initializer)
        self.reset_parameter(self.offset, self.offset_initializer)

    def pairwise_similarity(self, offset: Tensor) -> Tensor:
        if self.metric == 'euclidean':
            return -torch.cdist(offset, self.offset)

        return offset @ self.offset.T

    def forward(self, x: Tensor, edge_index: Adj,
                edge_attr: OptTensor = None, pos: OptTensor = None) -> Tensor:
        if pos is None:
            pos = x

        if torch.is_tensor(edge_index):
            row, col = edge_index
            val = edge_attr
        else:
            row, col, val = edge_index.coo()

        if val is None:
            val = torch.ones_like(row, dtype=torch.float)

        sim = self.pairwise_similarity(pos[col] - pos[row])
        alpha = torch.softmax(sim, dim=-1)

        W = torch.einsum('ij,j...->i...', alpha, self.weight)

        if self.depthwise:
            msg = W*x[col]
        else:
            msg = torch.bmm(W, x[col].unsqueeze(-1)).squeeze(-1)

        if self.bias is not None:
            b = alpha @ self.bias
            msg = msg + b

        return scatter(msg*val.view(-1, 1), row, dim=0, dim_size=x.size(0))
