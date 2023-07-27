import math
from typing import Optional, Union

import torch
from torch import Tensor
from torch.nn import Parameter, Module, functional as F

from torch_geometric.nn import inits, knn_graph
from torch_geometric.utils import scatter
from torch_geometric.typing import OptTensor, Adj


class GenConv(Module):
    def __init__(self, in_channels: int,
                 out_channels: Optional[int] = None,
                 pos_channels: Optional[int] = None,
                 bias: bool = True,
                 num_offsets: int = 8,
                 trainable_offsets: bool = True,
                 metric: str = 'euclidean',
                 temperature: Union[float, str] = 'same',
                 groups: int = 1,
                 aggr: str = 'add',
                 offset_initializer: str = 'uniform',
                 weight_initializer: str = 'kaiming_uniform',
                 bias_initializer: str = 'uniform'):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.pos_channels = pos_channels or in_channels

        self.num_offsets = num_offsets
        self.metric = metric
        self.groups = groups
        self.aggr = aggr
        self.temperature = temperature

        if temperature == 'same':
            self.temperature = self.num_offsets

        self.offset_initializer = offset_initializer
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer

        self.offset = Parameter(Tensor(self.num_offsets, self.pos_channels),
                                requires_grad=trainable_offsets)

        self.weight = Parameter(Tensor(self.num_offsets, self.out_channels*self.in_channels//self.groups))

        if bias:
            self.bias = Parameter(Tensor(1, self.out_channels))
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
        
        if self.metric == 'cosine':
            obs = F.normalize(offset, p=2, dim=-1)
            exp = F.normalize(self.offset, p=2, dim=-1)

            return obs @ exp.T
        
        if self.metric == 'delta-norm':
            obs = torch.norm(offset, p=2, dim=-1, keepdim=True)
            exp = torch.norm(self.offset, p=2, dim=-1, keepdim=True)

            return -torch.abs(obs - exp.T)

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
        alpha = torch.softmax(sim*self.temperature, dim=-1)

        W = alpha @ self.weight

        W = W.view(-1, self.groups, self.out_channels//self.groups, self.in_channels//self.groups)
        x = x.view(-1, self.groups, self.in_channels//self.groups, 1)

        msg = W @ x[col]
        msg = msg.view(-1, self.out_channels) * val.view(-1, 1)

        out = scatter(msg, row, dim=0, dim_size=x.size(0), reduce=self.aggr)

        if self.bias is not None:
            out = out + self.bias

        return out


class DynamicGenConv(GenConv):
    def __init__(self, in_channels: int,
                 out_channels: Optional[int] = None,
                 pos_channels: Optional[int] = None, 
                 k: int = 20,
                 loop: bool = True,
                 *args, **kwargs):
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         pos_channels=pos_channels,
                         *args, **kwargs)
        self.k = k
        self.loop = loop

    def forward(self, x: Tensor, pos: OptTensor = None, batch: OptTensor = None) -> Tensor:
        if pos is None:
            pos = x
        
        edge_index = knn_graph(pos, k=self.k, batch=batch, loop=self.loop)
        return super().forward(x, edge_index, None, pos)
