import math
from typing import Optional, Union, Sequence, Callable
from itertools import product

import torch
from torch import Tensor
from torch.nn import Parameter, Module, functional as F

from torch_geometric.nn import inits, knn_graph, MessagePassing
from torch_geometric.utils import scatter
from torch_geometric.typing import OptTensor, Adj

Similarity = Callable[[Tensor, Tensor], Tensor]


class GenConv(MessagePassing):
    def __init__(self, in_channels: int,
                 out_channels: Optional[int] = None,
                 pos_channels: Optional[int] = None,
                 bias: bool = True,
                 offsets: Union[int, Sequence[float], Tensor] = 8,
                 learn_offsets: bool = True,
                 similarity: Union[str, Similarity] = 'cosine',
                 temperature: Union[float, str] = 'learn',
                 groups: int = 1,
                 offset_initializer: str = 'uniform',
                 weight_initializer: str = 'kaiming_uniform',
                 bias_initializer: str = 'uniform',
                 *args, **kwargs):
        super(GenConv, self).__init__(*args, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.pos_channels = pos_channels or in_channels

        assert self.in_channels % groups == 0, \
            "`in_channels` must be divisible by `groups`"
        assert self.out_channels % groups == 0, \
            "`out_channels` must be divisible by `groups`"

        self.similarity = similarity
        self.groups = groups

        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer

        if isinstance(offsets, int):
            self.num_offsets = offsets
            offsets = Tensor(self.num_offsets, self.pos_channels)
            self.offset_initializer = offset_initializer
        else:
            if isinstance(offsets, (tuple, list)):
                offsets = Tensor(list(product(*[offsets]*self.pos_channels)))
            
            self.num_offsets = offsets.size(0)
            self.offset_initializer = offsets.clone()

        if temperature == 'learn':
            self.temperature = Parameter(Tensor([1.]))
        elif temperature == 'same':
            self.temperature = self.num_offsets
        else:            
            self.temperature = float(temperature)

        learn_offsets &= not math.isinf(self.temperature)
        self.offsets = Parameter(offsets, requires_grad=learn_offsets)
        
        param_per_offset = self.out_channels*self.in_channels//self.groups
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

        if torch.is_tensor(self.temperature):
            self.reset_parameter(self.temperature, 'ones')

    def pairwise_similarity(self, offsets: Tensor) -> Tensor:
        if self.similarity == 'dot':
            return offsets @ self.offsets.T
        
        if self.similarity == 'cosine':
            obs = F.normalize(offsets, p=2, dim=-1)
            exp = F.normalize(self.offsets, p=2, dim=-1)

            return obs @ exp.T
        
        if self.similarity == 'delta-norm':
            obs = torch.norm(offsets, p=2, dim=-1, keepdim=True)
            exp = torch.norm(self.offsets, p=2, dim=-1, keepdim=True)

            return -torch.abs(obs - exp.T)
        
        if self.similarity == 'neg-euclidean':
            return -torch.cdist(offsets, self.offsets)
        
        if self.similarity == 'inv-euclidean':
            return 1/(1 + torch.cdist(offsets, self.offsets))

        return self.similarity(offsets, self.offsets)

    def forward(self, x: Tensor, edge_index: Adj,
                edge_attr: OptTensor = None, pos: OptTensor = None) -> Tensor:
        if pos is None:
            pos = x

        # propagate_type: (x: Tensor, pos: Tensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, pos=pos, edge_attr=edge_attr)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j: Tensor, pos_i: Tensor, pos_j: Tensor, 
                edge_attr: OptTensor = None) -> Tensor:
        sim = self.pairwise_similarity(pos_j - pos_i)

        if math.isinf(self.temperature):
            if self.temperature > 0:
                idx = torch.argmax(sim, dim=-1)
            else:
                idx = torch.argmin(sim, dim=-1)

            W_j = self.weights[idx]
        else:
            alpha = torch.softmax(sim*self.temperature, dim=-1)
            W_j = alpha @ self.weights

        W_j = W_j.view(-1, self.groups, self.out_channels//self.groups,
                       self.in_channels//self.groups)
        x_j = x_j.view(-1, self.groups, self.in_channels//self.groups, 1)

        msg = (W_j @ x_j).view(-1, self.out_channels)

        if edge_attr is not None:
            msg = msg * edge_attr.view(-1, 1)
        
        return msg


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
