import math
from typing import Union, Callable
import warnings

import torch
from torch import Tensor
from torch.nn import functional as F

from torch_geometric.nn.pool import knn_graph
from torch_geometric.typing import OptTensor, Adj

from .base_conv import BaseGenConv

Similarity = Callable[[Tensor, Tensor], Tensor]


class GenGraphConv(BaseGenConv):
    def __init__(self, *args,
                 similarity: Union[str, Similarity] = 'neg-euclidean',
                 **kwargs):
        super(GenGraphConv, self).__init__(*args, **kwargs)
        self.similarity = similarity

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
            return 1 / (1 + torch.cdist(offsets, self.offsets))

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

    def message(self, x_j: Tensor, pos_i: Tensor, pos_j: Tensor,  # noqa
                edge_attr: OptTensor = None) -> Tensor:
        sim = self.pairwise_similarity(pos_j - pos_i)

        if math.isinf(self.temperature):
            if self.temperature > 0:
                idx = torch.argmax(sim, dim=-1)
            else:
                idx = torch.argmin(sim, dim=-1)

            W_j = self.weights[idx]
        else:
            alpha = torch.softmax(sim * self.temperature, dim=-1)
            W_j = alpha @ self.weights

        W_j = W_j.view(-1, self.groups, self.out_channels // self.groups,
                       self.in_channels // self.groups)
        x_j = x_j.view(-1, self.groups, self.in_channels // self.groups, 1)

        msg = (W_j @ x_j).view(-1, self.out_channels)

        if edge_attr is not None:
            if edge_attr.dim() == 1 or edge_attr.size(1) == 1:
                msg = msg * edge_attr.view(-1, 1)
            else:
                warnings.warn("Ignoring `edge_attr` as it has more than 1 channel.")

        return msg


class DynamicGenConv(GenGraphConv):
    def __init__(self, *args,
                 k: int = 20,
                 loop: bool = True,
                 **kwargs):
        super(GenGraphConv).__init__(*args, **kwargs)
        self.k = k
        self.loop = loop

    def forward(self, x: Tensor, pos: OptTensor = None,  # noqa
                batch: OptTensor = None) -> Tensor:
        if pos is None:
            pos = x

        edge_index = knn_graph(pos, k=self.k, batch=batch, loop=self.loop)
        return super().forward(x, edge_index, None, pos)
