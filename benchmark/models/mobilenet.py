from typing import Tuple

from .baseline import Baseline

import torch
from torch import Tensor
from torch.nn import Module, functional as F

from gconv.conv import GenGraphConv
from gconv.pool import KMISPooling

from torch_geometric.data import InMemoryDataset
from torch_geometric.nn import Sequential
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn.dense import Linear
from torch_geometric.nn.pool.glob import global_mean_pool
from torch_geometric.typing import Adj, OptTensor


class InvertedResidualBlock(Module):
    def __init__(self, in_channels: int,
                 out_channels: int = None,
                 stride: int = 1,
                 multiplier: int = 6,
                 **conv_kwargs):
        super().__init__()
        conv_kwargs['depthwise'] = True

        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.stride = stride

        self.exp_lin = Linear(in_channels, multiplier*in_channels, bias=False)
        self.exp_norm = BatchNorm(multiplier*in_channels)
        self.conv = GenGraphConv(multiplier*in_channels, **conv_kwargs)
        self.conv_norm = BatchNorm(multiplier*in_channels)
        self.red_lin = Linear(multiplier*in_channels, out_channels, bias=False)
        self.red_norm = BatchNorm(out_channels)
        self.pool = None

        if stride > 1:
            self.pool = KMISPooling(in_channels=multiplier*in_channels, k=stride - 1)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None,
                pos: OptTensor = None, batch: OptTensor = None) \
            -> Tuple[Tensor, Adj, OptTensor, OptTensor, OptTensor]:
        y = self.exp_lin(x)
        y = self.exp_norm(y)
        y = F.relu6(y)
        y = self.conv(y, edge_index, edge_attr, pos)

        if self.stride > 1:
            y, edge_index, edge_attr, pos, batch, _, _ = \
                self.pool(y, edge_index, edge_attr, pos, batch)

        y = self.conv_norm(y)
        y = F.relu6(y)
        y = self.red_lin(y)
        y = self.red_norm(y)

        if self.stride == 1 and self.out_channels == self.in_channels:
            y = y + x

        return y, edge_index, edge_attr, pos, batch


class MobileNetV2(Baseline):
    def __init__(self, dataset: InMemoryDataset,
                 *args, **kwargs):
        super().__init__(dataset=dataset, *args, **kwargs)
        in_channels = dataset.num_node_features
        out_channels = dataset.num_classes
        pos_channels = None

        if dataset[0].pos is not None:
            pos_channels = dataset[0].pos.size(1)

        c = 32

        self.conv = GenGraphConv(in_channels=in_channels, out_channels=c,
                                 pos_channels=pos_channels)
        self.conv_norm = BatchNorm(c)
        self.pool = KMISPooling(in_channels=c, k=1)
        signature = 'x, e_i, e_w, pos, b -> x, e_i, e_w, pos, b'

        self.model = Sequential('x, e_i, e_w, pos, b', [
            (InvertedResidualBlock(in_channels=c, out_channels=c//2,
                                   multiplier=1, pos_channels=pos_channels), signature),
            (InvertedResidualBlock(in_channels=c//2, out_channels=3*c//4,
                                   stride=2, pos_channels=pos_channels), signature),
            (InvertedResidualBlock(in_channels=3*c//4, out_channels=3*c//4,
                                   pos_channels=pos_channels), signature),
            (InvertedResidualBlock(in_channels=3*c//4, out_channels=c,
                                   stride=2, pos_channels=pos_channels), signature),
            (InvertedResidualBlock(in_channels=c, out_channels=c,
                                   pos_channels=pos_channels), signature),
            (InvertedResidualBlock(in_channels=c, out_channels=c,
                                   pos_channels=pos_channels), signature),
            (InvertedResidualBlock(in_channels=c, out_channels=2*c,
                                   pos_channels=pos_channels, stride=2), signature),
            (InvertedResidualBlock(in_channels=2*c, out_channels=2*c,
                                   pos_channels=pos_channels), signature),
            (InvertedResidualBlock(in_channels=2*c, out_channels=2*c,
                                   pos_channels=pos_channels), signature),
            (InvertedResidualBlock(in_channels=2*c, out_channels=2*c,
                                   pos_channels=pos_channels), signature),
            (InvertedResidualBlock(in_channels=2*c, out_channels=3*c,
                                   pos_channels=pos_channels), signature),
            (InvertedResidualBlock(in_channels=3*c, out_channels=3*c,
                                   pos_channels=pos_channels), signature),
            (InvertedResidualBlock(in_channels=3*c, out_channels=3*c,
                                   pos_channels=pos_channels), signature),
            (InvertedResidualBlock(in_channels=3*c, out_channels=5*c,
                                   stride=2, pos_channels=pos_channels), signature),
            (InvertedResidualBlock(in_channels=5*c, out_channels=5*c,
                                   pos_channels=pos_channels), signature),
            (InvertedResidualBlock(in_channels=5*c, out_channels=5*c,
                                   pos_channels=pos_channels), signature),
            (InvertedResidualBlock(in_channels=5*c, out_channels=10*c,
                                   pos_channels=pos_channels), signature),
        ])

        self.lin = Linear(in_channels=10*c, out_channels=40*c, bias=False)
        self.lin_norm = BatchNorm(40*c)
        self.out = Linear(in_channels=40*c, out_channels=out_channels)

    def forward(self, x=None, pos=None, edge_index=None, edge_attr=None, batch=None):
        x = self.conv.forward(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)
        x, edge_index, edge_attr, pos, batch, _, _ = \
            self.pool(x, edge_index, edge_attr, pos, batch)

        x = self.conv_norm(x)
        x = F.relu6(x)

        x, edge_index, edge_attr, pos, batch = self.model(x, edge_index, edge_attr, pos, batch)
        x = self.lin(x)
        x = global_mean_pool(x, batch)
        x = self.lin_norm(x)

        return self.out(x)
