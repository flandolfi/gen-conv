from typing import Tuple

from .baseline import Baseline

import torch
from torch import Tensor
from torch.nn import Module, functional as F, ReLU

from gconv.conv import GenGraphConv
from gconv.pool import KMISPooling

from torch_geometric.data import InMemoryDataset
from torch_geometric.nn import Sequential
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn.dense import Linear
from torch_geometric.nn.pool.glob import global_max_pool
from torch_geometric.typing import Adj, OptTensor


class VGGBlock(Module):
    def __init__(self, in_channels: int,
                 out_channels: int = None,
                 pos_channels: int = None,
                 num_layers: int = 2,
                 batch_norm: bool = True,
                 **conv_kwargs):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels

        layers = []

        for _ in range(num_layers):
            layers.append((GenGraphConv(in_channels=in_channels, out_channels=out_channels,
                                        pos_channels=pos_channels, **conv_kwargs),
                           'x, e_i, e_w, p -> x'))

            if batch_norm:
                layers.append((BatchNorm(out_channels), 'x -> x'))
            
            layers.append((ReLU(), 'x -> x'))
            in_channels = out_channels

        self.convs = Sequential('x, e_i, e_w, p', layers)
        self.pool = KMISPooling(in_channels=out_channels, k=1, aggr_x='max')

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None,
                pos: OptTensor = None, batch: OptTensor = None) \
            -> Tuple[Tensor, Adj, OptTensor, OptTensor, OptTensor]:
        x = self.convs(x, edge_index, edge_attr, pos)
        x, edge_index, edge_attr, pos, batch, _, _ = \
            self.pool(x, edge_index, edge_attr, pos, batch)

        return x, edge_index, edge_attr, pos, batch


class VGG16(Baseline):
    def __init__(self, dataset: InMemoryDataset,
                 *args, **kwargs):
        super().__init__(dataset=dataset, *args, **kwargs)
        in_channels = dataset.num_node_features
        out_channels = dataset.num_classes
        pos_channels = None

        if dataset[0].pos is not None:
            pos_channels = dataset[0].pos.size(1)

        c = 64
        signature = 'x, e_i, e_w, pos, b -> x, e_i, e_w, pos, b'

        self.model = Sequential('x, e_i, e_w, pos, b', [
            (VGGBlock(in_channels=in_channels, out_channels=c, 
                      pos_channels=pos_channels, num_layers=2), signature),
            (VGGBlock(in_channels=c, out_channels=2*c, 
                      pos_channels=pos_channels, num_layers=2), signature),
            (VGGBlock(in_channels=2*c, out_channels=4*c, 
                      pos_channels=pos_channels, num_layers=3), signature),
            (VGGBlock(in_channels=4*c, out_channels=8*c, 
                      pos_channels=pos_channels, num_layers=3), signature),
            (VGGBlock(in_channels=8*c, out_channels=8*c, 
                      pos_channels=pos_channels, num_layers=3), signature),
        ])

        self.lin_1 = Linear(in_channels=8*c, out_channels=64*c, bias=False)
        self.norm_1 = BatchNorm(64*c)
        self.lin_2 = Linear(in_channels=64*c, out_channels=1000, bias=False)
        self.norm_2 = BatchNorm(1000)
        self.out = Linear(in_channels=1000, out_channels=out_channels)

    def forward(self, x=None, pos=None, edge_index=None, edge_attr=None, batch=None):
        x, edge_index, edge_attr, pos, batch = self.model(x, edge_index, edge_attr, pos, batch)
        x = global_max_pool(x, batch)
        x = self.lin_1(x)
        x = self.norm_1(x)
        x = F.relu(x)
        x = self.lin_2(x)
        x = self.norm_2(x)
        x = F.relu(x)

        return self.out(x)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, 20, 2)
        return [opt], [sch]
