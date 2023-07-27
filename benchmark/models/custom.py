import torch
from torch.nn import LeakyReLU, Sequential, Dropout, Module, SiLU

from torch_geometric.nn import global_max_pool, global_mean_pool
from torch_geometric.nn import Linear, BatchNorm, DynamicEdgeConv, Sequential as PyGSeq
from torch_geometric.data import InMemoryDataset
from torch_geometric.nn import knn_graph
from torch_geometric.nn.aggr import MultiAggregation, SumAggregation, MeanAggregation, MaxAggregation
from torch_geometric.typing import Tensor, Adj, OptTensor

from .baseline import Baseline
from gconv.conv import GenConv, DynamicGenConv
from gconv.pool import KMISPooling


class ConvBlock(Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 *args, **kwargs):
        super().__init__()
        self.module = PyGSeq('x, e_i, e_w, p', [
            (GenConv(in_channels=in_channels, out_channels=out_channels,
                     *args, **kwargs), 'x, e_i, e_w, p -> x'),
            (BatchNorm(out_channels), 'x -> x'),
            (SiLU(), 'x -> x'),
            (Linear(out_channels, out_channels, bias=False), 'x -> x'),
            (BatchNorm(out_channels), 'x -> x'),
            (SiLU(), 'x -> x'),
        ])
    
    def forward(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None, 
                pos: OptTensor = None) -> Tensor:
        return self.module(x, edge_index, edge_attr, pos)


class CustomGNN(Baseline):
    def __init__(self, dataset: InMemoryDataset,
                 *args, **kwargs):
        super().__init__(dataset=dataset, *args, **kwargs)
        
        in_channels = dataset.num_features
        out_channels = dataset.num_classes
        pos_channels = None

        if dataset.data.pos is not None:
            pos_channels = dataset.data.pos.size(1)

        c = 32
        emb_dim = 1024
        conv_kwargs = dict(bias=False, aggr='add', metric='cosine', 
                           temperature='same', pos_channels=pos_channels)
        
        self.aggr = MultiAggregation([SumAggregation(), MaxAggregation()])
        self.pool = KMISPooling(aggr_x=self.aggr, aggr_pos=MeanAggregation())

        self.conv1 = ConvBlock(in_channels=in_channels, out_channels=c, **conv_kwargs)
        self.conv2 = ConvBlock(in_channels=c*2, out_channels=c, **conv_kwargs)
        self.conv3 = ConvBlock(in_channels=c*2, out_channels=c*2, **conv_kwargs)
        self.conv4 = ConvBlock(in_channels=c*4, out_channels=c*4, **conv_kwargs)
        self.conv5 = ConvBlock(in_channels=c*8, out_channels=c*8, **conv_kwargs)
        
        self.mlp = Sequential(
            Linear(c*32, emb_dim, bias=False),
            BatchNorm(emb_dim),
            SiLU(),
            Dropout(0.3),
            Linear(emb_dim, emb_dim//2),
            BatchNorm(emb_dim//2),
            SiLU(),
            Dropout(0.3),
            Linear(emb_dim//2, out_channels),
        )

    def forward(self, x=None, pos=None, edge_index=None, edge_attr=None, batch=None):
        x = self.conv1(x, edge_index, edge_attr, pos)
        x1 = self.aggr(x, batch)

        x, edge_index, edge_attr, pos, batch, _, _ = self.pool(x, edge_index, edge_attr, pos, batch)
        x = self.conv2(x, edge_index, edge_attr, pos)
        x2 = self.aggr(x, batch)
        
        x, edge_index, edge_attr, pos, batch, _, _ = self.pool(x, edge_index, edge_attr, pos, batch)
        x = self.conv3(x, edge_index, edge_attr, pos)
        x3 = self.aggr(x, batch)

        x, edge_index, edge_attr, pos, batch, _, _ = self.pool(x, edge_index, edge_attr, pos, batch)
        x = self.conv4(x, edge_index, edge_attr, pos)
        x4 = self.aggr(x, batch)

        x, edge_index, edge_attr, pos, batch, _, _ = self.pool(x, edge_index, edge_attr, pos, batch)
        x = self.conv5(x, edge_index, edge_attr, pos)
        x5 = self.aggr(x, batch)

        x = torch.cat([x1, x2, x3, x4, x5], dim=-1)

        return self.mlp(x)