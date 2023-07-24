import torch
from torch.nn import LeakyReLU, Sequential, Dropout

from torch_geometric.nn import global_max_pool, global_mean_pool
from torch_geometric.nn import Linear, BatchNorm, DynamicEdgeConv, Sequential as PyGSeq
from torch_geometric.data import InMemoryDataset
from torch_geometric.nn import knn_graph
from torch_geometric.nn.aggr import MultiAggregation, MeanAggregation, MaxAggregation

from .baseline import Baseline
from gconv.conv import GenConv, DynamicGenConv
from gconv.pool import KMISPooling


class DGCNN(Baseline):
    def __init__(self, dataset: InMemoryDataset,
                 k: int = 20,
                 *args, **kwargs):
        super().__init__(dataset=dataset, *args, **kwargs)

        self.k = k
        
        in_channels = dataset.num_features
        out_channels = dataset.num_classes

        c = 64
        emb_dim = 1024

        self.conv1 = DynamicEdgeConv(Sequential(Linear(in_channels=in_channels*2, out_channels=c, bias=False),
                                         BatchNorm(c), LeakyReLU(0.2)), self.k)
        self.conv2 = DynamicEdgeConv(Sequential(Linear(in_channels=c*2, out_channels=c, bias=False),
                                         BatchNorm(c), LeakyReLU(0.2)), self.k)
        self.conv3 = DynamicEdgeConv(Sequential(Linear(in_channels=c*2, out_channels=c*2, bias=False),
                                         BatchNorm(c*2), LeakyReLU(0.2)), self.k)
        self.conv4 = DynamicEdgeConv(Sequential(Linear(in_channels=c*4, out_channels=c*4, bias=False),
                                         BatchNorm(c*4), LeakyReLU(0.2)), self.k)
        self.conv5 = Sequential(Linear(c + c + c*2 + c*4, emb_dim, bias=False), BatchNorm(emb_dim), LeakyReLU(0.2))
        self.mlp = Sequential(
            Linear(emb_dim*2, emb_dim//2, bias=False),
            BatchNorm(emb_dim//2),
            LeakyReLU(0.2),
            Dropout(),
            Linear(emb_dim//2, emb_dim//4),
            BatchNorm(emb_dim//4),
            LeakyReLU(0.2),
            Dropout(),
            Linear(emb_dim//4, out_channels),
        )

    def forward(self, x=None, pos=None, edge_index=None, edge_attr=None, batch=None):
        x1 = self.conv1(x, batch)
        x2 = self.conv2(x1, batch)
        x3 = self.conv3(x2, batch)
        x4 = self.conv4(x3, batch)

        x = torch.cat((x1, x2, x3, x4), dim=-1)
        x = self.conv5(x)

        x1 = global_max_pool(x, batch)
        x2 = global_mean_pool(x, batch)
        x = torch.cat((x1, x2), dim=-1)

        return self.mlp(x)


class CustomDGCNN(Baseline):
    def __init__(self, dataset: InMemoryDataset,
                 *args, **kwargs):
        super().__init__(dataset=dataset, *args, **kwargs)
        
        in_channels = dataset.num_features
        out_channels = dataset.num_classes
        pos_channels = None

        if dataset.data.pos is not None:
            pos_channels = dataset.data.pos.size(1)

        c = 32
        emb_dim = 512
        conv_kwargs = dict(bias=True, aggr='add', metric='cosine', temperature='same', pos_channels=pos_channels)

        self.conv1 = PyGSeq('x, e_i, e_w, p', [
            (GenConv(in_channels=in_channels, out_channels=c, **conv_kwargs), 'x, e_i, e_w, p -> x'),
            (BatchNorm(c), 'x -> x'),
            (LeakyReLU(0.2), 'x -> x'),
        ])
        self.conv2 = PyGSeq('x, e_i, e_w, p', [
            (GenConv(in_channels=c, out_channels=c, **conv_kwargs), 'x, e_i, e_w, p -> x'),
            (BatchNorm(c), 'x -> x'),
            (LeakyReLU(0.2), 'x -> x'),
        ])
        self.conv3 = PyGSeq('x, e_i, e_w, p', [
            (GenConv(in_channels=c, out_channels=c*2, **conv_kwargs), 'x, e_i, e_w, p -> x'),
            (BatchNorm(c*2), 'x -> x'),
            (LeakyReLU(0.2), 'x -> x'),
        ])
        self.conv4 = PyGSeq('x, e_i, e_w, p', [
            (GenConv(in_channels=c*2, out_channels=c*4, **conv_kwargs), 'x, e_i, e_w, p -> x'),
            (BatchNorm(c*4), 'x -> x'),
            (LeakyReLU(0.2), 'x -> x'),
        ])
        self.conv5 = Sequential(Linear(c + c + c*2 + c*4, emb_dim, bias=False), BatchNorm(emb_dim), LeakyReLU(0.2))
        
        self.mlp = Sequential(
            Linear(emb_dim*2, emb_dim//2, bias=False),
            BatchNorm(emb_dim//2),
            LeakyReLU(0.2),
            Dropout(),
            Linear(emb_dim//2, emb_dim//4),
            BatchNorm(emb_dim//4),
            LeakyReLU(0.2),
            Dropout(),
            Linear(emb_dim//4, out_channels),
        )

    def forward(self, x=None, pos=None, edge_index=None, edge_attr=None, batch=None):
        x1 = self.conv1(x, edge_index, edge_attr, pos)
        x2 = self.conv2(x1, edge_index, edge_attr, pos)
        x3 = self.conv3(x2, edge_index, edge_attr, pos)
        x4 = self.conv4(x3, edge_index, edge_attr, pos)

        x = torch.cat((x1, x2, x3, x4), dim=-1)
        x = self.conv5(x)

        x1 = global_max_pool(x, batch)
        x2 = global_mean_pool(x, batch)
        x = torch.cat((x1, x2), dim=-1)

        return self.mlp(x)