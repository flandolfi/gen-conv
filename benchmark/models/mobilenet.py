from typing import Tuple, Type

import torch
from torch import Tensor
from torch.nn import Module, functional as F

from torchvision.models import MobileNetV2

from gconv.conv import GenGraphConv, GenPointConv, BaseGenConv
from gconv.pool import KMISPooling

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
                 conv_cls: Type[BaseGenConv] = GenGraphConv,
                 **conv_kwargs):
        super().__init__()
        conv_kwargs['groups'] = multiplier*in_channels
        conv_kwargs.setdefault('similarity', 'neg-euclidean')
        conv_kwargs.setdefault('bias', False)

        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.expand = multiplier > 1
        self.stride = stride
        self.residual = stride == 1 and out_channels == in_channels

        if self.expand:
            self.exp_lin = Linear(in_channels, multiplier*in_channels, bias=False)
            self.exp_norm = BatchNorm(multiplier*in_channels)

        self.conv = conv_cls(multiplier*in_channels, **conv_kwargs)
        self.conv_norm = BatchNorm(multiplier*in_channels)
        self.red_lin = Linear(multiplier*in_channels, out_channels, bias=False)
        self.red_norm = BatchNorm(out_channels)
        self.pool = None

        if stride > 1:
            self.pool = KMISPooling(in_channels=multiplier*in_channels, k=stride - 1,
                                    scorer='lexicographic', score_heuristic=None,
                                    score_random_on_train=False)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None,
                pos: OptTensor = None, batch: OptTensor = None) \
            -> Tuple[Tensor, Adj, OptTensor, OptTensor, OptTensor]:
        y = x

        if self.expand:
            y = self.exp_lin(y)
            y = self.exp_norm(y)
            y = F.relu6(y)

        if isinstance(self.conv, GenGraphConv):
            y = self.conv(y, edge_index, None, pos)
        else:
            y = self.conv(y, pos, batch)

        if self.stride > 1:
            y, edge_index, edge_attr, pos, batch, _, _ = \
                self.pool(y, edge_index, edge_attr, pos, batch)

            pos = pos / self.stride

        y = self.conv_norm(y)
        y = F.relu6(y)
        y = self.red_lin(y)
        y = self.red_norm(y)

        if self.residual:
            y = y + x

        return y, edge_index, None, pos, batch


class GenMobileNetV2(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 pos_channels: int = None,
                 variant: str = 'graph',
                 **conv_kwargs):
        super().__init__()
        conv_cls = GenGraphConv if variant == 'graph' else GenPointConv

        conv_kwargs['pos_channels'] = pos_channels or in_channels
        conv_kwargs['conv_cls'] = conv_cls
        conv_kwargs.setdefault('bias', False)
        conv_kwargs.setdefault('offsets', [-1, 0, 1])
        c = 32

        self.conv = conv_cls(in_channels=in_channels, out_channels=c, **conv_kwargs)
        self.conv_norm = BatchNorm(c)
        self.pool = KMISPooling(in_channels=c, k=1, scorer='lexicographic',
                                score_heuristic=None, score_random_on_train=False)
        signature = 'x, e_i, e_w, pos, b -> x, e_i, e_w, pos, b'

        self.model = Sequential('x, e_i, e_w, pos, b', [
            (InvertedResidualBlock(in_channels=c, out_channels=c//2, multiplier=1, **conv_kwargs), signature),
            (InvertedResidualBlock(in_channels=c//2, out_channels=3*c//4, stride=2, **conv_kwargs), signature),
            (InvertedResidualBlock(in_channels=3*c//4, out_channels=3*c//4, **conv_kwargs), signature),
            (InvertedResidualBlock(in_channels=3*c//4, out_channels=c, stride=2, **conv_kwargs), signature),
            (InvertedResidualBlock(in_channels=c, out_channels=c, **conv_kwargs), signature),
            (InvertedResidualBlock(in_channels=c, out_channels=c, **conv_kwargs), signature),
            (InvertedResidualBlock(in_channels=c, out_channels=2*c, **conv_kwargs, stride=2), signature),
            (InvertedResidualBlock(in_channels=2*c, out_channels=2*c, **conv_kwargs), signature),
            (InvertedResidualBlock(in_channels=2*c, out_channels=2*c, **conv_kwargs), signature),
            (InvertedResidualBlock(in_channels=2*c, out_channels=2*c, **conv_kwargs), signature),
            (InvertedResidualBlock(in_channels=2*c, out_channels=3*c, **conv_kwargs), signature),
            (InvertedResidualBlock(in_channels=3*c, out_channels=3*c, **conv_kwargs), signature),
            (InvertedResidualBlock(in_channels=3*c, out_channels=3*c, **conv_kwargs), signature),
            (InvertedResidualBlock(in_channels=3*c, out_channels=5*c, stride=2, **conv_kwargs), signature),
            (InvertedResidualBlock(in_channels=5*c, out_channels=5*c, **conv_kwargs), signature),
            (InvertedResidualBlock(in_channels=5*c, out_channels=5*c, **conv_kwargs), signature),
            (InvertedResidualBlock(in_channels=5*c, out_channels=10*c, **conv_kwargs), signature),
        ])

        self.lin = Linear(in_channels=10*c, out_channels=40*c, bias=False)
        self.lin_norm = BatchNorm(40*c)
        self.out = Linear(in_channels=40*c, out_channels=out_channels)

    def forward(self, x=None, pos=None, edge_index=None, edge_attr=None, batch=None):
        if isinstance(self.conv, GenGraphConv):
            x = self.conv(x=x, edge_index=edge_index, pos=pos)
        else:
            x = self.conv(x=x, pos=pos, batch=batch)

        x, edge_index, edge_attr, pos, batch, _, _ = \
            self.pool(x, edge_index, edge_attr, pos, batch)
        pos = pos / 2

        x = self.conv_norm(x)
        x = F.relu6(x)

        x, edge_index, edge_attr, pos, batch = self.model(x, edge_index, None, pos, batch)
        x = self.lin(x)
        x = self.lin_norm(x)
        x = F.relu6(x)
        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=0.2, training=self.training)
        return self.out(x)

    @torch.no_grad()
    def from_original_model(self, model: MobileNetV2):
        state_dict = GenGraphConv.from_regular_conv(model.features[0][0]).state_dict()
        state_dict['temperature'] = self.conv.state_dict()['temperature']
        self.conv.load_state_dict(state_dict)
        self.conv_norm.module.load_state_dict(model.features[0][1].state_dict())

        for gen_block, block in zip(self.model, model.features[1:]):  # noqa
            if len(block.conv) == 4:
                state_dict = block.conv[0][0].state_dict()
                state_dict['weight'] = state_dict['weight'].squeeze(-1, -2)
                gen_block.exp_lin.load_state_dict(state_dict)
                gen_block.exp_norm.module.load_state_dict(block.conv[0][1].state_dict())

            state_dict = GenGraphConv.from_regular_conv(block.conv[-3][0]).state_dict()
            state_dict['temperature'] = gen_block.conv.state_dict()['temperature']
            gen_block.conv.load_state_dict(state_dict)
            gen_block.conv_norm.module.load_state_dict(block.conv[-3][1].state_dict())

            state_dict = block.conv[-2].state_dict()
            state_dict['weight'] = state_dict['weight'].squeeze(-1, -2)
            gen_block.red_lin.load_state_dict(state_dict)

            gen_block.red_norm.module.load_state_dict(block.conv[-1].state_dict())

        last = model.features[-1]
        state_dict = last[0].state_dict()
        state_dict['weight'] = state_dict['weight'].squeeze(-1, -2)
        self.lin.load_state_dict(state_dict)
        self.lin_norm.module.load_state_dict(last[1].state_dict())
        self.out.load_state_dict(model.classifier[1].state_dict())
