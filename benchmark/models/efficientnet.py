from typing import Callable, Tuple, Optional

import torch
from torch import Tensor
from torch.nn import Module, SiLU, Dropout, functional as F

from torch_geometric.nn import Sequential
from torch_geometric.data import InMemoryDataset
from torch_geometric.nn.glob import global_mean_pool
from torch_geometric.nn.dense import Linear
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.typing import OptTensor, Adj
from torch_geometric.nn.aggr import MultiAggregation, MaxAggregation, MeanAggregation

from gconv.conv import GenGraphConv
from gconv.pool import KMISPooling

from .baseline import Baseline


class StochasticDepth(Module):
    def __init__(self, p: float, mode: str = 'row'):
        super().__init__()

        self.p = p
        self.mode = mode

    def forward(self, x: Tensor, batch: OptTensor = None):
        if not self.training or self.p == 0.0:
            return x

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            batch_size = 1
        else:
            batch_size = batch.max() + 1

        survival_rate = 1.0 - self.p

        if self.mode == 'row':
            size = [batch_size] + [1] * (x.ndim - 1)
        else:
            size = [1] * x.ndim

        noise = torch.empty(size, dtype=x.dtype, device=x.device)
        noise = noise.bernoulli_(survival_rate)

        if survival_rate > 0.0:
            noise.div_(survival_rate)

        return x * noise[batch]


class SqueezeExcitation(Module):
    def __init__(self, in_channels: int,
                 ratio: float = 0.25,
                 activation: Callable[[Tensor], Tensor] = F.relu,
                 scale_activation: Callable[[Tensor], Tensor] = F.sigmoid):
        super().__init__()

        self.in_channels = in_channels
        self.squeeze_channels = int(in_channels*ratio)
        self.activation = activation
        self.scale_activation = scale_activation

        self.lin_s = Linear(self.in_channels, self.squeeze_channels)
        self.lin_e = Linear(self.squeeze_channels, self.in_channels)

    def forward(self, x: Tensor, batch: OptTensor) -> Tensor:
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        squeeze = global_mean_pool(x, batch)
        squeeze = self.activation(self.lin_s(squeeze))
        excite = self.scale_activation(self.lin_e(squeeze))
        return x*excite[batch]


class MobileBottleneckConv(Module):
    def __init__(self, in_channels: int,
                 out_channels: int = None,
                 fused: bool = False,
                 multiplier: int = 6,
                 stride: int = 1,
                 squeeze: Optional[float] = 0.25,
                 stochastic_depth: float = 0.0,
                 bias: bool = False,
                 **conv_kwargs):
        super().__init__()
        conv_kwargs['groups'] = 1 if fused else in_channels
        conv_kwargs['bias'] = bias
        conv_kwargs.setdefault('metric', 'cosine')

        bn_kwargs = dict(eps=0.001, momentum=0.1, affine=True)

        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.exp_channels = multiplier*in_channels
        self.fused = fused

        if fused:
            self.conv = GenGraphConv(self.in_channels, self.exp_channels, **conv_kwargs)
        else:
            self.exp_lin = Linear(in_channels, self.exp_channels, bias=bias)
            self.exp_norm = BatchNorm(self.exp_channels, **bn_kwargs)
            self.conv = GenGraphConv(self.exp_channels, **conv_kwargs)

        self.conv_norm = BatchNorm(self.exp_channels, **bn_kwargs)
        self.red_lin = Linear(self.exp_channels, out_channels, bias=bias)
        self.red_norm = BatchNorm(out_channels, **bn_kwargs)
        self.stochastic_depth = StochasticDepth(p=stochastic_depth)

        self.se = None
        self.pool = None

        if squeeze is not None:
            self.se = SqueezeExcitation(self.exp_channels, ratio=squeeze,
                                        activation=SiLU())

        if stride > 1:
            self.pool = KMISPooling(in_channels=multiplier*in_channels, k=stride - 1)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None,
                pos: OptTensor = None, batch: OptTensor = None) \
            -> Tuple[Tensor, Adj, OptTensor, OptTensor, OptTensor]:
        if self.fused:
            y = x
        else:
            y = self.exp_lin(x)
            y = self.exp_norm(y)
            y = F.silu(y)

        y = self.conv(y, edge_index, edge_attr, pos)

        if self.pool is not None:
            y, edge_index, edge_attr, pos, batch, mis, _ = \
                self.pool(y, edge_index, edge_attr, pos, batch)
            x = x[mis]

        y = self.conv_norm(y)
        y = F.silu(y)

        if self.se is not None:
            y = self.se(y, batch)

        y = self.red_lin(y)
        y = self.red_norm(y)
        y = self.stochastic_depth(y, batch)

        if self.out_channels != self.in_channels:
            x = F.pad(x, (0, self.out_channels - self.in_channels))

        return y + x, edge_index, edge_attr, pos, batch


class EfficientNetV2(Baseline):
    def __init__(self, dataset: InMemoryDataset,
                 delta_drop_probability: float = 0.005,
                 *args, **kwargs):
        super().__init__(dataset=dataset, *args, **kwargs)

        in_channels = dataset.num_features
        out_channels = dataset.num_classes
        pos_channels = None

        if dataset.pos is not None:
            pos_channels = dataset.pos.size(1)

        signature = 'x, e_i, e_w, p, b'
        c = 24

        bn_kwargs = dict(eps=0.001, momentum=0.1, affine=True)

        layers = [
            (GenGraphConv(in_channels=in_channels, out_channels=c,
                          pos_channels=pos_channels, bias=False), 'x, e_i, e_w, p -> x'),
            (KMISPooling(in_channels=c, k=1), f'{signature} -> {signature}, m, c'),
            (BatchNorm(in_channels=c, **bn_kwargs), 'x -> x'),
            (SiLU(True), 'x -> x'),
        ]

        def _p_gen():
            p = 0
            while p < 1:
                yield p
                p += delta_drop_probability

        drop_p_gen = _p_gen()
        mb_signature = f'{signature} -> {signature}'

        layers.extend([
            (MobileBottleneckConv(in_channels=c, out_channels=c,
                                  pos_channels=pos_channels,
                                  fused=True, multiplier=1, squeeze=None,
                                  stochastic_depth=next(drop_p_gen)), mb_signature),
            (MobileBottleneckConv(in_channels=c, out_channels=c,
                                  pos_channels=pos_channels,
                                  fused=True, multiplier=1, squeeze=None,
                                  stochastic_depth=next(drop_p_gen)), mb_signature),
        ]),

        for i in range(4):
            layers.append((MobileBottleneckConv(in_channels=c if i == 0 else c*2,
                                                pos_channels=pos_channels,
                                                out_channels=c*2, stride=2 if i == 0 else 1,
                                                fused=True, multiplier=4, squeeze=None,
                                                stochastic_depth=next(drop_p_gen)), mb_signature))

        for i in range(4):
            layers.append((MobileBottleneckConv(in_channels=c*2 if i == 0 else 64,
                                                pos_channels=pos_channels,
                                                out_channels=64, stride=2 if i == 0 else 1,
                                                fused=True, multiplier=4, squeeze=None,
                                                stochastic_depth=next(drop_p_gen)), mb_signature))

        for i in range(6):
            layers.append((MobileBottleneckConv(in_channels=64 if i == 0 else 128,
                                                pos_channels=pos_channels,
                                                out_channels=128, stride=2 if i == 0 else 1,
                                                fused=False, multiplier=4, squeeze=0.25,
                                                stochastic_depth=next(drop_p_gen)), mb_signature))

        for i in range(9):
            layers.append((MobileBottleneckConv(in_channels=128 if i == 0 else 160,
                                                pos_channels=pos_channels,
                                                out_channels=160, stride=1,
                                                fused=False, multiplier=6, squeeze=0.25,
                                                stochastic_depth=next(drop_p_gen)), mb_signature))

        for i in range(15):
            layers.append((MobileBottleneckConv(in_channels=160 if i == 0 else 256,
                                                pos_channels=pos_channels,
                                                out_channels=256, stride=2 if i == 0 else 1,
                                                fused=False, multiplier=6, squeeze=0.25,
                                                stochastic_depth=next(drop_p_gen)), mb_signature))

        layers.extend([
            (Linear(256, 1280, bias=False), 'x -> x'),
            (BatchNorm(in_channels=1280, **bn_kwargs), 'x -> x'),
            (SiLU(True), 'x -> x'),
            (global_mean_pool, 'x, b -> x'),
            (Dropout(p=0.2), 'x -> x'),
            (Linear(1280, out_channels)),
        ])

        self.model = Sequential(signature, layers)

    def forward(self, x=None, pos=None, edge_index=None, edge_attr=None, batch=None):
        return self.model(x, edge_index, edge_attr, pos, batch)


class CustomEfficientNetV2(Baseline):
    def __init__(self, dataset: InMemoryDataset,
                 delta_drop_probability: float = 0.005,
                 *args, **kwargs):
        super().__init__(dataset=dataset, *args, **kwargs)

        in_channels = dataset.num_features
        out_channels = dataset.num_classes
        pos_channels = None

        if dataset[0].pos is not None:
            pos_channels = dataset[0].pos.size(1)

        signature = 'x, e_i, e_w, p, b'
        c = 32

        bn_kwargs = dict(eps=0.001, momentum=0.1, affine=True)

        layers = [
            (GenGraphConv(in_channels=in_channels, out_channels=c,
                          pos_channels=pos_channels, bias=False), 'x, e_i, e_w, p -> x'),
            # (KMISPooling(in_channels=c, k=1), f'{signature} -> {signature}, m, c'),
            (BatchNorm(in_channels=c, **bn_kwargs), 'x -> x'),
            (SiLU(True), 'x -> x'),
        ]

        def _p_gen():
            p = 0
            while p < 1:
                yield p
                p += delta_drop_probability

        drop_p_gen = _p_gen()
        mb_signature = f'{signature} -> {signature}'

        layers.extend([
            (MobileBottleneckConv(in_channels=c, out_channels=c,
                                  pos_channels=pos_channels,
                                  fused=True, multiplier=1, squeeze=None,
                                  stochastic_depth=next(drop_p_gen)), mb_signature),
            (MobileBottleneckConv(in_channels=c, out_channels=c,
                                  pos_channels=pos_channels,
                                  fused=True, multiplier=1, squeeze=None,
                                  stochastic_depth=next(drop_p_gen)), mb_signature),
        ])

        for i in range(3):
            layers.append((MobileBottleneckConv(in_channels=c if i == 0 else c*2,
                                                pos_channels=pos_channels,
                                                out_channels=c*2, stride=2 if i == 0 else 1,
                                                fused=True, multiplier=4, squeeze=None,
                                                stochastic_depth=next(drop_p_gen)), mb_signature))

        # for i in range(4):
        #     layers.append((MobileBottleneckConv(in_channels=c*2 if i == 0 else 128,
        #                                         pos_channels=pos_channels,
        #                                         out_channels=128, stride=2 if i == 0 else 1,
        #                                         fused=True, multiplier=4, squeeze=None,
        #                                         stochastic_depth=next(drop_p_gen)), mb_signature))

        for i in range(6):
            layers.append((MobileBottleneckConv(in_channels=64 if i == 0 else 128,
                                                pos_channels=pos_channels,
                                                out_channels=128, stride=2 if i == 0 else 1,
                                                fused=False, multiplier=4, squeeze=0.25,
                                                stochastic_depth=next(drop_p_gen)), mb_signature))

        # for i in range(9):
        #     layers.append((MobileBottleneckConv(in_channels=128 if i == 0 else 160,
        #                                         pos_channels=pos_channels,
        #                                         out_channels=160, stride=1,
        #                                         fused=False, multiplier=6, squeeze=0.25,
        #                                         stochastic_depth=next(drop_p_gen)), mb_signature))

        for i in range(9):
            layers.append((MobileBottleneckConv(in_channels=128 if i == 0 else 256,
                                                pos_channels=pos_channels,
                                                out_channels=256, stride=2 if i == 0 else 1,
                                                fused=False, multiplier=6, squeeze=0.25,
                                                stochastic_depth=next(drop_p_gen)), mb_signature))

        layers.extend([
            (Linear(256, 512, bias=False), 'x -> x'),
            (BatchNorm(in_channels=512, **bn_kwargs), 'x -> x'),
            (SiLU(True), 'x -> x'),
            (MultiAggregation([
                MeanAggregation(), 
                MaxAggregation(), 
            ]), 'x, b -> x'),
            (Dropout(p=0.2), 'x -> x'),
            (Linear(1024, out_channels)),
        ])

        self.model = Sequential(signature, layers)

    def forward(self, x=None, pos=None, edge_index=None, edge_attr=None, batch=None):
        return self.model(x, edge_index, edge_attr, pos, batch)
