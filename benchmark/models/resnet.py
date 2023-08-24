from typing import Optional, Union, Type, Tuple, List

import torch
from torch import nn

from torchvision.models import ResNet, ResNet50_Weights, resnet50

from torch_geometric.nn import Sequential
from torch_geometric.nn.aggr import MeanAggregation
from torch_geometric.nn.dense import Linear
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.typing import Tensor, OptTensor, Adj

from gconv.conv import GenGraphConv, GenPointConv, BaseGenConv
from gconv.pool import KMISPooling
from gconv.utils import aggregate_k_hop, k_hop


class GenBottleneck(nn.Module):
    expansion: int = 4

    def __init__(self, in_channels: int,
                 hidden_channels: int,
                 stride: int = 1,
                 conv_cls: Type[BaseGenConv] = GenGraphConv,
                 **conv_kwargs) -> None:
        super().__init__()

        self.conv1 = Linear(in_channels, hidden_channels, bias=False)
        self.bn1 = BatchNorm(hidden_channels)
        self.conv2 = conv_cls(hidden_channels, hidden_channels, **conv_kwargs)
        self.pool = KMISPooling(k=stride - 1)
        self.bn2 = BatchNorm(hidden_channels)
        self.conv3 = Linear(hidden_channels, hidden_channels*self.expansion, bias=False)
        self.bn3 = BatchNorm(hidden_channels*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride != 1 or in_channels != hidden_channels*self.expansion:
            self.downsample = nn.Sequential(
                Linear(in_channels, hidden_channels*self.expansion, bias=False),
                BatchNorm(hidden_channels*self.expansion),
            )

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None,
                pos: OptTensor = None, batch: OptTensor = None) \
            -> Tuple[Tensor, Adj, OptTensor, OptTensor, OptTensor]:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if isinstance(self.conv2, GenGraphConv):
            out = self.conv2(out, edge_index, None, pos)
        else:
            out = self.conv2(out, pos, batch)

        if self.stride != 1:
            out, edge_index, edge_attr, pos, batch, mis, cluster = \
                self.pool(out, edge_index, edge_attr, pos, batch)
            pos /= self.stride
            x = x[mis]

        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x
        out = self.relu(out)

        return out, edge_index, edge_attr, pos, batch


class GenResNet(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 pos_channels: int,
                 layers: List[int] = None,
                 variant: str = 'graph',
                 **conv_kwargs) -> None:
        super().__init__()
        conv_cls = GenGraphConv if variant == 'graph' else GenPointConv

        if layers is None:
            layers = [3, 4, 6, 3]  # ResNet50

        conv_kwargs['pos_channels'] = pos_channels or in_channels
        conv_kwargs['conv_cls'] = conv_cls
        conv_kwargs.setdefault('bias', False)
        c = 64

        self.conv1 = conv_cls(in_channels, c, offsets=list(range(-3, 4)), **conv_kwargs)
        conv_kwargs.setdefault('offsets', [-1, 0, 1])
        self.pool = KMISPooling(k=1)

        self.bn1 = BatchNorm(c)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(c, 64, layers[0], **conv_kwargs)
        self.layer2 = self._make_layer(64*GenBottleneck.expansion, 128, layers[1], stride=2, **conv_kwargs)
        self.layer3 = self._make_layer(128*GenBottleneck.expansion, 256, layers[2], stride=2, **conv_kwargs)
        self.layer4 = self._make_layer(256*GenBottleneck.expansion, 512, layers[3], stride=2, **conv_kwargs)
        self.avgpool = MeanAggregation()
        self.fc = nn.Linear(512 * GenBottleneck.expansion, out_channels)

    @staticmethod
    def _make_layer(in_channels: int, hidden_channels: int, num_blocks: int,
                    stride: int = 1, **conv_kwargs) -> nn.Module:
        sig = 'x, e_i, e_w, p, b'
        layers = [
            (GenBottleneck(in_channels, hidden_channels, stride, **conv_kwargs), f'{sig} -> {sig}')
        ]
        in_channels = hidden_channels*GenBottleneck.expansion

        for _ in range(1, num_blocks):
            layers.append(
                (GenBottleneck(in_channels, hidden_channels, **conv_kwargs), f'{sig} -> {sig}')
            )

        return Sequential(sig, layers)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None,
                pos: OptTensor = None, batch: OptTensor = None) -> Tensor:
        k_hop_index, _ = k_hop(edge_index, None, k=3, num_nodes=x.size(0))

        if isinstance(self.conv1, GenGraphConv):
            x = self.conv1(x, k_hop_index, None, pos)
        else:
            x = self.conv1(x, pos, batch)

        x, edge_index, edge_attr, pos, batch, mis, cluster = \
            self.pool(x, edge_index, edge_attr, pos, batch)

        x = self.bn1(x)
        x = self.relu(x)

        x = aggregate_k_hop(x, edge_index, edge_attr, k=1, reduce='max')
        x, edge_index, edge_attr, pos, batch, mis, cluster = \
            self.pool(x, edge_index, edge_attr, pos, batch)

        x, edge_index, edge_attr, pos, batch = self.layer1(x, edge_index, edge_attr, pos, batch)
        x, edge_index, edge_attr, pos, batch = self.layer2(x, edge_index, edge_attr, pos, batch)
        x, edge_index, edge_attr, pos, batch = self.layer3(x, edge_index, edge_attr, pos, batch)
        x, edge_index, edge_attr, pos, batch = self.layer4(x, edge_index, edge_attr, pos, batch)

        x = self.avgpool(x, batch)
        x = self.fc(x)

        return x

    @torch.no_grad()
    def from_original_model(self, model: ResNet):
        state_dict = BaseGenConv.from_regular_conv(model.conv1).state_dict()
        state_dict['temperature'] = self.conv1.state_dict()['temperature']
        self.conv1.load_state_dict(state_dict)
        self.bn1.module.load_state_dict(model.bn1.state_dict())

        gen_layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        reg_layers = [model.layer1, model.layer2, model.layer3, model.layer4]

        for gen_layer, reg_layer in zip(gen_layers, reg_layers):
            for gen_bn, reg_bn in zip(gen_layer, reg_layer):
                state_dict = reg_bn.conv1.state_dict()
                state_dict['weight'] = state_dict['weight'].squeeze(-1, -2)
                gen_bn.conv1.load_state_dict(state_dict)
                gen_bn.bn1.module.load_state_dict(reg_bn.bn1.state_dict())

                state_dict = BaseGenConv.from_regular_conv(reg_bn.conv2).state_dict()
                state_dict['temperature'] = gen_bn.conv2.state_dict()['temperature']
                gen_bn.conv2.load_state_dict(state_dict)
                gen_bn.bn2.module.load_state_dict(reg_bn.bn2.state_dict())

                state_dict = reg_bn.conv3.state_dict()
                state_dict['weight'] = state_dict['weight'].squeeze(-1, -2)
                gen_bn.conv3.load_state_dict(state_dict)
                gen_bn.bn3.module.load_state_dict(reg_bn.bn3.state_dict())

                if gen_bn.downsample is not None:
                    state_dict = reg_bn.downsample[0].state_dict()
                    state_dict['weight'] = state_dict['weight'].squeeze(-1, -2)
                    gen_bn.downsample[0].load_state_dict(state_dict)
                    gen_bn.downsample[1].module.load_state_dict(reg_bn.downsample[1].state_dict())

        self.fc.load_state_dict(model.fc.state_dict())


def gen_resnet50(weights: Optional[Union[str, ResNet50_Weights]] = None,
                 progress: bool = True, config: Optional[dict] = None, **kwargs):
    config = config or {}
    mobilenet = resnet50(weights=weights, progress=progress, **config)

    kwargs.setdefault('in_channels', 3)
    kwargs.setdefault('pos_channels', 2)
    kwargs.setdefault('out_channels', 1000)
    model = GenResNet(**kwargs)
    model.from_original_model(mobilenet)
    return model
